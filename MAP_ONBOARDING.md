# Adding a map for known-map localization

How a new deploy map goes from a top-down image to `deploy_run.py --map <name>`. Dark Passage went
through this as K1-K5 (archive/KNOWN_MAP_LOCALIZATION_PLAN.md); this is the same route with the
one-off work taken out. Crystal Eye Castle was the first map through it.

Since 2026-11-04 the target is the live game's own Showdown map, which rotates every 24 hours.
A map therefore gets about one day of work, and comes back later in the rotation. Starfruit
Supernova replaced Crystal Eye Castle before Crystal Eye Castle's recordings were made (section 4).

## 0. What a map needs, and what it does not

- **The label is the data.** The localizer (`brawl_deployment/perception/localize.py`) is not
  trained per map. It scores what the terrain classifier sees against the hand label, so a map is
  ready when its label is right and the classifier can read its palette.
- **The terrain classifier is shared by every map** (`brawl_vision/data/terrain.pt`). A new map
  needs new classifier labels only if its palette defeats it. Step M3 measures that before anyone
  labels a frame. Since 2026-11-04 it is the one trained with Starfruit Supernova's labels; the
  one before it is kept, and `--terrain dark_passage` goes back to it (section 4, Starfruit).
- **Recordings are for checking, not training.** Tuning clips set nothing new unless a gate fails;
  held-out matches are replayed once and are then spent.
- **The sim is untouched.** The policy never sees a map id, so no retraining and no sim copy.

## 1. Steps

| step | who | what | done when |
|---|---|---|---|
| M0 | user | a clean top-down image (no annotations) | it is `brawl_deployment/data/maps/<name>.png` |
| M1 | both | the label, in `scripts/map_label.py` | it passes the checks, the user signs off its overlay, a test pins its spawns |
| M2 | user | record 3 tuning clips + 2 held-out matches (section 2) | the 5 clips are in `brawl_vision/data/training_videos/` with their spawns noted |
| M3 | Claude | replay the tuning clips, report, label check | the gates in section 3 pass, or the miss is named |
| M3b | both, only if M3 says so | classifier labels on the tuning clips, retrain, rescore | the held-out score for the new clips does not fall and M3 passes |
| M4 | Claude | replay the held-out pair once | the gates pass; the pair is then spent |
| M5 | user | live: dry run, then with control | the live CSVs pass the same report |

### M1. The label

```bash
.venv/Scripts/python.exe scripts/map_label.py <name> --corners X0 Y0 X1 Y1
```

- `--corners` are the play area's top-left and bottom-right corners in image pixels. The
  checkerboard fit only works on a map with a visible floor checker; it worked on Dark Passage
  (clarity 2.78) and not on Crystal Eye Castle (1.04). Claude measures the corners and checks that
  every spawn ring lands on a tile centre (fractional parts near .5) before the user opens the tool.
- The first pass stamps the spawns and proposes 12 colour clusters. The window shows the image on
  the left and the label on the right, every cell opaque in its legend colour (`?` magenta), with
  each class's count in the legend. Hovering a cell names its (col, row) and class under the
  panels and outlines its cluster in yellow: `c` gives the held class to exactly the outlined
  cells. The user names each cluster that way and fixes stragglers by hand. `s` saves and runs
  the checks, `q` saves and quits, and closing the window saves unsaved edits too.
- **Before the user opens the tool, Claude previews the first pass.** It names the clusters in a
  dry run that writes nothing, runs the save checks on the result, checks every cell it suspects
  at 5x against the image, and hands over the cells the naming gets wrong as `--mark C,R ...`:
  the tool rings them in cyan on both panels and `n` centres each in turn. The user's session is
  then "name the clusters, then `n` through the rings".
- **Transcription rules** (unchanged from Dark Passage and the screenshot maps):
  - a raised sprite is labelled at its BASE, the cell it stands on;
  - stumps, blocks, tombstones, spiked fences, candles and anything else that stops a body: `#`;
  - no crates: power cube boxes are never drawn on a map image (they spawn semi-randomly each
    match), so every sprite on the image is fixed terrain, never a box spot;
  - when it is unclear which terrain class a sprite is, it is the user's call, asked before
    labelling, never guessed (the candles were stamped as crates once).
- Then Claude adds the map's spawn list to `tests/test_known_map.py`, as Dark Passage's is.

### M2. Recording: see section 2.

### M3. Replay and report (tuning clips only)

```bash
.venv/Scripts/python.exe scripts/probes/replay_clip.py <clip> --map <name> --mask on --device cpu --csv runs/map_checks/<name>/<clip>.csv
```

```bash
.venv/Scripts/python.exe scripts/probes/map_localize_report.py runs/map_checks/<name>/*.csv --map <name> --spawn C,R --spawn C,R --spawn C,R
```

- The replay prints the label check: the classifier's own map against the label. Cells it is
  sure of that still disagree are listed; the user rechecks each in `map_label.py --review`.
  Each is a label error or a classifier error. (On Dark Passage all six were wall tops leaning
  north in perspective: the label was right.)
- `--device cpu` because a training run may hold the GPU.

### M3b. Classifier labels, only when M3's agreement says the palette is the problem

- Label 4-6 frames per tuning clip in `scripts/vision_label.py <clip> <frame> --hud emulator`,
  spread over every material on the map. Never frames from a held-out clip.
- Retrain with `scripts/vision_train_terrain.py` and judge it with
  `scripts/vision_score_map.py --hold-out-each <the new clips>` next to `--held-out-model` for
  the old weights (that script's docstring has the procedure). Then rerun M3.
- **Without hand-labelling (first used on Starfruit Supernova, 2026-11-04).** Once the map label
  is right, frame labels can be projected from it instead of drawn:
  1. register each frame against the map image;
  2. give each cell the ground-footprint class under it;
  3. refuse whatever the labels README refuses.

  `scripts/map_project_labels.py` does this for any map and BlueStacks clip, and fits the
  emulator's scale for each clip. Its docstring has the usage, and
  `scripts/probes/map_label_projection/README.md` has the rules and the first run. Run it twice,
  on the CPU (`CUDA_VISIBLE_DEVICES=-1`):
  1. `scripts/map_project_labels.py <map> <clip>` registers the clip, which takes about 2 min
     and is cached. It prints the scale fit and the frames it can label.
  2. Add `--train <frames> --held-out <frames>`. Name any cells the map label hasn't settled with
     `--leave C,R` or `--leave-row R:C0-C1,...`, plus `--leave-why "<reason>"`. Labels take about
     3 s a frame. The user then reviews the overlays in the work folder's `review/` instead of
     drawing.
  - Keep the held-out frames on ground the training frames never saw. The tool prints how many
    map tiles the two sets share; Starfruit's shared none.
  - Put them in a folder of their own, because `--hold-out` separates by clip and both sets come
    from the same clip. The tool's default is `tests/fixtures/vision/labels_<map>_test/`; a new
    one goes in `HELD_OUT_LABELS` in `tests/test_vision_classifier.py`.
  - Any error in the map label is copied into every frame, so the map label's review comes first.
  - **Ground that breaks mid-match (first on Rockwall Brawl, 2026-11-11).** The map label is the
    ground as the match starts. A cone, cube or bush broken later shows floor, so the projection
    would still label it from the map. Project once without `--broken`, then run a broken-cells
    probe on those labels (`scripts/probes/map_label_projection/rockwall_brawl/broken_cells.py`;
    its lines suit bright walls on a dark floor) and settle each tile it flags on its crops. Name
    each break as `--broken C,R@F`, with F a frame that shows it broken, and project again. Rerun
    the probe on the final labels: no tile should stay past the line.
  - **Skip frames under a screen-wide overlay:** the "SHOWDOWN!" banner when two brawlers are left,
    or a full-screen tint. The projection refuses HUD text, detector boxes and cells that change
    against nearby frames, but a banner holds still for seconds, so the frame pickers drop it
    (Truth's `pick_heldout.py`, Rockwall Brawl's `pick_frames.py`).

### M4-M5. Held out, then live

- M4 is M3's two commands on the held-out pair, once, after nothing is left to tune.
- M5: `deploy_run.py --map <name> --dry-run --telemetry <csv>` while the user plays by hand, then
  with control. Each live CSV goes through the same report.

## 2. What to record

All of it in BlueStacks with OBS, set up exactly as for Dark Passage, in **real Showdown matches
on the map**: Training Grounds cannot test anything here.

**Every clip:**

1. Start OBS in the loading screen, so the clip holds the match start: the fix anchors on the spawn.
2. **Stand still for about 1 second after the HUD appears.** The report's truth is the hero on its
   spawn when the gate opens, about 0.5 s after the HUD shows; a hero already running reads a tile
   off all match.
3. Note which spawn you started on (the list in section 4, or "top-left", and Claude checks it on
   the gate frames).
4. Name it `bluestacks-<map-with-dashes>-<spawn>.mp4`, held-out ones
   `bluestacks-<map-with-dashes>-holdout-<spawn>.mp4`, in `brawl_vision/data/training_videos/`.
   The `bluestacks-` prefix is what gives a clip the emulator HUD mask; a date-named OBS file
   silently gets the phone's.

**3 tuning clips, 40-60 s each, from three different spawns: "wall walks".** Walk slowly past and
along every material on the map, the way a tuning clip should cover what the classifier will be
asked to read. Dying early is fine; only the walk matters.

**2 held-out matches, from spawns the tuning clips did not use if possible.** Play them as whole
matches, normally. Include runs along an edge, into a corner and through bushes: the camera box
lives at the edges, and both right fixes ever dropped were in bushes.

Do not run `vision_export_frames.py` with no clip names after recording: it takes every .mp4 in
that folder, held-out clips included.

## 3. Gates (M3 and M4, from K4)

| gate | target |
|---|---|
| spawn picked right | every clip |
| first fix after the gate | under 2 s |
| time without a fix | under 5% of the match |
| crate spread about the map's crate offset | at most 0.25 tile |
| label check best offset | (0, 0) |
| agreement while fixed, median (Dark Passage: 0.73-0.81) | watched, not gated; well under 0.70 points at the palette, so M3b |

The localizer's thresholds in `configs/deployment.yaml` are shared by every map. A miss is
reported with its cause first; changing a shared threshold is the user's call, and would be checked
on Dark Passage's held-out footage too.

## 4. Maps

| map | M0 | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|
| dark_passage | done | done 09-28 | done 09-28 | done | done | done 09-29 |
| crystal_eye_castle | done 11-03 | first pass previewed 11-03; the user's turn in the tool (no CSV yet) | paused 11-04: rotated out | | | |
| starfruit_supernova | done 11-04 | drafted 11-04, spawns pinned; the user's review | one match clip 11-04, not the section 2 set | shipped classifier never fixes; M3b labels projected; candidate (held-out F1 .946 vs .502) fixes at the gate, Dark Passage replays no worse; shipped 11-04, old one kept (`--terrain dark_passage`) | | |
| truth | done 11-05 | drafted 11-05 from the user's annotated references; spawns and half-turn symmetry pinned; the user's review | one clip with two matches, 11-05, not the section 2 set | the default fixes at the gate but reads floor as blocking; M3b labels projected, split by ground; `truth` classifier (held-out F1 .982 vs .907) fixes at the gate on both matches and passes the regression replays; shipped 11-05 under its own name (`--terrain truth`), default unchanged | | |
| doom_shroom | done 11-10 | drafted 11-10, spawns pinned; the orange barriers and the fences are walls (the user, 11-10); the user's review | two match clips 11-10, the west then the east, not the section 2 set | the default never fixes (wrong-map verdict) and `truth` goes unfixed 28% of the match; M3b labels projected, split by ground; `doom_shroom` classifier shipped 11-10 under its own name (`--terrain doom_shroom`), default unchanged; v2 added the east recording's labels the same day; v3 (11-11), trained after the fences became walls, holds the fix all match on both clips (v2 on the new label goes unfixed 10.7% of the first) and every regression replay keeps its fix; held-out F1 west .982, east .959 (v2 .967, .971); v1 and v2 deleted | | |
| rockwall_brawl | done 11-11 | done 11-11: the user signed off the draft with no changes; spawns pinned; the destructible cones are walls (the user, 11-11) | one match clip 11-11, not the section 2 set | no older classifier fixes: each reads bush and wall swapped, and the localizer calls the map wrong; M3b labels projected, split by ground, broken ground left `?`; `rockwall_brawl` classifier (held-out F1 .987 vs .03-.04) fixes at the gate and holds the fix all match, and every regression replay keeps its fix; shipped 11-11 under its own name (`--terrain rockwall_brawl`), default unchanged | | |

### Crystal Eye Castle (measured 2026-11-03)

| item | value |
|---|---|
| image | `brawl_deployment/data/maps/crystal_eye_castle.png`, 981 x 969, converted from the user's clean screenshot |
| play area | x 16-961, y 14-960: `--corners 16 14 961 960`, 15.75 x 15.77 px per tile, 60 x 60 |
| why not the fit | the purple floor's checker is too faint: clarity 1.04 and a y pitch half a tile off by the bottom edge |
| lattice check | all 10 spawn rings land on tile centres, fractional parts .40-.62 (Dark Passage .44-.54) |
| spawns, (col, row) | (40,2) top; (19,6) top-left; (4,18) west upper; (57,18) east upper; (31,26) centre ring, upper; (28,33) centre ring, lower; (55,40) east lower; (2,42) west lower; (41,53) bottom-right; (19,57) bottom-left |

**Materials, from the user's annotated copy:**

| looks like | class | source |
|---|---|---|
| purple floor | `.` | |
| bright green checkered bands (the centre ring and the two big arcs) | `.` floor | the user's label |
| dark green tall grass | `b` | the user's label |
| teal shapes with dark outlines | `~` | the user's label |
| orange rounded blocks | `#` | the user's label |
| lavender ghosts, dark green pillars with pink tops | `#` | the user's label |
| tall lavender tombstones/spikes | `#` | the spiked-fence rule |
| slate-blue blocks with a skull, in twos and threes | `#` | the user, 2026-11-03 |
| dark green pillars with an orange cap | `#` | the pink-capped pillar's body and shape, another cap |

**First pass, previewed 2026-11-03.** Each of the 12 clusters holds one material: floor 4 (purple,
purple shaded under a sprite, the bright green bands, the spawn-ring cells), wall 6 (orange x2,
pillars x2, ghosts / tombstones / skull blocks x2), bush 1, water 1. Named that way, the label
passes the save checks. 31 cells come out wrong; each was checked at 5x against the image:

- **28 pillars read as bush:** paint `#`. Their body is as dark a green as the bushes, and the
  band the clusters read sits under the cap. 26 are orange-capped, the two at (45,56) and (45,57)
  pink-capped. (col, row): (20,13) (30,13) (55,13) (56,13) (17,14) (28,16) (53,20) (52,22)
  (16,24) (46,28) (47,28) (12,31) (13,31) (43,35) (52,35) (7,37) (6,39) (31,43) (46,43) (42,45)
  (29,46) (39,46) (32,48) (13,53) (45,56) (46,56) (45,57) (46,57).
- **3 floor cells under a spawn ring read as wall:** paint `.`. The ring's lower arc reaches into
  their band. (40,3) (19,7) (57,19). These matter most: each sits in the landing view of every
  match started on that spawn, and the move mask would treat a step south off the spawn as a step
  into a wall.

With the 31 fixes the label still passes: floor 71%, wall 17%, bush 8.5%, water 3%, 10 spawns.
The command rings all 31 (`--mark`, in the order above): `n` centres each in turn, the first 28
to paint `#`, the last 3 `.`.

```bash
.venv/Scripts/python.exe scripts/map_label.py crystal_eye_castle --corners 16 14 961 960 --mark 20,13 30,13 55,13 56,13 17,14 28,16 53,20 52,22 16,24 46,28 47,28 12,31 13,31 43,35 52,35 7,37 6,39 31,43 46,43 42,45 29,46 39,46 32,48 13,53 45,56 46,56 45,57 46,57 40,3 19,7 57,19
```

**Risks M3 will measure:**

- **Bright green floor next to dark green bush.** The classifier has never seen a floor this green,
  and its training uses hue rotation, so it leans on texture. If it reads the bands as bush, every
  band cell scores -1 and agreement falls. The label check will show it as band cells the
  classifier calls bush.
- **The centre is close to 180-degree symmetric** (two water-and-block pieces, two spawns). The
  camera never rotates, so a view and its rotated twin score differently, but a tuning clip from a
  centre spawn should show it.
- **Purple floor.** Dark Passage's floor was green-grey; this is the first deploy map with a
  saturated floor.

**Suggested clips:**

| clip | spawn | route |
|---|---|---|
| tuning 1 | (31,26) or (28,33), centre | circle inside and outside the bright green ring, past both water-and-block pieces and the ghost and pillar walls |
| tuning 2 | (2,42) or (4,18), west | up and down the lavender spike columns, past the edge water and orange clusters |
| tuning 3 | (40,2) or (41,53), top or bottom | through the orange belts, along the green arc and the dark bush beside it |
| held out 1 and 2 | any two not used above, (57,18) and (19,57) for instance | whole matches, edges, corners and bushes included |

### Starfruit Supernova (2026-11-04)

| item | value |
|---|---|
| image | `brawl_deployment/data/maps/starfruit_supernova.png`, 976 x 980, from the user |
| lattice | checker fit, pitch 15.739 x 15.718 px, origin (21.039, 20.218); all 10 spawn rings centred within .47-.52 |
| spawns, (col, row) | (26,4) (44,6) (14,7) (51,14) (7,22) west upper (56,35) (5,40) (55,45) (17,53) (37,54) |
| label | 60 x 60: floor 2481, wall 675, bush 389, water 45, spawn 10; spawns pinned in `tests/test_known_map.py` |
| recording | `bluestacks-starfruit-supernova-west-upper.mp4`: the user's Shelly match at 2560 x 1440 from spawn (7,22), copied from `Videos/2026-11-04 15-03-50.mp4` |

**Materials** (the user: walls pink and blue, grass green, floor purple with yellow patterning,
water blue):

| looks like | class | note |
|---|---|---|
| purple jigsaw floor | `.` | |
| pink X pads, yellow `?` and star pads, the orange `¿` tile, white dots | `.` | Floor art: flat, and the hero walks over it in the footage. The map image leaves it out. |
| pink blocks; blue blocks, many with a glyph (x, skull, `?`, `¿`, arrows) | `#` | |
| green grass | `b` | |
| light blue pools | `~` | |
| crimson meeple figures, map rows 26 and 33, 66 cells | `#` for now | **The user's call: wall or fence?** No footage shows a shot crossing one. They're `#` in the map label and never a classifier label (`?`). The cells are listed in `scripts/probes/map_label_projection/project.py`. |
| dice, number tiles and cards around the board | outside the 60 x 60 | |

**M3 on the match clip** (`runs/map_checks/starfruit_supernova/`):

- **The match gate never opened for a brawler without a hypercharge. FIXED 2026-11-04**
  (BRAWL_DEPLOYMENT_DESIGN.md §5.2).
  - The gate anchored on the hypercharge disc alone. This account's Shelly (power 3) has no
    hypercharge or gadget button, so the replay waited all match. So did the first live run on
    the real app, which is how it surfaced: this was never only a footage problem.
  - Gating on the super disc at one radius opened the gate, but it flickered as the super lit
    up, because the charged face is a wider ring.
  - The gate is now the best of three looks (`match_gate.looks` in
    `brawl_deployment/data/control_calibration.json`): the hypercharge, then the Super's
    uncharged and charged faces. On this clip it opens at 15.7 s and holds to the end. The
    gadget stays masked unless its disc shows in the first second of the match.
- **The shipped classifier never gets a fix.**
  - Over 58.3 s from the super-disc gate: 35% search, 65% mismatch.
  - Agreement: median 0.17–0.19, max 0.45, against the 0.6 commit floor. Dark Passage reads
    0.73–0.81 once fixed.
  - The hero was already running at the gate (2.6 tiles/s), so the spawn truth is rough. This
    is a match clip, not a section 2 clip.
  - The palette is the problem, so this map needs M3b.

**M3b: projected labels.** The labels were made by machine from the map label; nobody drew them.
`scripts/probes/map_label_projection/README.md` has the method, and the
`scripts/map_project_labels.py` command that rebuilds all 17 byte-identical.

| set | frames | cells |
|---|---|---|
| training, `tests/fixtures/vision/labels/` | 460 550 640 690 820 910 1000 1090 1180 1340 | 1808: floor 1315, wall 283, bush 172, water 38 |
| held out, `tests/fixtures/vision/labels_starfruit_test/` | 1840 1900 1960 2020 2080 2140 2200 | 1346: floor 1093, wall 178, bush 58, water 17 |

- The two sets share no world tile: training covers the north and the spawn, the held-out set
  covers the south.
- Review sheets: `runs/map_checks/starfruit_supernova/label_review/sheet_train.jpg` and
  `sheet_test.jpg`, with one image per frame under `train/` and `test/`.

**The emulator rectifies 2.2% too wide in x (0.4% in y).** Odometry and map-image NCC agree on
this over 143 frames, and the 1080p Dark Passage clip shows it too. Live odometry therefore
over-reads x motion by about 2%. The labels correct for it. The camera model does not; that's a
separate recalibration.

**The candidate classifier.** The two models below were trained by `vision_train_terrain.py`
on `tests/fixtures/vision/labels/`, with and without the ten Starfruit frames. Both are kept in
`runs/terrain_starfruit/`. The candidate was shipped as `brawl_vision/data/terrain.pt` on
2026-11-04 (the user's call); "old" below is the classifier it replaced, kept as described in
**Going back to the old classifier** at the end of this section.

| on the seven held-out frames | blocking F1 | recall | precision | 5-class |
|---|---|---|---|---|
| old (`terrain_dark_passage.pt`) | 0.502 | 0.589 | 0.438 | 0.831 |
| control: same recipe, no Starfruit labels | 0.548 | 0.526 | 0.572 | 0.866 |
| candidate | **0.946** | 0.963 | 0.929 | **0.984** |

The held-out README (`tests/fixtures/vision/labels_starfruit_test/README.md`) has the full table and
its caveats.

**No regression on Dark Passage.** The three Dark Passage tuning clips were replayed with each
model, using the normal hypercharge gate. The telemetry is in
`runs/map_checks/dark_passage_regression/`.

| clip | fix, both models | agreement while fixed, median / 5th pct / min | label check |
|---|---|---|---|
| bottom-left | 0.00 s after the gate, 100% | 0.74 / 0.65 / 0.56 → 0.77 / 0.64 / 0.59 | 0.921 → 0.941 |
| bottom-right | 0.00 s, 100% | 0.79 / 0.65 / 0.55 → 0.82 / 0.65 / 0.58 | 0.950 → 0.968 |
| top-right | 0.10 s, 100% | 0.73 / 0.65 / 0.53 → 0.78 / 0.66 / 0.54 | 0.932 → 0.960 |

Each pair is old → candidate.

- **Fixes and offsets are identical.** Crate spread is 0.208 → 0.214 on bottom-left and
  0.186 → 0.182 on bottom-right.
- **Sub-tile corrections a minute are mixed:** 6.5 → 15.2, 10.9 → 8.5 and 6.5 → 4.9.
- **The old model's column matches K4's table exactly.** A replay's perception is deterministic,
  and since 2026-11-04 so are its decisions (**Going back to the old classifier**, at the end of
  this section).
- **Dark Passage isn't unseen.** Both models have its three `2maps_dataset` frames (m01) in
  training, so this checks for regression on a seen map's new footage, not generalization.
- **The two held-out matches were not used,** since they are spent once used.

**M3 again, with the candidate.** The replay used the same super-disc wrapper; its telemetry is in
`bluestacks-starfruit-supernova-west-upper_candidate.csv`. `map_localize_report.py --spawn 7,22`
gives:

- **It fixes at the gate (0.00 s) and picks spawn (7,22).** Agreement while fixed: median 0.79,
  5th percentile 0.74, minimum 0.73. That is Dark Passage's range.
- **There are no corrections.** The crate-residual spread is 0.216, inside the 0.25 gate.
- **27% of the match has no fix (gate < 5%, a miss).** This stretch starts at 50.8 s, when the
  super-disc gate reopened. The loop treats that as a new match and resets the localizer, which
  then searches 15.9 s to fix again. It refixes on the south half, ground the classifier never
  trained on. A Bull match's hypercharge gate doesn't reopen like this.
- **This is the clip the training frames came from.** Agreement in the north half is partly
  memorized. The refix in the south half is the cleaner test.
- **The two fixes differ by -0.24 tile in x.** The hero went 14.7 odometry tiles east between them,
  and a 2.2% over-read over that predicts -0.32. In y they differ by -0.60, where the 0.4% error
  predicts -0.10, so fixes are only good to a few tenths of a tile. It's a hint, not a measurement.

**Label check: 0.946 over 1172 tiles, best at offset (0, 0).** In 37 cells, at least 90% of views
disagree with the label. Each was checked in the map image, and all but (23,14) and (24,15) in
three clip frames too; those two never came well inside the view. None is a label error. The sheets are in `runs/map_checks/starfruit_supernova/label_review/`:
`stragglers_*.png` for the map image and `straggler_frames_*.png` for the frames.

| cells | seen | cause |
|---|---|---|
| (3,26) (8,26) (9,26) (10,26) (21,26) | floor | Meeples. They never get a classifier label. |
| (17,1) (17,2) (20,6) (27,8) (22,11) (22,12) (19,13) (23,14) (24,15) (34,40) (31,36) | floor | The walls are standing in every frame that shows them. Ten of these eleven are the west edge of their run, which fits the x over-read: evidence lands east of where it came from, by about 2% of the distance from the fix. |
| (19,20) (20,20) (16,24) (17,24) | floor, water | The bushes are standing in every frame. |
| (24,11) (20,36) (21,37) (8,41) (10,41) (29,41) (7,45) (13,49) (30,55) (18,20) | wall, bush | A wall or bush sprite's top overhangs the tile north of it. The label reads sprites at their base. |
| (15,16) | wall | A power cube box. Boxes are never drawn on the map image. |
| (28,49) (29,49) | wall | A big pink floor pad. Pads are floor art the map image leaves out. The training frames, in the north, hold few of them. |
| (27,1) (20,22) | wall | Plain floor beside walls. Late frames have the kill banner over (27,1), and the aim joystick crosses (20,22). |
| (19,45) (15,46) | floor, bush | The rounded ends of the pools. |

**Going back to the old classifier.** Shipping replaced one file and added one; nothing was
deleted.

| file | sha256, first 16 | what it is |
|---|---|---|
| `brawl_vision/data/terrain.pt` | `578f14d968a06430` | the candidate, the default since 2026-11-04 |
| `brawl_vision/data/terrain_dark_passage.pt` | `f7a788b25eadd1d0` | the old classifier, byte for byte; `tests/test_vision_classifier.py` pins its hash and runs it |
| `brawl_vision/data/terrain_truth.pt` | `f897391f994d078c` | `truth`, added 2026-11-05 for Truth (its section, below); pinned the same way |

A session picks its classifier by name (`TERRAIN_MODELS` in `brawl_vision/config.py`), never by
path, so the files above are the only ones it can load:

| to run the old one in | do this |
|---|---|
| one live session | `./.venv/Scripts/python.exe scripts/deploy_run.py --map dark_passage --terrain dark_passage` |
| every live session | `terrain.model: dark_passage` in `configs/deployment.yaml` |
| a replay | `./.venv/Scripts/python.exe scripts/probes/replay_clip.py <clip> --map dark_passage --terrain dark_passage` |
| a score | `./.venv/Scripts/python.exe scripts/vision_score_map.py --model brawl_vision/data/terrain_dark_passage.pt` (scoring takes paths: it's how a model earns a name) |

- **The choice is printed.** `deploy_run.py` logs `terrain dark_passage (terrain_dark_passage.pt)`
  at startup, and `replay_clip.py` puts `terrain dark_passage` in its header. A missing file stops
  `deploy_run.py` before it builds anything.
- **The replays say neither map needs it.** The candidate fixes everywhere the old one did, and
  Starfruit only fixes with the candidate. It's kept in case live Dark Passage play shows
  something the replays missed.
- **If `terrain_dark_passage.pt` is ever lost,** there are two more copies:
  - `terrain.pt`'s blob at commit `7db0cc0`. Restore it from Git Bash with
    `git show 7db0cc0:brawl_vision/data/terrain.pt > brawl_vision/data/terrain_dark_passage.pt`.
    PowerShell 5.1's `>` re-encodes the bytes, and the hash test then fails.
  - `runs/terrain_2maps/graveyard_all.pt`, which isn't tracked.
- **Checked 2026-11-04 by replay** (`runs/map_checks/terrain_switch/`, every column but the two
  timing ones). `--terrain dark_passage` reproduces the old model's replays of all three Dark
  Passage tuning clips cell for cell. The default reproduces the candidate's bottom-left and
  Starfruit replays the same way, though Starfruit needed a second run (next bullet).
- **Until 2026-11-04, a replay's decisions read the wall clock.** The same Starfruit command, run
  twice, gave two answers. The decision at 68.7 s was `attack` 0 in one run and 4 in the other.
  The shadow's ammo and the moves after it followed for the last 5 s, 47 cells in all.
  - Three of four runs agreed cell for cell: both that loaded the candidate by path (the
    pre-ship way) and one of the two default runs. No perception column differed in any run.
  - **The cause.** `replay_clip.py` gave the shadow the clip's clock, but `meta.time_frac` read
    the wall clock. The run that flipped was the slowest. Its loop had spent 9.3 s since the gate
    reopened, against 7.9–8.8 s in the others, for 7.8 s of clip.
  - **Fixed the same day.** The loop now reads match time through `DeployLoop._match_clock`, and
    the replay sets that to the clip's time. Two Starfruit runs now agree cell for cell
    (`runs/map_checks/match_clock/`). Two tests pin it, and both fail if any of the loop's three
    match-time reads goes back to the wall clock.
  - **Older replays.** Replays made before the fix still compare on every perception column, but
    not decision for decision. Replayed again, 2–8% of their decisions change: bottom-left 10 of
    104, bottom-right 3 of 188, top-right 12 of 142, and Starfruit 2 or 9 of 99, depending on
    which of the two old runs is the baseline.

### Truth (2026-11-05)

| item | value |
|---|---|
| image | `brawl_deployment/data/maps/truth.png`, 978 x 977, from the user, with two annotated references naming each item (copies in `runs/map_checks/truth/`) |
| lattice | checker fit, pitch 15.747 x 15.728 px, origin (16.7, 16.7) |
| spawns, (col, row) | (28,12) (38,16) (19,17) (46,23) (12,26) match 1 (47,33) match 2 (13,36) (40,42) (21,43) (31,47); pinned in `tests/test_known_map.py` |
| label | 60 x 60: floor 2400, wall 632, bush 410, water 148, spawn 10. The game draws Truth with exact half-turn symmetry. The label has it too, wall textures included, and a test pins it. |
| recording | `bluestacks-truth-west-upper.mp4`, copied from `Videos/2026-11-05 15-05-44.mp4`: 2560 x 1440, 30 fps, 3441 frames, two matches. Match 1 gates at 17.0 s on (12,26), and the hero dies at about 27 s. Match 2 gates at 47.4 s on (47,33) and runs to the end of the clip. |

**Materials.** From the user: walls come as pink stone, pink crates, blue mailboxes and blue
barrels. The teleporter section (TELE on the references) counts as walls. Water and bushes look
alike.

| looks like | class | tiles | texture in `scripts/probes/map_label_projection/truth/textures.csv` |
|---|---|---|---|
| pink checker floor | `.` | 2400 | |
| pink stone blocks | `#` | 544 | stone |
| pink crates | `#` | 24 | crate |
| blue mailboxes | `#` | 20 | mailbox |
| blue barrels | `#` | 12 | barrel |
| teleporter pads | `#` | 32 | teleporter. **Walls for the bot, the user's call:** it was never trained to teleport. |
| teal grass | `b` | 410 | |
| blue pools | `~` | 148 | |

**M3 with the default classifier: it already fixes.** Unlike on Starfruit, `terrain.pt` fixes at
the gate on both matches, on the right spawn. Its weakness is precision. On the held-out frames
below it reads 48 floor cells as wall and 26 as bush, for a blocking precision of 0.839. Truth
went through M3b anyway: the user asked for strong results, not a proof of concept.

**The projection's scale was set, not fitted.** The odometry fit read S = (0.939, 0.935) over 197
samples. Two events bent it:

- a "SHOWDOWN!" banner at 89–93 s (frames 2630–2840), which froze odometry while the camera moved
  about 7 tiles;
- a y jump in odometry at 80–83 s (frames 2390–2480).

Both were the odometry, and the overlay rule fixed both on 2026-11-05 (below). The fit hasn't been
re-run, because the labels were built on the set scale.

`--scale 1.022 1.004`, Starfruit's fit on the same emulator, raised the registration's NCC median
from 0.483 to 0.634. It cut the block residuals from 0.36 / 0.29 to 0.10 / 0.12 tile (x / y).
The stretch left across a frame is x +0.0013, y −0.0078, which puts the frame edge 0.07 tile off.
`scripts/map_project_labels.py`'s docstring (**Scale.**) has the check.

**M3b: projected labels, split by ground, not time.** Match 2's last minute re-crosses the
north-west that match 1 covered, so a cut in time would put the same ground in both sets. The
held-out set labels only R, the south-west quadrant (map tiles x 0–28, y 29–59). The training
labels leave R `?`.

| set | frames | cells | map tiles |
|---|---|---|---|
| training, `tests/fixtures/vision/labels/` | 530 570 610 650 690 730 1460 1510 1560 1600 1640 1680 1720 1760 1800 2490 2580 2850 2880 2910 2950 3000 3030 3060 3100 3140 3220 3250 3280 3320 | 5450: floor 4310, wall 572, bush 300, water 268 | 1418, none in R |
| held out, `tests/fixtures/vision/labels_truth_test/` | 1830 1870 1910 1940 1980 2010 2090 2220 2260 2300 | 1910: floor 1457, wall 225, bush 139, water 89 | 639, all in R |

- **Every wall texture is in both sets.** Wall cells / tiles, training: stone 506 / 164,
  teleporter 25 / 19, crate 21 / 9, barrel 15 / 6, mailbox 5 / 5. Held out: stone 209 / 71,
  crate 8 / 5, mailbox 4 / 3, barrel 2 / 2, teleporter 2 / 1.
- **Teleporter pads are mostly `?`.** They animate, so the transient check refuses them in most
  views.
- Frame 770 was dropped: a damage flash left 13 of its cells standing.
- The rebuild commands are in the held-out README. Review sheets are in
  `runs/map_checks/truth/projection/bluestacks-truth-west-upper/review/`.

**The `truth` classifier, shipped under its own name.** `vision_train_terrain.py` trained it on
`tests/fixtures/vision/labels/`: the 30 Truth frames plus every older label, 132 frames. It used
the defaults (200 epochs, seed 0) on the CPU, in 11 min. `terrain.pt` stays the default, because
which classifier is the default is the user's call.

| on the ten held-out frames | blocking F1 | recall | precision | 5-class |
|---|---|---|---|---|
| default (`terrain.pt`) | 0.907 | 0.986 | 0.839 | 0.954 |
| `terrain_dark_passage.pt` | 0.711 | 0.721 | 0.702 | 0.891 |
| **`truth`** | **0.982** | 0.984 | **0.980** | **0.992** |
| candidate 2: 17 more Truth frames | 0.981 | 0.990 | 0.972 | 0.989 |

- **Bush and water stay apart.** `truth` reads 1 of 137 bush cells as water and 0 of 82 water
  cells as bush. In the replay, inside R, it reads water 36 of 36 and bush 84 of 86 (2 as wall).
  The held-out README has the per-class and per-texture tables.
- **Starfruit is unchanged:** held-out blocking F1 0.945, against the default's 0.946.
- **Candidate 2 was not shipped.** It tied `truth` on Truth and on Starfruit, but failed the
  regression replays below.
  - On Dark Passage bottom-left it lost the fix at 12 s and never found it again.
  - On bottom-right its crate spread was 0.299, past the 0.25 gate.
  - One training run can't say whether the 17 frames or the run's luck did that. Its labels are
    in `runs/terrain_truth/labels_plus/`.

**Regression replays** (`runs/map_checks/truth/regression/`, by `regress.sh` in
`scripts/probes/map_label_projection/truth/`). Each cell reads default → `truth` → candidate 2.

| clip | time fixed | agreement median | sub-tile corrections a minute | crate spread | label check |
|---|---|---|---|---|---|
| Dark Passage bottom-left | 100% → 100% → **44%** | 0.77 → 0.85 → 0.69 | 15.2 → 4.4 → 14.8 | 0.214 → 0.184 → 0.215 | 0.941 → 0.951 → 0.855 |
| Dark Passage bottom-right | 100% → 100% → 96% | 0.82 → 0.86 → 0.80 | 8.5 → 7.2 → 21.3 | 0.182 → 0.210 → **0.299** | 0.968 → 0.973 → 0.948 |
| Dark Passage top-right | 100% → 99% → 99% | 0.78 → 0.72 → 0.73 | 4.9 → 3.3 → 4.9 | | 0.960 → 0.955 → 0.961 |
| Starfruit | 100% → 100% → 100% | 0.82 → 0.86 → 0.87 | 5.2 → 5.2 → 6.2 | 0.251 → 0.257 → 0.268 | 0.956 → 0.958 → 0.956 |

- **`truth`'s one step back is top-right.** Its fix dropped once, for 0.2 s at 11.0 s, and came
  back at the same offset. The agreement median fell from 0.78 to 0.72.
- **Starfruit's crate spread sits on the gate with every model.**
- **`truth` loads only when asked for.** These replays don't guard a change on those maps. They
  catch a classifier that learned something brittle.

**M3 again, with `truth`** (`runs/map_checks/truth/shipped/`: `replay_clip.py --map truth
--terrain truth --run runs/mortis_ppo-20261103-092146`). Outside the two timing columns, the
replay is identical to the one that loaded the candidate by path. It predates the overlay rule,
and that section below has the same replay with the rule on.

| gate | match 1, spawn (12,26) | match 2, spawn (47,33) |
|---|---|---|
| first fix | 0.00 s after the gate, 0.72 tiles from spawn | 0.00 s, 0.58 tiles |
| time without a fix (gate < 5%) | 0% | 0.7%: 0.3 s at the banner, 0.2 s confirming after an odometry cut at the end |
| crate spread (gate ≤ 0.25) | 0.209 | 0.163 |
| agreement median, default → `truth` | 0.68 → 0.73 | 0.79 → 0.86 |
| corrections | 1 whole-tile, 3 sub-tile in 10.3 s | 2 whole-tile, 4 sub-tile in 67 s |

The label check reads 0.963 over 2678 tiles, best at offset (0, 0); the default reads 0.942.

**What the localizer did on Truth before the overlay rule.** `loc_vs_image.py` compares its
offset with the image registration of each frame. While the fix is right, the difference stays
constant. The first three events below were the odometry, and the overlay rule removed them.

- **23.8 s, "slipped a tile (+1, −1)": a correction.** Odometry had drifted 1.6 tiles off the image
  in under 4 s. The slip brought it back to 0.45.
- **83.3 s, "slipped a tile (0, −1)": the y jump.** Odometry's y drifted 1.0 tile off the image
  over 80–83 s, and the slip took the error back to 0.0.
- **91.5 s, "fix dropped": the banner.** The refix at 91.8 s was a tile off in x and in y, and
  slipped back (+1, +1) at 93.8 s. That left it 0.44 tile off in x.
  - `map_localize_report.py` calls every fix after the banner WRONG, by about 7 tiles. That
    verdict is an artifact: its truth is odometry counted from the spawn, and the frozen odometry
    lost the 7 tiles the camera moved. The image registration says the fix is right.
- **A slow x drift: the emulator's x over-read** (EMULATOR_RECTIFY_PLAN.md, the user's call).
  Match 2 walks 26 tiles west across the south between 53 and 67 s. Over that walk the fix slid
  0.28 tile west of the image, 1.1% of the walk, and no correction caught it.

These numbers come from `loc_vs_image.py` as fixed on 2026-11-05, which interpolates the decision
rows to each frame's time. The first version took the nearest row, up to 1/12 s away, and gave
0.6 for the slip's remainder, 1.6 for the y jump and 0.6 for the walk.

**The odometry overlay rule (2026-11-05).** Screen-static content covers some of odometry's eight
windows: the banner, the hero and its name tag, the recording's mouse cursor. Those windows read
zero while the camera moves.

- On the banner, 5 of 8 read zero for 3 s, and 3 or 4 for a second or two either side. The
  median read zero with the five, so odometry stood still from 89.2 to 92.4 s and fell 7.5 tiles
  behind the camera.
- With three, the per-axis median can land halfway between the two groups. Every window still
  "agrees", so the frame comes back `ok` with the camera slowed. That was the 23.8 s drift and the
  y jump.
- Nothing flagged it: most of those ticks were `ok`. That held for 41 of 49 over 20–24 s, 30 of 36
  over 80–83 s, and 59 of 84 under the banner.

`Odometry.update` now looks for that split: still windows (under `odometry.still_tiles`, 0.03
tile) against a coherent moving group. A per-pixel vote decides which shift explains the frame
pair when either holds:

- the median doesn't sit with the movers;
- the still windows alone would make the frame `uncertain`.

The movers carry the estimate if they explain more than `odometry.overlay_share` (0.5) of the
deciding pixels, and the still windows then don't count against agreement. Every other frame is
estimated and gated as before. The docstring in `brawl_vision/terrain/odometry.py` has the rule and
its measurements, and `scripts/probes/odometry_overlay/` runs it over recorded frame pairs.
`overlay_share: 1.0` in `configs/vision.yaml` turns it off.

The replay, before → after (`runs/map_checks/truth/shipped/replay_truth` →
`runs/map_checks/truth/overlay_fix/truth`, the same command as M3 again):

| | before | after |
|---|---|---|
| ticks in a match: ok / uncertain / lost | 770 / 160 / 2 | 897 / 33 / 2 |
| decision ticks | 249 | 290 |
| label check at (0, 0) | 0.963 over 2678 tiles | 0.978 over 2694 tiles |
| disputed tiles | 30 | 13 |

| `map_localize_report.py`, per match | match 1 | match 2 |
|---|---|---|
| agreement median (5th percentile, min) | 0.73 (0.46, 0.38) → 0.84 (0.70, 0.51) | 0.86 (0.47, 0.12) → 0.91 (0.75, 0.69) |
| corrections, whole-tile + sub-tile | 1 + 3 → 0 + 2 | 2 + 4 → 0 + 3 |
| crate spread | 0.209 → 0.196 | 0.163 → 0.139 |
| time without a fix | 0% → 0% | 0.5% → 0% |
| WRONG, against odometry counted from the spawn | 34% → 0% | 34% → 0% |

The WRONG row measured the odometry, not the localizer: the image registration called those fixes
right. Now the two agree.

| `loc_vs_image.py` (one constant in both runs, so absolute) | before | after |
|---|---|---|
| match 1, 18–26 s | 1.6 tiles off by 23.7 s, 0.45 after the slip | within 0.2 in x and 0.4 in y |
| match 2's walk west, 53–67 s | x slid 0.28 (1.1%) | x slid 0.43 (1.7%), and a sub-tile correction took back 0.3 at 67 s |
| 80–83 s | y 1.0 off until the slip | no jump |
| the banner, 87 → 95 s | 0.44 off in x after the refix and slip | moved under 0.2 |

Both runs move 0.3 in x at 26.7 s, the last registered frame before the hero dies.

**The rule on other maps.** These are the `truth` classifier's regression replays on the default
terrain, before → after (`runs/map_checks/truth/regression/*-shipped` →
`runs/map_checks/truth/overlay_fix/`):

| clip | uncertain ticks in the match | label check | agreement median | sub-tile corrections a minute | crate spread | crate spread, rule on in a match only |
|---|---|---|---|---|---|---|
| Dark Passage bottom-left | 9 → 4 | 0.941 → 0.933 | 0.77 → 0.77 | 15.2 → 13.1 | 0.214 → 0.230 | 0.215 |
| Dark Passage bottom-right | 21 → 8 | 0.968 → 0.964 | 0.82 → 0.82 | 8.5 → 7.2 | 0.182 → 0.198 | 0.202 |
| Dark Passage top-right | 10 → 3 | 0.960 → 0.960 | 0.78 → 0.78 | 4.9 → 4.9 | | |
| Starfruit | 16 → 11 | 0.956 → 0.960 | 0.82 → 0.84 | 5.2 → 3.1 | 0.251 → 0.277 | 0.256 |

- Every fix held for the whole match, with no drops.
- **The label check moves within its noise.** Fed from 0.1 s or 0.2 s instead of 0, bottom-left
  reads 0.936–0.941 with the rule off and 0.933–0.939 with it on.
- **The crate spread rose on every pair, by 0.014–0.026. The cause is the world origin.**
  - Outside a match the vote lands anywhere from 0.05 to 0.89, which moves where odometry's origin
    sits when the gate opens.
  - The localizer then fixes on another integer offset: on bottom-left, (−8, +2) instead of
    (−9, +3). The spread is then measured on another sub-tile phase.
  - With the rule on in a match only (`runs/map_checks/truth/overlay_fix/inmatch/`), bottom-left
    fixes on (−9, +3) again, and four of the five pairs come back within 0.005.
  - Bottom-right's +0.020 stays. That's inside the 0.034 that the feed start alone moves
    bottom-left's spread (0.180–0.214, rule off).
  - In a live session the origin depends on when the session started anyway.

**What the rule costs.**

- **A few ticks go the other way.** At 12 Hz in a match, the vote probe moves 133 Truth frame pairs
  from `uncertain` to `ok`, and 8 from `ok` to `uncertain` (frames 625–680 and 2435).
  - On those 8, a window over near-still ground reads just over `still_tiles`. Once the movers
    carry the estimate, it counts as a mover that disagrees.
  - The estimate is right, and the tick holds its action.
- **About 8 ms on a split frame:** 10.4 ms with the vote against 2.7 ms without, well inside the
  250 ms budget. Frames that don't split never vote.
- **The banner still hides the ground.** Its ticks deposit what the classifier reads under the
  letters, as most did before (59 of 84 were `ok`). Those deposits now land in the right place.
- **More of the emulator's x over-read shows** (the walk above). The old median, pulled toward the
  still windows, under-read the walk and hid part of the over-read. That's EMULATOR_RECTIFY_PLAN.md,
  the user's call.

**The label check's 30 disputed tiles before the overlay rule,** where at least 90% of views
disagree with the label.
Contact sheets: `runs/map_checks/truth/shipped/disputed_*.jpg`. Per-frame reads at each tile's
registered position come from `probe_cells.py`. None is a label error.

| tiles | label, seen | cause |
|---|---|---|
| (8,53) | floor, wall | A Power Cube Box. Boxes are never drawn on the map image. |
| (5,55) | floor, wall | A blue mailbox's top overhangs the tile north of its base. The label reads sprites at their base. |
| (36,1) (37,1) (18,3) (17,5) (17,6) (9,12) (7,13) (58,30) (59,31) (27,35) (45,48) (35,58) (35,59) | various | `truth` reads these right, or nearly, where the image registration puts them. The dispute is the localizer's offset: the x drift and the banner refix. (7,13) is water read as water, not as bush. (17,5)'s one water read is under the blue kill-feed banner. |
| (27,1) (15,5) (14,6) (59,32) (57,38) (54,41) (55,44) (48,47) | various | Never well in view. Each was seen in at most one registered frame, at the view's edge, or only in the windows the registration skips (the banner, the y jump). |
| (38,1) (39,1) (40,1) (56,21) (57,27) (54,36) (58,34) | wall, floor; (56,21) floor, bush | **Real misses, all at the edge of the view:** the top row and the east columns, where every label set is thin. |

**The 13 disputed tiles with the rule.** Four are in the table above: (7,13), (59,32), (55,44) and
(48,47). The other 26 cleared. Nine are new, and none of them is a label error either.

Each was traced vote by vote with two probes in `scripts/probes/map_label_projection/truth/`:

- `trace_deposits.py` lists every vote the map took on a tile.
- `crop_offset.py` gives each vote's d: how far its crop sat off the tile, by the image
  registration.

**How a localizer error reaches a vote.**

- The loop re-aims every frame onto the localizer's sub-tile position (`RectifyPlan.registered`).
- So a localizer error of d tiles moves each crop d off its tile.
- The crop then carries 1 − |d| of the tile and |d| of the neighbour on that side.
- Leaning 0.2–0.4 tile toward a wall to the east or west was enough to read `#`.

| tiles | label, seen | cause |
|---|---|---|
| (43,16) (46,18) | floor, wall | **The x scale error at the view's right edge.**<br>• Their crops sat 0.22–0.44 west, into the walls (42,16) and (45,18).<br>• Without the rule they sat on the tiles.<br>• (43,16) is a floor corner with walls west and south. It reads `#` 13 times of 29 even on the tile. |
| (18,59) (1,52) | floor, wall; wall, floor | **The x over-read on match 2's walk west,** before the sub-tile correction at 67 s.<br>• (18,59): with the rule, its crops sat 0.21–0.36 east, toward the wall (19,59), and read `#` 27 of 27. Without it they sat 0.07–0.15 west and read `.` 15 of 15. Row 59 is the map's bottom row, and on screen the out-of-bounds hedge covers most of it.<br>• (1,52): its crops, at the view's left edge, sat 0.63–0.68 east, mostly on the floor (2,52). Without the rule it took no votes. |
| (58,35) | wall, floor | **The localizer's y in match 2's first seconds,** 0.4 tile off until about 49 s.<br>• Both runs' crops sat 0.6–0.7 west, onto the wall (57,35).<br>• With the rule they also sat 0.4–0.5 north, across the stepped wall's top edge, and read floor twice.<br>• Without it they read `#` 4 of 4. |
| (25,19) | wall (teleporter), floor | **The teleporter misread** (below).<br>• All 35 crops that sat on the tile read floor.<br>• The pad's east half, (26,19), reads `#` 32 of 35 on the tile.<br>• Without the rule, its crops at 98.5–101.3 s leaned 0.26–0.40 east, toward the pad's centre, and read `#` 21 of 22. That kept it under 90%. |
| (28,2) | floor, wall | **One vote, mid-fight,** beside a Power Cube Box.<br>• The crop sat 0.38 west, into the wall (27,2).<br>• Without the rule its two crops leaned 0.34 east and read floor. |
| (1,30) | floor, wall | **One vote in the y-jump window,** which the registration skips.<br>• The crop holds the top of the wall step (1,31) south of it, in the gas.<br>• Without the rule the odometry was `uncertain` on that tick. |
| (45,3) | water, floor | **Two votes under the kill feed,** at 26.9 and 27.3 s, in the view's top-right cells.<br>• The hero was out at 26.8 s.<br>• The gate's exit hysteresis kept the phase `playing` until 27.3 s. |

- **The sub-tile anchor decides which side of the view is off.**
  - At 99–100 s, with the rule: the crops at view column 8 sat on their tiles, and those at column
    26 sat 0.39 west.
  - Without it: column 8 sat 0.35 east, and column 26 on its tiles.
  - Both runs fall 0.022 tile a column. That's the 2.2% x scale error.
  - The 0.37 between the runs is the same at both columns. That's where the localizer put the
    anchor.
  - The 30-column view spans 0.65 tile of that error. Wherever the anchor sits, one edge or the
    other is a third of a tile off or worse.
  - Four of the nine come from this 2.2%: two across the view and two along the walk.
  - EMULATOR_RECTIFY_PLAN.md is the user's call.
- **Five rest on one or two votes:** (28,2), (45,3), (46,18), (1,30) and (58,35).
  - `confidence()` gives a tile seen once 1.0, so one bad crop disputes it.
- **The kill-feed votes cost nothing live.**
  - The map keeps taking deposits for the hysteresis's 0.5 s after the hero is out.
  - Nothing it holds then reaches a decision that matters.

**Teleporter pads.** The classifier reads about half the pads as floor: in the replay's map, 12
of the 24 pads it saw. That's harmless while the localizer is fixed, which was 99–100% of both
matches. The policy then gets the map label (`KnownTerrain.best()`), where pads are walls.

**Selecting it.** A Truth session needs both the map and the classifier:

| to play Truth with | do this |
|---|---|
| a live session (Bull) | `./.venv/Scripts/python.exe scripts/deploy_run.py --map truth --run runs/mortis_ppo-20261103-092146 --terrain truth` |
| a replay | `./.venv/Scripts/python.exe scripts/probes/replay_clip.py <clip> --map truth --terrain truth --run runs/mortis_ppo-20261103-092146` |

Without `--terrain truth`, Truth runs on the default. That also fixes, but it reads floor as
blocking more often. `deploy_run.py` logs `terrain truth (terrain_truth.pt)` at startup.

**Open items.**

- **The user's:** M1 sign-off on `truth.csv`, M4 on held-out matches, M5 live, and whether
  `truth` becomes the default.
- **One recording.** The east half's label rests on the map image and the half-turn symmetry.
  Little footage shows it.

### Doom Shroom (2026-11-10)

| item | value |
|---|---|
| image | `brawl_deployment/data/maps/doom_shroom.png`, 976 x 971, from the user's screenshot (the source in `runs/map_checks/doom_shroom/`) |
| lattice | checker fit, pitch 15.751 x 15.723 px, origin (16.7, 16.05) |
| spawns, (col, row) | (33,2) (12,8) (49,12) (8,19) (30,22) (49,33) (28,38) the clip (6,40) (53,53) (26,55); pinned in `tests/test_known_map.py` |
| label | 60 x 60: floor 2513, wall 566, bush 409, water 102, spawn 10. The map has no symmetry, so every cell is read off the image. The 79 fence tiles were floor until later on 11-10 (floor 2592, wall 487), when the user made them walls. |
| recording | `bluestacks-doom-shroom-south-center.mp4`, copied from `Videos/2026-11-10 17-25-29.mp4`: 2560 x 1440, 30 fps, 2499 frames, one Shelly match. The gate opens at 17.27 s (frame 518) on (28,38), and "Defeated" shows at about 80 s. |

**Materials.** From the user: teleporters, bounce pads, heal pads and speed tiles are floor. Gray
walls, the pink walls, barrels and planter boxes are walls. Fences started as floor, a rough fit
the user meant to fix. Later on 11-10 the user found the agent struggling with them and made them
walls (**Classifier v3**, below). The map has one bounce pad, one heal pad and two teleporters.

| looks like | class | tiles | texture in `scripts/probes/map_label_projection/doom_shroom/textures.csv` |
|---|---|---|---|
| pink checker floor | `.` | 2425 | |
| speed tiles, a green pattern in two strips | `.` | 72 | speed |
| bounce pad, 2 x 2 with a yellow arrow | `.` | 4 | bounce |
| heal pad, 2 x 2 with a green cross | `.` | 4 | heal |
| teleporters, two 2 x 2 pads | `.` | 8 | teleporter |
| pink blocks with an emblem (a rose in the game) | `#` | 220 | pink_cross |
| plain pink blocks (crates with an X plank in the game) | `#` | 170 | pink_plain |
| gray blocks (lavender with a face in the game) | `#` | 12 | gray |
| barrels | `#` | 31 | barrel |
| planter boxes | `#` | 29 | planter |
| purple mailboxes (a star badge in the game) | `#` | 14 | mailbox. Not on the user's list: walls by Truth's precedent. |
| orange barriers, a row above the mailboxes at (47-51,24) and a column by the heal pad at (51,25) (51,29-33) | `#` | 11 | orange. Destructible in the game. The sim has no destructible walls, so from the user (later on 11-10): plain walls. |
| fences, brown and gold poles, in four lines; three posts stand in grass, at (20,48) (19,50) (17,58) | `#` | 79 | fence. Floor until later on 11-10, then walls (the user). |
| teal grass | `b` | 409 | |
| blue pools | `~` | 102 | |

**What the clip shows.** After the gate the match stays in the west, map x 0-49 and y 0-53.

- The heal pad is never in view.
- The west teleporter shows only in the pre-match intro (frames 0-150), and the east one never.
- Two of the orange barriers, (47,24) and (48,24), show in two frames, at the view's top edge.

No classifier label covers those three. The map label holds them as the table says.

**M3 with the existing classifiers: neither passes.**

- **The default (`terrain.pt`) never fixes.** Its best agreement anywhere stays near 0.51, under
  the 0.55 wrong-map floor. Five seconds after the gate the localizer calls the map wrong ("map
  mismatch: expected doom_shroom") and keeps that verdict for the whole match.
- **`truth` fixes at the gate, on the right spawn, then drops the fix three times.** It has no fix
  for 28% of the match, against the 5% gate. Agreement while fixed has a median of 0.73 and a 5th
  percentile of 0.53. Its label check reads 0.920 at (0, 0), and 81 of its 156 disagreements are
  floor read as wall.
- `map_localize_report.py --map doom_shroom --spawn 28,38` calls `truth`'s fix "a tile off"
  (-0.89, -0.34). The image registration says it's right: `loc_vs_image.py` holds the localizer's
  offset minus the registration's at a constant (+13.9, +10.2), the same constant Truth's
  verified replay gives. The report's truth is the spawn's centre minus the hero's odometry at the
  gate, and the report itself warns that the hero was already running then, at 2.6 tiles a
  second. It reads Truth's right fixes 0.58-0.72 tile off too.

**The projection's scale: the blocks, not the odometry.** The odometry fit read S = (1.010, 1.010)
over 191 samples, which left a block stretch of x -0.0105 across the frame. That put the frame
edge 0.16 tile off, over the tool's 0.12 limit. `--scale 1.021 1.011`, where the blocks pointed:

- left a stretch of x -0.0006, y +0.0003, with the frame edge 0.01 tile off;
- moved the NCC median from 0.664 to 0.668, and the block residuals from 0.11 / 0.09 to
  0.09 / 0.09 tile (x / y).

The x scale matches Starfruit's 1.022 on the same emulator (EMULATOR_RECTIFY_PLAN.md, the user's
call).

**M3b: projected labels, split by ground.** The hero walks the west edge twice, at frames 700-770
and 1370-1770, so a cut in time would put the same ground in both sets. Region R is the west
strip, map x 0-13 without the bounce pad, plus the planter patch at (21-23, 33-35). The held-out
set labels only R, and the training labels leave R `?`. R was chosen so that every texture the
match shows is in both sets, except the one bounce pad, which trains.

| set | frames | cells | map tiles |
|---|---|---|---|
| training, `tests/fixtures/vision/labels/` | 530 570 610 650 690 790 830 870 910 950 1000 1040 1080 1120 1160 1210 1250 1290 1330 1370 1410 1450 1520 1600 1680 1790 1830 1870 1910 1950 1990 2030 2070 2110 2150 2190 2230 2270 2310 2350 | 6976: floor 5622, wall 563, bush 444, water 347 | 1217, none in R |
| held out, `tests/fixtures/vision/labels_doom_shroom_test/` | 710 740 770 1470 1500 1550 1580 1630 1660 1710 1740 1770 | 1153: floor 902, wall 123, bush 89, water 39 | 362, all in R |

- **Cells / tiles by texture, training:** pink_cross 196 / 41, pink_plain 136 / 42, planter
  102 / 22, barrel 66 / 11, gray 49 / 8, mailbox 14 / 6; floor: speed 244 / 42, fence 242 / 45,
  bounce 13 / 3.
- **Held out:** pink_cross 71 / 29, pink_plain 23 / 7, barrel 16 / 3, planter 9 / 3, mailbox
  3 / 2, gray 1 / 1; floor: speed 57 / 18, fence 24 / 10. Gray and mailbox are too thin here to
  rank anything; the region check below counts them over the whole replay.
- **Left out of both sets** with `--leave`, until v3:
  - the 11 orange barriers, whose class the user had not named yet (walls, the user said later
    that day, so labels from later clips include them; this clip sees two, at the view's edge);
  - three fence posts standing in grass at (20,48) (19,50) (17,58), which may be bush.
- **These are the sets v1 and v2 trained and scored on.** After the fences became walls, both were
  rebuilt for v3 without either `--leave`, and the counts above changed (**Classifier v3**, below).
- The rebuild commands, v3's, are in the held-out README. Review sheets are in
  `runs/map_checks/doom_shroom/projection/bluestacks-doom-shroom-south-center/review/`.

**The `doom_shroom` classifier, shipped under its own name.** `vision_train_terrain.py` trained it
on `tests/fixtures/vision/labels/`: the 40 Doom Shroom frames plus every older label, Truth's
included, 172 frames. It used the defaults (200 epochs, seed 0) on the CPU, in 30 min; replays
sharing the CPU made it slower than Truth's 11. `terrain.pt` stays the default, because which
classifier is the default is the user's call.

This was v1. Later the same day `doom_shroom` became v2, trained with the second recording's labels
too (**Classifier v2**, below), and then v3, trained after the fences became walls (**Classifier
v3**). Every `doom_shroom` number from here to the second recording is v1's, scored against the
label with fences as floor.

| on the twelve held-out frames | blocking F1 | recall | precision | 5-class |
|---|---|---|---|---|
| default (`terrain.pt`) | 0.827 | 0.914 | 0.755 | 0.899 |
| `terrain_dark_passage.pt` | 0.660 | 0.669 | 0.652 | 0.830 |
| `truth` | 0.892 | 0.925 | 0.861 | 0.939 |
| **`doom_shroom`** | **0.994** | **0.994** | **0.994** | **0.996** |

- **The older classifiers miss this map's own floor and walls.**
  - The green speed strips read as bush or wall. The default reads 4 of 42 speed cells as floor,
    and `truth` 9 of 54. In `truth`'s replay map, of the 60 speed tiles it saw, 45 read as wall,
    4 as bush and 11 as floor.
  - Neither reads a planter box as a wall: 0 of 9 held-out cells, and 0 of 25 tiles in `truth`'s
    replay.
  - `doom_shroom` reads 0.947 of the speed cells as floor and every planter cell as wall. The
    held-out README has the per-class and per-texture tables.
- **Bush and water stay apart.** `doom_shroom` reads 0 of 83 bush cells as water and 0 of 39 water
  cells as bush. `terrain_dark_passage.pt` reads 27 of 36 water cells as bush.
- **The other maps' held-out sets lose nothing** (blocking F1):

  | held-out set | default | `truth` | `doom_shroom` |
  |---|---|---|---|
  | Truth, 10 frames | 0.908 | 0.990 | 0.989 |
  | Starfruit, 7 frames | 0.946 | 0.945 | 0.978 |

  Truth's own section quotes 0.982 and 0.907 for the same frames. Those were scored before the
  overlay rule, and `vision_score_map.py` builds its map through the odometry.

**Regression replays** (`runs/map_checks/doom_shroom/regression/`, by `regress.sh` in
`scripts/probes/map_label_projection/doom_shroom/`). Each cell reads default → `doom_shroom`; on
Truth, default → `truth` → `doom_shroom`. They replay with `replay_clip.py`'s default run, the
Mortis one, as Truth's did.

| clip | time fixed | agreement median (5th percentile) | corrections a minute | crate spread | label check |
|---|---|---|---|---|---|
| Dark Passage bottom-left | 100% → 100% | 0.77 (0.68) → 0.77 (0.68) | 13.1 → 8.7 | 0.230 → 0.174 | 0.933 → 0.943 |
| Dark Passage bottom-right | 100% → 100% | 0.82 (0.64) → 0.83 (0.68) | 7.2 → 10.9 | 0.198 → 0.194 | 0.964 → 0.967 |
| Dark Passage top-right | 100% → 100% | 0.78 (0.66) → 0.72 (0.65) | 4.9 → 4.9 | no crates | 0.960 → 0.955 |
| Starfruit | 100% → 100% | 0.84 (0.66) → 0.87 (0.65) | 3.1 → 5.2 | 0.277 → 0.272 | 0.960 → 0.955 |
| Truth | 100% → 100% → 100% | 0.77 (0.67) → 0.91 (0.74) → 0.93 (0.72) | 6.1 → 3.7 → 3.7 | 0.173 → 0.184 → 0.205 | 0.938 → 0.978 → 0.975 |

- **Every fix holds, and no gate newly fails.** Starfruit's crate spread was already over 0.25 with
  the default (0.277 since the overlay rule).
- **Top-right's agreement median falls from 0.78 to 0.72,** as it did with `truth`, but without
  `truth`'s 0.2 s drop. Both classifiers spend 0.1 s searching at the gate there.
- **Two clips correct more often.** Bottom-right goes from 6 to 9 sub-tile corrections at the same
  crate spread, and Starfruit from 3.1 to 5.2 a minute.
- **On Truth, `doom_shroom` matches `truth`:** the same corrections, agreement 0.93 against 0.91,
  and a label check of 0.975 against 0.978. Its crate spread is 0.205 against 0.184.

**M3 again, with `doom_shroom`** (`runs/map_checks/doom_shroom/shipped/`: `replay_clip.py --map
doom_shroom --terrain doom_shroom --run runs/mortis_ppo-20261103-092146`, through
`replay_seen.py`). It matches the replay that loaded the candidate by path in every column but
`tick_ms`, and its seen map is the same. `map_localize_report.py --map doom_shroom --spawn 28,38`:

| gate | `truth` | **`doom_shroom`** |
|---|---|---|
| spawn picked right | yes, (28,38) | yes, (28,38) |
| first fix after the gate (< 2 s) | 0.00 s | 0.00 s |
| time without a fix (< 5%) | **28%**: dropped 3 times, found again twice | 0%, no drops |
| crate spread (≤ 0.25) | 0.196 | 0.190 |
| agreement median (5th percentile, min) | 0.73 (0.53, 0.48) | 0.92 (0.80, 0.59) |
| corrections | 0 whole-tile, 2 sub-tile | 0 whole-tile, 3 sub-tile, 2.9 a minute |
| label check, best at (0, 0) | 0.920 over 1957 tiles | 0.977 over 2048 tiles |

The default never fixes, as above. `loc_vs_image.py` keeps `doom_shroom`'s offset minus the image
registration's within 0.41 tile of (+13.9, +10.2) over 179 registered frames
(`runs/map_checks/doom_shroom/m3/loc_vs_image.txt`), so the fix never moves to a wrong tile.
`truth`'s 124 registered frames stay within 0.42.

**The fix still sits up to half a tile off.** `crop_split.py` (below) puts the zero, where the fix
is exact at the view's centre, at (+13.93, +10.08). Against it, `doom_shroom`'s fix is up to 0.43
tile off in x (58.7 s) and 0.48 in y (57.0 s). An axis is more than 0.25 off in 70 of the 179
frames, 66 of them after 40 s. `truth`'s fix goes just past 0.5 in y in three frames (57.0, 65.0
and 65.3 s).

**Region R in the replay.** `region_score.py` in the probe folder scores each replay's seen map
against `doom_shroom.csv`, inside R and outside it. No training label exists inside R. Spawn tiles
count as floor.

| | `truth`, in R | **`doom_shroom`, in R** | `truth`, outside R | **`doom_shroom`, outside R** |
|---|---|---|---|---|
| tiles seen | 658 of 845 | 682 of 845 | 1299 of 2755 | 1366 of 2755 |
| 5-class | 0.954 | 0.974 | 0.903 | 0.979 |
| blocking F1 (recall, precision) | 0.875 (0.968, 0.798) | 0.940 (0.959, 0.922) | 0.764 (0.832, 0.706) | 0.948 (0.948, 0.948) |
| speed tiles read as floor | 6 of 22 | 25 of 27 | 5 of 38 | 44 of 44 |
| planters read as wall | 0 of 3 | 3 of 3 | 0 of 22 | 22 of 22 |
| barrels read as wall | 6 of 6 | **3 of 7** | 7 of 11 | 12 of 14 |

**The four barrels `doom_shroom` reads as floor in R were seen through crops that sat off them.**
`trace_deposits.py` (Truth's) listed every vote the map took on them. `crop_offset.py` (the probe
folder's copy) gave each vote's d, how far its crop sat off the tile. `crop_split.py` split d into
the fix's error at the view's centre and the scale term
(`runs/map_checks/doom_shroom/shipped/trace_barrels*.txt`).

- **No crop ever sat on (4,37) (4,38) (3,41).**
  - Their 42 votes came in the two walks along the west edge, at view columns 6-8 and rows 14-17.
    Every crop sat 0.3-0.5 tile east of its tile, and 0.1-0.5 north.
  - Each barrel is the east end of a short wall, with floor to its east.
  - In the first walk (24.4-25.0 s), the crops leaned 0.1-0.2 tile north, and (4,37) and (4,38)
    read wall 9 of 12 times. In the second (53.4-55.4 s), they leaned 0.4-0.5 north, and the three
    tiles read floor 26 of 30 times.
  - The fix's error gave 0.2-0.4 tile of the east lean. It gave 0.03-0.12 of the north lean in the
    first walk and 0.34-0.44 in the second. The scale term gave the rest, 0.12-0.16 east and
    0.05-0.08 north at those view cells (EMULATOR_RECTIFY_PLAN.md, the user's call).
- **(21,33) tipped the same way.** It took 253 votes. On the tile it reads wall 117 of 150. The 102
  crops leaning 0.25-0.5 north, onto the floor (21,32), read floor 96 times, so the tile tips to
  floor by 129 to 124. 110 of its 129 floor votes came after 40 s, when the fix sat 0.1-0.5 tile
  south.
- **The held-out frames read 15 of 16 barrel cells,** on (7,36), (7,37) and (21,33). None of them
  shows the three west-edge barrels, so how `doom_shroom` reads those on the tile is untested.
  While the fix holds, the policy gets the map label, so these reads cost only localization margin.

**The 16 disputed tiles,** where at least 90% of the replay's views disagree with the label. None is
a label error.

| tiles | label, seen | cause |
|---|---|---|
| (5,3) (24,3) (26,4) (4,10) (28,12) (2,17) | bush or floor, wall | **A block's top leans north.** Each sits just north of a wall base, and in the game view a block's top covers about a tile north of its base. The label reads sprites at their base. |
| (35,18) (47,24) (46,28) (47,36) (45,37); (45,34); (47,35) | wall, floor; floor, wall; bush, floor | **The view's east edge,** view columns 24.6-28.9 of 30 ((35,18) is at the top edge as well). Each reads as its west neighbour: the emulator's x over-read (EMULATOR_RECTIFY_PLAN.md, the user's call). Four were seen in only 1-3 frames. (47,24) is an orange sprite. |
| (25,13) | floor, wall | **A Power Cube Box,** read `#` in 18 frames; frame 1030 shows it. Boxes are never drawn on the map image. |
| (20,48) (19,50) | floor, bush | **Fence posts standing in grass,** `.` by the fence rule then. Walls since v3 (the user). |

#### Second recording: the east (2026-11-10)

| item | value |
|---|---|
| recording | `bluestacks-doom-shroom-north-center.mp4`, copied from `Videos/2026-11-10 20-06-26.mp4`: 2560 x 1440, 30 fps, 2786 frames, one Shelly match. The gate opens at 18.53 s on (33,2), and the match ends at 91.9 s. |
| coverage after the gate | 2389 tiles, map x 0-59 and y 0-57. All 11 orange barriers are in view (36-64 frames each), and so is the heal pad (47-48 frames a tile). The teleporters and the bounce pad aren't; the intro shows the west teleporter. |

**The orange barriers are walls** (the user, 11-10). They're destructible in the game, but the sim
has no destructible walls. `doom_shroom.csv` already held all 11 as `#`, so the label needed no
`--mark`, and this recording's labels include them.

**M3 with the shipped classifiers** (`runs/map_checks/doom_shroom/north_center/`:
`scripts/probes/map_label_projection/truth/replay_seen.py` with `--map doom_shroom --run
runs/mortis_ppo-20261103-092146`, then `map_localize_report.py --map doom_shroom --spawn 33,2`):

| gate | `truth` | `doom_shroom` (v1) |
|---|---|---|
| spawn picked right | yes, (33,2) | yes, (33,2) |
| first fix after the gate (< 2 s) | 0.07 s | 0.07 s |
| time without a fix (< 5%) | **24%**: dropped at 74.0 s, never found again | 0.1% |
| crate spread (≤ 0.25) | **0.278** | **0.345** |
| agreement median (5th percentile, min) | 0.81 (0.64, 0.57) | 0.90 (0.78, 0.57) |
| corrections | 0 whole-tile, 4 sub-tile | 0 whole-tile, 3 sub-tile, 2.5 a minute |
| label check, best at (0, 0) | 0.931 over 1926 tiles | 0.982 over 2310 tiles |

- **The report calls every fix "a tile off". The fix is right.** As on the first recording, the
  report's truth is the spawn's centre minus the hero's odometry at the gate, and the hero was
  already running, at 2.7 tiles a second. `loc_vs_image.py` (`CLIP=bluestacks-doom-shroom-north-center`)
  holds v1's offset minus the image registration's at a median of (+13.86, +10.26) over 216
  registered frames. That's the first recording's constant, (+13.9, +10.2).
- **The fix wanders by up to 0.6 tile.** `crop_split.py` puts this recording's zero at
  (+13.93, +10.00), the median of 1337 votes' K (the first recording's: (+13.93, +10.08)). Against
  it (`wander.py` in the probe folder), v1's fix is more than 0.25 tile off on an axis in 143 of
  215 registered frames, and at most 0.50 in all but six. Those six reach 0.51-0.60 in y at
  38.7-40.0 s and -0.52 in x at 76.0 s. `truth`'s fix does the same: 0.52-0.62 in y at 38.3-40.0 s,
  and -0.51 in x at 54.3 s.
- **v1's crate spread, 0.345, is that wander.** Each crate residual moves against the fix's error
  at that moment. Over 385 residuals the correlation is -0.84 in x and -0.86 in y, with slopes of
  -1.03 and -1.27. With the error added back, the spread is 0.171, inside the gate. No correction is
  whole-tile.
- **One more frame, 91.3 s, reads 2.27 tiles off in x.** That's the probe, not the fix.
  `loc_vs_image.py` interpolates the hero's position toward the 91.5 s row, where the hero has died
  ("no hero hp") and the row's hero offset is (0, 0). `wander.py` leaves it out.

**The projection's scale.** The odometry fit read S = (1.013, 1.010), a block stretch of x -0.0064
and y -0.0086, the frame edge 0.10 tile off. The first recording's S, (1.021, 1.011), left y
-0.0078 (0.07 tile). `--scale 1.021 1.019`, where the blocks pointed, left x +0.0006, y -0.0003
(0.01 tile), with an NCC median of 0.625 and block residuals of 0.09 / 0.09 tile. The x scale is the
first recording's; the y scale is 0.008 above it (EMULATOR_RECTIFY_PLAN.md, the user's call).

**M3b: the east strip held out.** Region R2 is map x 52-59. Neither recording's training labels
touch it, and this recording's also leave R `?`.

| set | frames | cells | map tiles |
|---|---|---|---|
| training, `tests/fixtures/vision/labels/` | 590 630 670 710 750 860 940 1070 1110 1180 1220 1270 1310 1350 1390 1430 1470 1530 1570 1610 1650 1690 1740 1780 1820 1860 1960 2000 2040 2080 2120 2160 2200 2240 2280 2320 2360 2400 2440 2480 2520 2560 2600 2640 2680 | 7421: floor 5784, wall 829, bush 486, water 322 | with the first recording's 40 frames, 1882 tiles, none in R or R2 |
| held out, `tests/fixtures/vision/labels_doom_shroom_east_test/` | 800 840 890 920 960 990 1020 1050 1090 1160 1250 1290 | 965: floor 564, wall 154, bush 190, water 57 | 315, all in R2 |

- **R2 starts at x 52 so that the new textures train.** The orange barriers and the heal pad sit
  at x 47-51. From x 50 the held-out set took 8 of the 11 barriers. From x 52, 9 barriers label in
  training, 35 cells. The other two, (51,29) and (51,30), border the heal pad and lose their
  labels to the gas refusal below.
- **The frames are `pick_frames_east.py`'s, with two changes.** Three of its held-out frames, 1880,
  1910 and 1940, came after the gas reached R2 and kept 2-3 cells each. Frames 960, 990 and 1020
  replaced them, and the two training frames then too near one, 980 and 1020, were dropped. Every
  training frame is at least 20 frames from a held-out one.
- **Left out of both sets, until v3:** the three fence posts standing in grass, with `--leave` as
  before.
- The rebuild commands are in the east README. Review sheets are in
  `runs/map_checks/doom_shroom/projection/bluestacks-doom-shroom-north-center/review/`.

**The heal pad never labels, and the live loop never reads it as gas.** The projection refuses any
cell at least 5% gas-coloured (`label_projection.GAS_FRACTION`). The pad's green cross fills
0.2-0.4 of each of its tiles, so all four refuse. The live loop refuses the same cells at the same
5% (`zone.at_least(0.05)`) before the classifier's reading reaches the occupancy map or the
localizer, so a label there would train an answer the loop throws away. The policy's gas map is
the other test: `GasMap` latches a cell at 60% (`zone.min_cell_fraction`). Run on every tenth frame
of each recording (`detect_zone` and the registration):

- the pad peaks at 0.41, on (49,29) at 69.0 s;
- no on-map cell latches before the real gas. The first latch is at 37.7 s here, (59,25), 19.1 s
  after the gate, and at 41.7 s on the first recording, (17,1). Both are on the map's outer ring.

**The 20 disputed tiles in v1's replay,** where at least 90% of the views disagree with the label.
None is a label error.

| tiles | label, seen | cause |
|---|---|---|
| (22,2) (23,2) (24,3) (25,3) (27,11) | floor, wall | **A block's top leans north,** as on the first recording. All are seen at 18.7-20.3 s, just after the fix. |
| (59,7) (59,23) (37,56); (35,48); (57,53) | wall, floor; floor, wall; bush, wall | **The crops sat 0.25-0.5 tile toward a neighbour of the other class.** |
| (51,30) (51,31) (51,32) (51,33) | wall, floor | **Orange barriers,** which no label had trained. |
| (3,29) (2,30) (3,30) | floor, wall | **Speed tiles in R as the gas arrives,** at 86.8-88.9 s. The crops leaned 0.4 tile east, toward the gray wall at (4,29-30). For (2,30) that lean is onto floor, so it's a misread. |
| (21,33) | wall, floor | **The planter-patch barrel,** as on the first recording. |
| (59,21) | floor, wall | **A Power Cube Box on (58,21).** 10 votes, every crop 0.32 tile west, onto the box. The crate gate (`loot.crate_occlusion`) blanks only the cells a detected box covers, and this one borders it. |
| (40,52) | floor, wall | **One vote.** |

#### Classifier v2 (2026-11-10)

**`doom_shroom` loaded v2 until v3 replaced it** (sha256 a4cd95f7...). It's v1's recipe (200
epochs, seed 0, on the CPU) on every label, 217 frames: v1's 172 plus this recording's 45 training
frames. It trained in 37 min. Its log and scores are in `runs/terrain_doom_shroom2/`. Its weights
and v1's were deleted when v3 shipped (the user). Every number in this section is scored against
the label with fences as floor.

**Held-out sets**, blocking F1 (recall, precision) and 5-class:

| held-out set | v1 | **v2** |
|---|---|---|
| Doom Shroom west, 12 frames | 0.994 (0.994, 0.994); 0.996 | **1.000** (1.000, 1.000); 0.996 |
| Doom Shroom east, 12 frames | 0.968 (0.951, 0.985); 0.984 | **0.971** (0.954, 0.989); 0.987 |
| Truth, 10 frames | 0.989 (0.994, 0.984); 0.996 | 0.990 (1.000, 0.981); 0.997 |
| Starfruit, 7 frames | 0.978 (0.973, 0.984); 0.994 | **0.952** (0.968, 0.937); 0.986 |

- **Every barrel cell now reads as wall:** 6 of 6 east (v1: 2 of 5) and 16 of 16 west (v1: 15 of 16).
  The east set's water still reads as bush in 8 of 52 cells (v1: 7 of 56). The lowest texture is
  the west set's speed strips, 0.930 (v1: 0.947), with 4 of 57 cells read as bush.
- **Starfruit's is the one score that fell,** by 0.026, almost all of it precision: about 12
  non-blocking cells read as blocking, against v1's 3. It's still above the default's 0.946, the
  classifier trained for Starfruit. No session runs `doom_shroom` there: the default serves
  Starfruit.

**Regression replays** (`regress.sh cand2=doom_candidate2`). Every replay keeps its fix, and no
gate fails that didn't with the default. Each cell reads default → v1 → v2:

| clip | agreement median (5th percentile) | corrections a minute | crate spread | label check |
|---|---|---|---|---|
| Dark Passage bottom-left | 0.77 (0.68) → 0.77 (0.68) → 0.78 (0.69) | 13.1 → 8.7 → 13.1 | 0.230 → 0.174 → 0.182 | 0.933 → 0.943 → 0.937 |
| Dark Passage bottom-right | 0.82 (0.64) → 0.83 (0.68) → 0.83 (0.72) | 7.2 → 10.9 → 12.1 | 0.198 → 0.194 → 0.230 | 0.964 → 0.967 → 0.961 |
| Dark Passage top-right | 0.78 (0.66) → 0.72 (0.65) → 0.76 (0.69) | 4.9 → 4.9 → 4.9 | no crates | 0.960 → 0.955 → 0.960 |
| Starfruit | 0.84 (0.66) → 0.87 (0.65) → 0.85 (0.65) | 3.1 → 5.2 → 3.1 | **0.277 → 0.272 → 0.299** | 0.960 → 0.955 → 0.952 |
| Truth | 0.77 (0.67) → 0.93 (0.72) → 0.90 (0.75) | 6.1 → 3.7 → 3.7 | 0.173 → 0.205 → 0.187 | 0.938 → 0.975 → 0.973 |

Every clip holds its fix from the gate to the end, except top-right's 0.1 s search at the gate,
which all three models make. Truth's one whole-tile correction, at the second match's start, is
in all three. Bottom-right's crate spread rises to 0.230, inside the gate.

**M3 on both recordings** (`runs/map_checks/doom_shroom/m3_v2/`). The replays loaded v2 by path,
as `doom_candidate2`, from the file the shipped one copied byte for byte.
`map_localize_report.py --map doom_shroom --spawn 28,38` and `--spawn 33,2`:

| gate | south-centre, v1 | **south-centre, v2** | north-centre, v1 | **north-centre, v2** |
|---|---|---|---|---|
| spawn picked right | yes | yes | yes | yes |
| first fix after the gate (< 2 s) | 0.00 s | 0.00 s | 0.07 s | 0.00 s |
| time without a fix (< 5%) | 0% | 0% | 0.1% | 0% |
| crate spread (≤ 0.25) | 0.190 | **0.133** | **0.345** | **0.333** |
| agreement median (5th percentile, min) | 0.92 (0.80, 0.59) | 0.93 (0.86, 0.78) | 0.90 (0.78, 0.57) | 0.92 (0.85, 0.73) |
| corrections, all sub-tile | 3, 2.9 a minute | 1, 1.0 a minute | 3, 2.5 a minute | 3, 2.5 a minute |
| label check, best at (0, 0) | 0.977 over 2048 tiles | 0.983 over 2050 | 0.982 over 2310 | 0.984 over 2324 |

- **The fix wanders less.** Against `crop_split.py`'s zeros (`wander.py`), v2's fix is more than
  0.25 tile off on an axis in 52 of 181 registered frames on the first recording and 86 of 215 on
  the second, and never more than 0.45. v1's was in 70 of 179 (at most 0.48) and 143 of 215 (six
  frames past 0.5). The error's median is (-0.06, +0.09) and (-0.05, +0.11).
- **The second recording's crate spread is still that wander,** and still over the gate. Over the
  391 residuals with a registered frame within 0.2 s, each follows the fix's error (correlation
  -0.76 in x, -0.93 in y). With the error added back, the spread is 0.173 (v1: 0.171).
- **The report's "a tile off" is again its own truth:** the hero was running at the gate, at 2.6
  and 2.7 tiles a second.

**Region scores in the replays** (`region_score.py`, `--r2` on the second recording), v1 → v2:

| | south-centre, R | south-centre, outside R | north-centre, R | north-centre, R2 | north-centre, the rest |
|---|---|---|---|---|---|
| blocking F1 | 0.940 → **0.955** | 0.948 → 0.952 | 0.921 → **0.852** | 0.936 → **0.977** | 0.964 → 0.977 |
| 5-class | 0.974 → 0.981 | 0.979 → 0.984 | 0.978 → 0.953 | 0.969 → 0.986 | 0.986 → 0.987 |

- **The west-edge barrels in R read as wall,** 7 of 7 tiles on the first recording (v1: 3 of 7).
  v2's fix sits nearer the tile, and v2 trained on 45 more frames. This replay can't say which
  of the two did it.
- **The orange barriers read as wall** on 7 of the 9 tiles v2's second replay holds (v1: 5 of
  11). The two misses, (51,25) and (51,30), carry 0.55 and 0.52 of their views. (51,30) borders
  the heal pad and never labelled. The first recording sees one barrier, (47,24), at the view's
  edge, and both versions read it as floor.
- **R2's pink crosses read 40 of 44** (v1: 37 of 44), and R2's blocking precision is 1.000.
- **R on the second recording fell,** 0.921 to 0.852. That recording sees R only in the match's
  last 22 s (70-92 s), as the gas closes, and v2's replay holds 19 of R's wall tiles. It misses
  four:
  - the barrel (21,33), as v1 did;
  - the planter (23,34);
  - the mailboxes (8,24) (as bush) and (1,26), both in view only from 85 s.

  On the first recording, which crosses R all match, R rose.
- **The heal pad reads as bush** on 2 of its 3 seen tiles (v1: 1 of 3). No label covers it (the
  gas refusal, above), and the views the refusal lets through read its green cross as bush.

#### Classifier v3 (2026-11-11)

**`doom_shroom` loads v3** (sha256 0353bfa7...). Late on 11-10 the user made the map's fences
walls: the agent struggled with them live, and the map label had them as floor. All 79 fence tiles
in `doom_shroom.csv` went from `.` to `#`, and every Doom Shroom label was projected again from the
new map. v3 trained on the result with v2's recipe (200 epochs, seed 0, on the CPU, 217 frames) in
43 min. Its log and scores are in `runs/terrain_doom_shroom3/`, and its replays are in
`runs/map_checks/doom_shroom/m3_v3/`. v1's and v2's weights were deleted (the user). Their logs and
scores stay in their folders. Every number in this section is scored against the new label.

**The relabel reaches the policy without the classifier.** Once the localizer fixes, the policy's
static planes and the move mask are the label's cells, so the fences block from the first fix on.
The classifier's part is keeping that fix.

**The relabel alone breaks v2.** v2 reads every fence as floor, and the localizer now compares
those reads with a label that says wall. Replayed against the new label
(`runs/map_checks/doom_shroom/m3_v2new/`), v2 loses the first recording's fix at 44.93 s and
searches for 6.7 s. That's 10.7% of the match, which fails the 5% gate. On the second recording
it keeps the fix.

**The rebuilt labels** use the same frames and splits. The fence-post `--leave` is gone.

- **Training:** 46860 → 46712 labelled cells, with floor 32296 → 31930 and wall 7643 → 7861. The
  Doom Shroom training frames hold 171 fence cells, all now wall.
- **The west held-out set:** 1153 → 1123 cells. 11 fence cells went from floor to wall. 30 cells
  that straddle a fence line went to `?`.
- **The east held-out set:** R2 holds no fence, so no cell changed.

**Held-out sets**, blocking F1 (recall, precision) and 5-class:

| held-out set | v2 | **v3** |
|---|---|---|
| Doom Shroom west, 12 frames | 0.967 (0.936, 1.000); 0.986 | **0.982** (0.971, 0.994); **0.991** |
| Doom Shroom east, 12 frames | **0.971** (0.954, 0.989); **0.987** | 0.959 (0.995, 0.926); 0.980 |
| Truth, 10 frames | 0.990 (1.000, 0.981); 0.997 | 0.987 (1.000, 0.975); 0.996 |
| Starfruit, 7 frames | 0.952 (0.968, 0.937); 0.986 | 0.942 (0.995, 0.895); 0.982 |

- **Fences:** all 11 west fence cells read as wall (v2: none).
- **The east set's water** reads as bush in 1 cell (v2: 8).
- **The east set's precision fell:** 15 floor cells read as wall (v2: 1). `cell_diff.py` (the
  Doom Shroom probe folder) places them:
  - 11 sit on the tile above or beside a tall sprite: the pink walls at (53-54, 18) and (53, 45),
    the barrels at (52-53, 33) and the orange barrier at (51, 24).
  - 3 sit above the Power Cube Box at (58, 21) in frame 990. A box blocks in the game, but no
    label holds one.
  - (53, 11) is the last. v2 misses it too.
- **Barrels are the one texture that fell on a held-out set:** 10 of 15 west cells read as wall
  (v2: 16 of 16). The misses are (21, 33), read as bush in 3 of 4 frames, and (7, 37), read as
  floor in 2 of 5. In the replays below, v3 reads 37 of the 39 barrel tiles it sees as wall (v2:
  35 of 40).
- **Starfruit fell 0.010,** in precision again. The default, which serves Starfruit, scores
  0.946.

**Regression replays** (`regress.sh cand3=doom_candidate3`). Every replay keeps its fix. Each cell
reads v2 → v3:

| clip | agreement median (5th percentile) | corrections a minute | crate spread | label check |
|---|---|---|---|---|
| Dark Passage bottom-left | 0.78 (0.69) → 0.72 (0.64) | 13.1 → 23.9 | 0.182 → 0.226 | 0.937 → 0.918 |
| Dark Passage bottom-right | 0.83 (0.72) → 0.79 (0.60) | 12.1 → 14.6 | 0.230 → 0.210 | 0.961 → 0.955 |
| Dark Passage top-right | 0.76 (0.69) → 0.74 (0.62) | 4.9 → 4.9 | no crates | 0.960 → 0.949 |
| Starfruit | 0.85 (0.65) → 0.84 (0.70) | 3.1 → 5.2 | **0.299 → 0.268** | 0.952 → 0.957 |
| Truth | 0.90 (0.75) → 0.88 (0.74) | 3.7 → 3.7 | 0.187 → 0.178 | 0.973 → 0.970 |

- **Bottom-right drops its fix once,** at 18.33 s, and has it back 0.4 s later. That leaves 1.1% of
  the match without one, inside the gate.
- **Dark Passage reads worse than with v2 or the default.** The default's bottom-left label check
  is 0.933. No session runs `doom_shroom` there: Dark Passage sessions run `--terrain
  dark_passage`.
- **Starfruit's crate spread** misses the gate with every model (default 0.277).

**M3 on both recordings** (`map_localize_report.py --map doom_shroom --spawn 28,38` and `--spawn
33,2`). Each cell reads v2 → v3, both against the new label:

| gate | south-centre | north-centre |
|---|---|---|
| spawn picked right | yes → yes | yes → yes |
| first fix after the gate (< 2 s) | 0.00 s → 0.00 s | 0.00 s → 0.00 s |
| time without a fix (< 5%) | **10.7% → 0%** | 0% → 0% |
| crate spread (≤ 0.25) | 0.133 → 0.207 | **0.331 → 0.257** |
| agreement median (5th percentile, min) | 0.89 (0.60, 0.50) → 0.91 (0.76, 0.69) | 0.91 (0.80, 0.71) → 0.87 (0.80, 0.49) |
| corrections, all sub-tile | 1.1 → 2.9 a minute | 2.5 → 1.6 a minute |
| label check, best at (0, 0) | 0.948 over 2019 tiles → 0.971 over 2039 | 0.971 over 2326 → 0.969 over 2307 |

- **The fix wanders more than v2's** (`wander.py`). It sits more than 0.25 tile off on an axis in
  74 of 179 registered frames on the first recording and 98 of 216 on the second. v2, on the old
  label, did so in 52 of 181 and 86 of 215.
  - **v3's largest error** is 0.49 on the first recording. On the second it passes 0.5 in nine
    frames, at most 0.55: seven in y at 39.7-42.0 s, and two in x at 90.7-91.0 s, as the match
    ends. v2's never passed 0.45.
  - **The y medians are +0.12 and +0.18** (v2: +0.09 and +0.11). Whether the tiles above tall
    sprites, which v3 reads as wall, pull the fix up is untested.
- **The second recording's crate spread is still that wander.** The residuals follow the fix's
  error (correlation -0.67 in x, -0.84 in y). With the error added back the spread is 0.172 (v2:
  0.173).

**Region scores in the replays** (`region_score.py`), v2 → v3:

| | south-centre, R | south-centre, outside R | north-centre, R | north-centre, R2 | north-centre, the rest |
|---|---|---|---|---|---|
| blocking F1 | 0.877 → **0.916** | 0.852 → **0.954** | 0.780 → 0.769 | **0.983** → 0.908 | 0.933 → **0.966** |
| 5-class | 0.955 → 0.953 | 0.945 → **0.979** | **0.932** → 0.902 | **0.988** → 0.943 | 0.972 → **0.985** |

- **Fences read as wall** on 55 of the 61 fence tiles the first recording sees, and on 24 of 27 on
  the second (v2: none of 65, none of 30). Two of the misses are the posts in grass, (20, 48) and
  (19, 50), read as bush.
- **The orange barriers read as wall** on all 11 in the second replay (v2: 9).
- **R2 fell,** 0.983 to 0.908.
  - Seven pink-cross tiles on the map's east edge (x 58-59) read as floor. v2 misses three of R2's
    pink-cross tiles, one of them on that edge. On the held-out set, which places each crop by the
    image registration, v3 reads all 73 of R2's pink-cross cells as wall. So the edge misses come from views the held-out frames don't hold,
    or from where the fix put the crops. The replay can't say which.
  - Six floor tiles around the Power Cube Box at (58, 21) read as wall.
- **R on the second recording** reads 15 floor tiles as wall (v2: 3). That recording sees R only
  in the match's last 22 s, as the gas closes.
- **The heal pad reads as bush** on all 3 of its seen tiles (v2: 2 of 3).

**Selecting it.** A Doom Shroom session needs both the map and the classifier:

| to play Doom Shroom with | do this |
|---|---|
| a live session (Bull) | `./.venv/Scripts/python.exe scripts/deploy_run.py --map doom_shroom --run runs/mortis_ppo-20261103-092146 --terrain doom_shroom` |
| a replay | `./.venv/Scripts/python.exe scripts/probes/replay_clip.py <clip> --map doom_shroom --terrain doom_shroom --run runs/mortis_ppo-20261103-092146` |

Run them from the repo root. In PowerShell a bare `python` is the Microsoft Store alias, which
prints "Python was not found", so name the venv's interpreter.

Without `--terrain doom_shroom`, the session runs on the default, which never fixes here. Five
seconds after the gate the localizer calls the map wrong and keeps that verdict for the match, so
the policy never gets the map label. `deploy_run.py` logs `terrain doom_shroom
(terrain_doom_shroom.pt)` at startup.

**Open items.**

- **The user's:** M1 sign-off on `doom_shroom.csv`
  (`./.venv/Scripts/python.exe scripts/map_label.py doom_shroom --review`), M4 on held-out
  matches, M5 live, and which classifier is the default.
- **What neither recording sees after the gate:** 385 tiles, nearly all in the south rows 54-59,
  plus row 0 at x 9-18. Both teleporter pairs, (6-7, 57-58) and (20-21, 57-58), are among them,
  so the teleporter is the one texture no label has trained. Every other texture there trains
  elsewhere on the map. While the fix holds, the policy gets the map label there.

### Rockwall Brawl (2026-11-11)

| item | value |
|---|---|
| image | `brawl_deployment/data/maps/rockwall_brawl.png`, 978 x 977, from the user's screenshot (the source is `runs/map_checks/rockwall_brawl/source_screenshot.webp`) |
| lattice | checker fit, pitch 15.746 x 15.733 px, origin (16.7, 16.6) |
| spawns, (col, row) | (23,2) (37,2) (2,22) (57,22) (23,29) (37,29) the clip (2,36) (57,36) (23,57) (37,57); pinned in `tests/test_known_map.py` |
| label | 60 x 60: floor 2012, wall 554, bush 1024, spawn 10. No flip or half-turn maps the label onto itself (744 cells differ under the best), so every cell is read off the image. |
| recording | `bluestacks-rockwall-brawl-center-east.mp4`, copied from `Videos/2026-11-11 15-32-46.mp4`: 2560 x 1440, 30 fps, 3886 frames, one Shelly match, which the hero wins. The gate opens at about 15.2 s (frame 458) on (37,29). |

**Materials.** From the user: the map has no water and no floor specials, only bushes and walls,
and its destructible cones are walls. Everything is a shade of blue or blue-white, which is what
makes it hard for the older classifiers.

| looks like | class | tiles | texture in `scripts/probes/map_label_projection/rockwall_brawl/textures.csv` |
|---|---|---|---|
| dark blue floor | `.` | 2012 | |
| rock blocks, cyan to white-blue | `#` | 536 | cube |
| destructible cones: rows of three at (29-31, 6) and (29-31, 53), columns of three at (6, 28-30) and (53, 28-30), and singles at (19, 25) (41, 25) (19, 29) (41, 29) (19, 33) (41, 33) | `#` | 18 | cone. The sim has no destructible walls, so from the user: plain walls. |
| white-lavender bushes | `b` | 1024 | |

**What the clip shows.** After the gate the hero crosses most of the map. 855 tiles are never in
view after it: 634 of them are the bush border along the map's edges (rows 54-59 and columns 0-5
and 55-59), plus 171 floor, 45 wall and 5 spawn tiles. Every cone is seen.

**M3 with the existing classifiers: none fixes.** The default, `truth` and `doom_shroom` each
search for 5 s after the gate, then call the map wrong ("map mismatch: expected rockwall_brawl", at
20.6 s) and keep that verdict for the whole match. Their best agreement anywhere is 0.25, 0.25 and
0.26 (medians 0.17, 0.19, 0.17), far under the 0.55 wrong-map floor. So none of them gives the
policy the map label here (`runs/map_checks/rockwall_brawl/m3/`).

**The projection's scale: the blocks, not the odometry.** The odometry fit read S = (1.024, 1.007),
which left a block stretch of x -0.0023, y -0.0130 across the frame. That put the frame edge 0.12
tile off, at the tool's limit. `--scale 1.026 1.020`, where the blocks pointed, left a stretch of
x -0.0012, y -0.0003, with the edge 0.02 tile off. The NCC median moved from 0.674 to 0.675, and
the block residuals from 0.08 / 0.08 to 0.08 / 0.06 tile (x / y). The x scale sits near Starfruit's
1.022 and Doom Shroom's 1.021 on the same emulator, but y reads 1.020 here against Doom Shroom's
1.011 (EMULATOR_RECTIFY_PLAN.md, the user's call).

**M3b: projected labels, split by ground.** The hero sees the north strip cleanly only once,
walking west below it at frames 880-1330, before the gas covers it. When the hero comes back, at
frames 2640-2910, the gas covers it. So region R is map rows 0-9, the whole width. The held-out set
labels only R, and the training labels leave R `?`.

| set | frames | cells | map tiles |
|---|---|---|---|
| training, `tests/fixtures/vision/labels/` | 76, about every 40 frames from 470 to 3850 (the held-out README lists them) | 13081: floor 9736, wall 2382, bush 963 | 1834, none in R |
| held out, `tests/fixtures/vision/labels_rockwall_brawl_test/` | 880 910 960 1000 1040 1070 1110 1190 1270 1300 1330 | 1393: floor 850, wall 191, bush 352 | 369, all in R |

- **Walls by texture:** training holds cube 2360 cells on 357 tiles and cone 21 cells on 7 tiles.
  The held-out set holds cube 191 cells on 52 tiles, and **no cone.** R's three cones, at
  (29-31, 6), sit at the top left of every held-out view until they break. There they're under the
  brawlers-left text or the kill card, or past the view's edge, and the projection refuses those
  cells. Frame 1110 already shows them broken. So no held-out number here scores a cone, and the
  region check below counts them over the replay.
- **Neither set takes a frame near a screen-wide overlay:** within 10 frames of the clip's four
  short magenta tints (747-753, 1617-1638, 1680-1686, 3789-3798), or showing the "SHOWDOWN!" banner
  (two brawlers left, frames 3319-3469).
- **Ground that breaks mid-match is left `?` from then on** (section 1, M3b). The clip breaks 16
  cells: 8 cones, 4 cubes and 4 bushes. `broken_cells.py` in the probe folder found them, on a
  projection made without `--broken`, and the frames were settled on its crop sheets. Every label
  takes `--broken` with these (cell@frame), each a frame that shows it broken, after the last
  labelled frame that shows it standing:

  | ground | cells |
  |---|---|
  | cones | (29,6) (30,6) (31,6) @1110, (41,33) @2100, (41,25) @2640, (19,25) @3010, (19,33) @3300, (41,29) @3510 |
  | cubes | (26,35) @3340, (26,32) @3420, (25,32) @3510, (26,36) @3630 |
  | bushes | (23,34) (23,35) @3340, (29,33) (29,34) @3420 |

  Run on the final labels, the probe finds no labelled tile out of its standing range to its last
  view. Three bush tiles leave it for a few frames and come back: the hero walking through a bush,
  which the game draws see-through around him.
- The rebuild commands are in the held-out README. Review sheets are in
  `runs/map_checks/rockwall_brawl/projection/bluestacks-rockwall-brawl-center-east/review/`.

**The `rockwall_brawl` classifier, shipped under its own name.** `vision_train_terrain.py` trained it
on `tests/fixtures/vision/labels/`: the 76 Rockwall Brawl frames plus every older label, both Doom
Shroom recordings' included, 293 frames. That is `doom_shroom`'s set plus Rockwall's. It used the
defaults (200 epochs, seed 0) on the CPU, at below-normal priority, in 44 min. Its log and scores
are in `runs/terrain_rockwall_brawl/` (sha256 9c95be49...). `terrain.pt` stays the default,
because which classifier is the default is the user's call.

| on the eleven held-out frames | blocking F1 | recall | precision | 5-class |
|---|---|---|---|---|
| default (`terrain.pt`) | 0.041 | 0.064 | 0.031 | 0.556 |
| `terrain_dark_passage.pt` | 0.038 | 0.058 | 0.028 | 0.562 |
| `truth` | 0.032 | 0.047 | 0.025 | 0.571 |
| `doom_shroom` | 0.034 | 0.052 | 0.025 | 0.554 |
| **`rockwall_brawl`** | **0.987** | **1.000** | **0.974** | **0.996** |

- **The older classifiers read bush as wall and wall as bush.** Each reads every held-out bush
  cell as wall, and 93-95% of the cube cells as bush. That swap is why none of them fixes: their
  reads disagree with the label wherever the map has bush or wall.
- **`rockwall_brawl` reads every wall and bush cell it answers right:** all 191 wall cells and 348
  of the 352 bush cells. The scorer skips the 8 cells it leaves unanswered (4 bush, 4 floor). Its
  six misses are floor beside a block or a bush, or just north of one. The held-out README lists
  them.
- **The other maps' held-out sets lose nothing.** Blocking F1 (recall, precision); 5-class:

  | held-out set | `doom_shroom` | **`rockwall_brawl`** |
  |---|---|---|
  | Doom Shroom west, 12 frames | 0.982 (0.971, 0.994); 0.991 | 0.997 (0.994, 1.000); 0.998 |
  | Doom Shroom east, 12 frames | 0.959 (0.995, 0.926); 0.980 | 0.971 (0.954, 0.988); 0.988 |
  | Truth, 10 frames | 0.987 (1.000, 0.975); 0.996 | 0.986 (0.990, 0.981); 0.995 |
  | Starfruit, 7 frames | 0.942 (0.995, 0.895); 0.982 | 0.952 (0.962, 0.942); 0.985 |

**M3 again, with `rockwall_brawl`** (`runs/map_checks/rockwall_brawl/m3_cand/`: `replay_seen.py`
with the candidate loaded by path). `map_localize_report.py --map rockwall_brawl --spawn 37,29`:

| gate | default, `truth`, `doom_shroom` | **`rockwall_brawl`** |
|---|---|---|
| spawn picked right | never fixes: "map mismatch" from 20.6 s | yes, (37,29) |
| first fix after the gate (< 2 s) | never | 0.00 s |
| time without a fix (< 5%) | 100% | 0%, no drops |
| crate spread (≤ 0.25) | none | 0.246 |
| agreement median (5th percentile, min) | best anywhere 0.25-0.26 | 0.84 (0.74, 0.52) |
| corrections | none | 0 whole-tile, 8 sub-tile, 4.2 a minute |
| label check, best at (0, 0) | 0 tiles | 0.949 over 2679 tiles |

- **The report calls the fix wrong, and it isn't.** It finds the fix 2.4 tiles off in y. Its truth
  is the spawn's centre minus the hero's odometry at the gate, and it warns that the hero was
  already running then, at 2.9 tiles a second. `loc_vs_image.py` holds the localizer's offset
  minus the image registration's at x +13.43 to +14.36, y +9.60 to +10.54, over 333 registered
  frames. That is the constant Doom Shroom's and Truth's verified replays give, (+13.93, +10.08),
  so the fix sits on the right tile all match. The label check's best offset, (0, 0), says the
  same.
- **The fix still wanders up to half a tile.** Against (+13.93, +10.08) it is up to 0.50 tile off
  in x and 0.48 in y, never past 0.5. An axis is more than 0.25 off in 196 of the 333 frames
  (Doom Shroom v3: 74 of 179 and 98 of 216).
  - The landing fix sits about 0.45 tile off in y.
  - Sub-tile corrections at 25.3 s and 42.8 s move it 0.23 and then 0.27 tile.
- **The crate spread passes by 0.004.**

**Region R in the replay** (`region_score.py --broken`, which leaves the 16 broken cells out). No
training label exists inside R. Spawn tiles count as floor.

| | R | outside R |
|---|---|---|
| tiles seen | 434 of 597 | 2229 of 2987 |
| 5-class | 0.931 | 0.955 |
| blocking F1 (recall, precision) | 0.824 (0.875, 0.778) | 0.912 (0.902, 0.922) |
| cubes read as wall | 49 of 56 | 387 of 423 |
| cones read as wall | none seen | 1 of 7 |
| bushes read as bush | 127 of 142 | 205 of 230 |

- **R's misses fit crops sitting about half a tile south of their tiles.**
  - Three cubes read as the bushes south of them, at (26-28, 6).
  - Four cubes at rows 8-9 read as the floor south of them: (45,8) (20,9) (40,9) (46,9).
  - Seven bushes in row 1 read as wall, most of them just north of a block.

  The hero sees R at frames 880-1330, 29-44 s, while the landing fix's y error lasts. On the
  held-out frames, where the image registration places each crop, `rockwall_brawl` reads all of
  R's cubes and bushes right.
- **Six of the seven cones the replay sees read as floor.** Each is a glimpse at the view's edge.
  - (53,28) and (53,29) are in view after the gate only at frames 2420-2430, when they're already
    broken. Frame 580 shows them standing at the edge.
  - (29,53) and (30,53) are in view only at frames 2270-2280, at the bottom edge.
  - (53,30) and (6,30) are in view in 6 and 13 frames, at the side edges.

  The one cone seen in the open, (19,29), reads as wall over 76 frames. Cones that broke where no
  label looked, like (53,28) and (53,29), are missing from the `--broken` list, but no label names
  them either.

**Regression replays** (`regress.sh cand=rock_cand`, in `runs/map_checks/rockwall_brawl/regression/`).
They replay the three Dark Passage tuning clips, Starfruit, Truth and both Doom Shroom recordings.
Every replay keeps its fix. Each cell reads `doom_shroom` (v3, from the Doom Shroom section) →
`rockwall_brawl`:

| clip | agreement median (5th percentile) | corrections a minute | crate spread | label check |
|---|---|---|---|---|
| Dark Passage bottom-left | 0.72 (0.64) → 0.85 (0.69) | 23.9 → 4.4 | 0.226 → 0.196 | 0.918 → 0.942 |
| Dark Passage bottom-right | 0.79 (0.60) → 0.82 (0.68) | 14.6 → 9.7 | 0.210 → 0.202 | 0.955 → 0.964 |
| Dark Passage top-right | 0.74 (0.62) → 0.74 (0.70) | 4.9 → 4.9 | no crates | 0.949 → 0.961 |
| Starfruit | 0.84 (0.70) → 0.87 (0.63) | 5.2 → 4.1 | **0.268 → 0.270** | 0.957 → 0.953 |
| Truth | 0.88 (0.74) → 0.91 (0.75) | 3.7 → 4.3 | 0.178 → 0.196 | 0.970 → 0.979 |
| Doom Shroom south-centre | 0.91 (0.76) → 0.92 (0.85) | 2.9 → 1.0 | 0.207 → 0.133 | 0.971 → 0.978 |
| Doom Shroom north-centre | 0.87 (0.80) → 0.92 (0.86) | 1.6 → 3.3 | **0.257 → 0.331** | 0.969 → 0.980 |

- **Every clip holds its fix from the gate to the end,** except top-right's 0.1 s search at the
  gate, which every model makes. Truth's one whole-tile correction, at the second match's start, is
  the one every model makes. Bottom-right keeps the fix that v3 dropped for 0.4 s.
- **Dark Passage reads better than with v3** on all three clips. No session runs `rockwall_brawl`
  there: Dark Passage sessions run `--terrain dark_passage`.
- **Two crate spreads miss the gate with every model replayed on them:**
  - Starfruit's (default 0.277).
  - Doom Shroom north-centre's, where `rockwall_brawl` gives v2's 0.331. Doom Shroom sessions run
    `--terrain doom_shroom`.

**Under its shipped name** (`runs/map_checks/rockwall_brawl/shipped/`), the replay with `--terrain
rockwall_brawl` matches the candidate's by-path replay in all 1554 decision rows and in the seen
map, apart from the two timing columns (`grab_ms`, `tick_ms`).

**Selecting it.** A Rockwall Brawl session needs both the map and the classifier:

| to play Rockwall Brawl with | do this |
|---|---|
| a live session (Bull) | `./.venv/Scripts/python.exe scripts/deploy_run.py --map rockwall_brawl --run runs/mortis_ppo-20261103-092146 --terrain rockwall_brawl` |
| a replay | `./.venv/Scripts/python.exe scripts/probes/replay_clip.py <clip> --map rockwall_brawl --terrain rockwall_brawl --run runs/mortis_ppo-20261103-092146` |

Run them from the repo root, with the venv's interpreter (Doom Shroom, **Selecting it**, says why).

Without `--terrain rockwall_brawl`, the session runs on the default, which never fixes here. Five
seconds after the gate the localizer calls the map wrong and keeps that verdict for the match, so
the policy never gets the map label. `deploy_run.py` logs `terrain rockwall_brawl
(terrain_rockwall_brawl.pt)` at startup.

**Open items.**

- **The user's:** M4 on held-out matches, M5 live, and which classifier is the default. M1 is
  done: the user signed off `rockwall_brawl.csv` on 11-11 with no changes, so every label above
  was projected from the signed-off file.
- **No held-out frame scores a cone** (M3b, above). Over the replay, the one cone seen in the open
  reads as wall, and the edge glimpses read as floor. While the fix holds, the policy gets every
  cone from the map label, as wall.
- **What the clip never sees after the gate:** 855 tiles, mostly the bush border (above). While
  the fix holds, the policy gets the map label there.
- **Broken ground keeps its class for the policy.** The map label is the ground as the match
  starts, and the sim has no destructible walls. So a cone or cube broken mid-match still blocks in
  the policy's planes and the move mask for the rest of the match.
