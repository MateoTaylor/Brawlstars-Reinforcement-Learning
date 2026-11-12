# Emulator rectification: the scale error and how to correct it

Status, 2026-11-04: **a plan for you to decide on. Nothing is built.** No code, calibration or label
file was changed. Every number below was measured read-only, on CPU, from tracked data and existing
clips. The scripts are listed at the end of section 9.

None of section 1's numbers go through the terrain classifier. They come from the corner CSVs, from
phase correlation on rectified frames, and from the projection tool, which never loads the
classifier. So the 11-04 swap of `terrain.pt` doesn't touch them. The baselines that do depend on
the classifier name its weights by sha256 (sections 6.2-6.4).

## 0. In short

- **Most of the "2.2% too wide" is the shared camera model, not the emulator.** The shipped model
  (`brawl_vision/data/homography.json`) leans. Along any screen row, its pixels per tile fall about
  8% from the left edge of the viewport to the right (section 1.3). The game camera only pitches,
  so the true value is constant along a row. All 13 corners the model was fit to (set A) sit in the
  left half of the frame, so nothing held the lean down.
- **Phone and emulator footage show the same error.** A world translation should move rectified
  ground rigidly. With the shipped plans, the x-shift grows about 10% across the patch width, and
  the y-shift shrinks about 5% down its height. This holds on 6 phone clips and 10 emulator clips
  alike (section 1.4).
- **The corners to fix it are already on disk.** The held-out calibration set B covers the right
  half, and is off the model by up to 0.12 tile. A refit on A+B (23 corners, 2.5 px mean residual)
  removes the lean, and predicts the footage gradients to within about 15% (sections 1.3 and 1.4).
- **About 1-2% of uniform x scale on the emulator is still unexplained.** At the hero, the refit
  and the shipped model agree to 0.2% in x, yet Starfruit measures 2.2%. The y scale is fully
  explained. Whether the x remainder is real can only be measured once the lean is gone
  (section 6.1).
- **A per-HUD scale cannot fix a lean.** It would fix odometry near the patch centre. It would
  leave up to about 0.3 tile of misplacement at the patch's left and right edges, on both devices.
- **Recommendation (section 7):** a separate emulator camera model, refit from sets A+B and picked
  per HUD.
  - Add a geometry fingerprint to the label check first.
  - Keep the drawn labels, tagged, and regenerate the 17 projected Starfruit labels.
  - Gate on a closure test.
  - Then verify on the Dark Passage tuning replays and on Starfruit, against today's numbers.

  The phone model, the sim's camera constants and their tests stay as they are.
- **Decisions D1-D7 are in section 8.**

## 1. Measurements

### 1.1 The starting point: the emulator against the map (11-04)

This was measured on `bluestacks-starfruit-supernova-west-upper` (2560x1440) while projecting the
Starfruit labels (see MAP_ONBOARDING.md, Starfruit section). S is rectified tiles per true tile.

| measurement | x | y |
|---|---|---|
| map-image NCC position against odometry, 143 frames: slope (residual sd) | 0.9784 (0.157 tile) | 0.9962 (0.117 tile) |
| S, as `scripts/map_project_labels.py` fits it | 1.022 | 1.004 |
| best uniform scale from a global NCC search (flat range) | 1.02 (1.02-1.025) | 1.015 (1.01-1.025) |
| `bluestacks-dark-passage-top-right` (1080p), `scripts/probes/map_label_projection/scale_check.py` | 1.01-1.03 | |

A full 2x2 fit gives map = A * odometry + c, with A = [[0.9799, 0.0107], [-0.0060, 0.9954]]. Its
inverse, [[1.0204, -0.0110], [0.0062, 1.0046]], is the scale that odometry sees.

Even at the fitted S, the projection tool's 3x3 block residuals remain. Each block is `MapMatcher.at_scale`
matching one ninth of the frame on its own; values are (dx, dy) in tiles, the median over 180
frames. A uniform error can't make this pattern. It is the part of the error that varies across
the view.

| | left | middle | right |
|---|---|---|---|
| top | HUD | (+0.03, +0.11) | (-0.13, +0.14) |
| middle | (-0.10, -0.02) | (+0.06, -0.04) | (-0.02, -0.07) |
| bottom | HUD | (+0.10, -0.04) | HUD |

### 1.2 What it costs today

Every Dark Passage and Starfruit gate passes (section 6.2). The cost is margin.

- **Odometry over-reads x by about 2% on the emulator.** On the Starfruit replay, the two fixes
  differ by -0.24 tile in x after 14.7 tiles east, where 2.2% predicts -0.32.
- **The localizer's view sits off the map at its edges.** The localizer places the view by
  translation only. A view stretched 2% sits 0.3 tile off the map at its edges, which broadens the
  agreement peak.
  - K4's check-3 tent reads only 0.3-0.85 of a real error.
  - Agreement medians sit at 0.73-0.82.
  - Crate spread sits at 0.18-0.24, against a 0.25 gate.
- **The deploy code has already seen the lean.** `brawl_deployment/perception/lattice.py:38-39` says
  readings lean with screen position, "by up to +0.16 tile x at the right edge (a camera-model
  term)".
- **Everything placed away from the hero carries the lean, on both devices.** That covers occupancy
  cells, loot, gas fronts and projected entities, off by up to about 0.3 tile at the patch's left
  and right edges (section 1.4). The projected labels correct the uniform part, because the tool
  fits S, but not the lean.

### 1.3 The shipped model leans; set B shows it; A+B removes it

**The lean.** These are local pixels per tile along tile x, from the shipped ground H, on the
2002-px-wide viewport:

| screen y | x = 100 | x = 700 | x = 1001 | x = 1300 | x = 1900 | left to right |
|---|---|---|---|---|---|---|
| 200 | 75.4 | 73.4 | 72.3 | 71.3 | 69.4 | -8.4% |
| 376 | 78.1 | 76.0 | 74.9 | 73.9 | 71.9 | -8.2% |
| 563 | 80.9 | 78.8 | 77.7 | 76.7 | 74.6 | -8.1% |
| 800 | 84.5 | 82.3 | 81.2 | 80.2 | 78.0 | -7.9% |
| 1000 | 87.5 | 85.3 | 84.2 | 83.1 | 80.9 | -7.8% |

Rows of tiles are horizontal on screen in every calibration frame, so the camera pitches and does
nothing else. Such a camera maps tile x with no foreshortening, which means H[2,0] = 0. The shipped
H has H[2,0] = 0.00175, with perspective row [0.00175, -0.01185]. That is a yaw the game camera
doesn't have, and `H_top` leans the same way.

The fit allowed this because set A (`brawl_vision/data/frames/calib_points_a.csv`) spans only
screen x 393-904. `frames/ANNOTATION.md` warns that clustered points "give a fit that is accurate
there and drifts badly elsewhere".

**Set B.** `calib_points_b.csv` holds 10 corners at screen x 825-1659. It is the repeat annotation
that ANNOTATION.md uses as Phase C's acceptance check. The table gives model minus clicked, in
tiles, after the best origin offset (B's origin is arbitrary):

| corner (px) | x | y |
|---|---|---|
| (1139, 378) | -0.002 | -0.000 |
| (1139, 436) | -0.029 | -0.039 |
| (1139, 504) | -0.060 | +0.060 |
| (1221, 500) | +0.002 | +0.000 |
| (1523, 376) | +0.122 | -0.034 |
| (1535, 497) | +0.104 | -0.034 |
| (1574, 904) | +0.030 | +0.058 |
| (1659, 906) | +0.065 | +0.100 |
| (831, 380) | -0.045 | +0.033 |
| (825, 436) | -0.104 | -0.046 |

The x residual grows with tile x at +0.0174 per tile, so the model reads B's x distances 1.74%
long (y: 1.18%). Set A's residuals show no slope, and all stay within 0.06 tile.
`test_annotated_corners_land_on_the_tile_lattice` holds B to 9.0/16.0 px, against A's 2.5/5.0,
which is why this never failed.

**The refit.** H_top is fit on A+B together: 23 corners, with B's origin as a free translation. The
ground H is rebuilt with the shipped vertical vanishing point and mu (1.0152).

| | shipped | A+B refit |
|---|---|---|
| H_top perspective row | [+0.001755, -0.011902] | [-0.000136, -0.010646] |
| px per tile along a row, left to right | -8% | +0.8% |
| residual mean (px) | 2.01 on A (its own fit) | 2.48 on A, 2.44 on B |
| residual max (px) | 4.28 on A | 6.89 |
| ground H | [[78.3015, -11.3722, 524.5876], [0.65841, 55.72367, 429.53441], [0.00175, -0.01185, 1]] | [[75.9427, -10.2218, 525.3261], [-0.45012, 56.28195, 433.95713], [-0.00014, -0.01059, 1]] |

The refit's residuals sit inside the click precision ANNOTATION.md gives (±3-5 px). A real
calibration run would also re-solve mu from `calib_height.csv`; this refit reused the shipped
value.

### 1.4 The shift field: one camera, one error, both devices

**Method** (`shift_field.py`, section 9):

- Each clip is rectified with the shipped plan for its HUD, and frames are paired 0.15 s apart.
- 192-px windows on a 96-px step inside the eroded valid patch are each phase-correlated on their
  own. A window counts if its response is above 0.3.
- A pair is kept if its median shift is 8-60 px. Within a pair, windows within 3 px of the median
  are kept.
- Per pair, each window's deviation from the median is fit linearly over the patch. The fits are
  then pooled across pairs, against the median shift.

With u and v as the rectified x and y, each a fraction of the patch, the four terms are:

- a1: the relative change of the x scale across the width;
- a2: the change of the x scale down the height;
- b1: the change of the y scale across the width;
- b2: the change of the y scale down the height.

A correct model reads 0 on all four. So does a uniform scale error, because a stretched patch
still translates rigidly. This test measures the lean, not the 2.2%.

| clip | device | pairs | a1 | a2 | b1 | b2 |
|---|---|---|---|---|---|---|
| day10_gameplay | phone | 769 | +0.100 ±0.002 | -0.024 ±0.002 | +0.050 ±0.002 | -0.051 ±0.003 |
| showdown_alternate_map | phone | 740 | +0.101 ±0.003 | -0.026 ±0.002 | +0.050 ±0.003 | -0.045 ±0.001 |
| 9_5_brawlstars_eval | phone | 612 | +0.103 ±0.006 | -0.023 ±0.003 | +0.046 ±0.003 | -0.052 ±0.005 |
| showdown_alternate_map2 | phone | 299 | +0.102 ±0.006 | -0.009 ±0.008 | +0.049 ±0.008 | -0.035 ±0.007 |
| day12_recording1 | phone | 223 | +0.098 * | -0.027 * | +0.057 ±0.011 | -0.054 ±0.006 |
| showdown_has_gadget | phone | 115 | +0.109 * | -0.026 * | +0.045 ±0.006 | -0.042 ±0.005 |
| 2maps_dataset | emulator | 601 | +0.103 ±0.005 | -0.059 ±0.010 | +0.051 ±0.003 | -0.046 ±0.002 |
| 9-10_new3 | emulator | 540 | +0.087 ±0.006 | -0.031 ±0.009 | +0.039 ±0.002 | -0.048 ±0.002 |
| bluestacks-starfruit-supernova-west-upper | emulator | 516 | +0.102 ±0.013 | -0.026 ±0.041 | +0.064 ±0.008 | -0.047 ±0.003 |
| bluestacks-dark-passage-bottom-right | emulator | 397 | +0.103 * | -0.024 * | +0.046 ±0.005 | -0.055 ±0.009 |
| 9-10_new1 | emulator | 335 | +0.103 ±0.006 | -0.023 ±0.007 | +0.053 ±0.007 | -0.044 ±0.006 |
| bluestacks-example3 | emulator | 250 | +0.102 ±0.010 | -0.022 ±0.008 | +0.054 ±0.010 | -0.049 ±0.003 |
| bluestacks-example-zone | emulator | 241 | +0.104 ±0.007 | -0.020 ±0.012 | +0.058 ±0.005 | -0.046 ±0.003 |
| bluestacks-dark-passage-top-right | emulator | 238 | +0.102 ±0.009 | -0.031 ±0.011 | +0.040 * | -0.035 * |
| bluestacks-example-new | emulator | 170 | +0.111 * | -0.005 * | +0.051 ±0.009 | -0.044 ±0.006 |
| bluestacks-dark-passage-bottom-left | emulator | 165 | +0.103 * | -0.012 * | +0.052 ±0.007 | -0.052 ±0.010 |
| **the A+B refit predicts** | | | **+0.116** | **-0.024** | **+0.058** | **-0.048** |

Notes on the table:

- ± is the standard error over 8 chunks of pairs.
- \* means that standard error is meaningless (0.15-3.3). Some chunks had almost no motion on that
  axis, which leaves their fits singular. The value shown is the pooled one over all pairs.
- `counted_walking` and `zone_grows_from_east` are left out. They move almost only in y (mean |dx|
  0.1 px), so their x terms are undetermined. Their b2 read -0.080 ±0.012 and -0.058.

Every clip with motion reads about a1 +0.10, a2 -0.02, b1 +0.05 and b2 -0.05, on phone and
emulator alike. The refit predicts each term to within 0.015; the biggest gap is a1, +0.116
against a measured +0.102. So the true camera lies most of the way from the shipped model to the
refit: 85-100%, depending on the term.

**The error in tiles.** Suppose the scale changes by a fraction g across a patch W tiles wide. Then
content at distance d from the centre is misplaced by about `g*d^2/(2W)`, with the same sign on both
sides. With g = 0.10 and W = 30, that is 0.17 tile at d = 10, and 0.28 tile at d = 13.

- A best-fit offset cuts the edge error to about two thirds of that.
- No uniform scale removes it.
- The cross term b1 adds about ±0.15 tile in y for content 6 tiles above or below the hero at
  either edge.

### 1.5 What the refit predicts, and what's left

If the A+B refit were the true camera, the shipped plan would show these values of S:

| where | x | y | measured on Starfruit, x / y |
|---|---|---|---|
| at the hero, screen (1008, 614) | 1.0017 | 1.0043 | 1.0204 / 1.0046 (odometry fit) |
| mean over the valid patch | 1.0087 | 1.0126 | 1.02 / 1.015 (NCC search) |
| 5th-95th percentile over the patch | 0.962-1.056 | 0.981-1.045 | |

The predicted cross terms at the hero are -0.0015 and +0.0008. The measured ones are -0.011 and
+0.006.

- **The y scale is explained:** 1.0043 against 1.0046 at the hero, and 1.013 against 1.015 over the
  patch.
- **The x scale is not:** 1.002 against 1.020 at the hero, and 1.009 against 1.02 over the patch.
  That leaves 1.1-1.8%.

Two things point to an x scale that only the emulator has:

- BULL_HERO_PLAN.md §5.0(b): a BlueStacks wall read 101 px per tile, where the model gives 102 to
  96 across the frame ("takes about 2% off"). That was measured in Training Grounds, so it is weak
  support.
- Dark Passage top-right reads 1.01-1.03.

Two things point against it:

- **Both pipelines keep square pixels.** `brawl_vision/capture.py::normalize_viewport` trims the
  phone's 2436-px width to 16:9 without resizing, and the emulator's 2560x1440 is resized
  uniformly. I can find no mechanism for an x-only difference between the devices.
- **Every emulator number above was measured through the leaning plan.** One 2x2 summarizes that
  poorly.

The closure test in 6.1 measures the remainder directly, once the lean is gone.

**The view under the refit.**

- **The footprint moves.** The viewport's tile footprint, relative to the hero anchor, moves 0.21 /
  0.79 / 0.22 / 0.39 tile at its TL / TR / BR / BL corners.
- **The top edge becomes symmetric about the screen centre.** Left extent minus right extent there
  reads +0.17 tile, where the anchor's +0.09 offset predicts +0.18; the shipped model gives -0.66.
  The bottom edge reads +0.32, against the shipped -0.18.
- **The window shrinks by a column.** The outward-snapped window becomes (-8,-8) 29x19 (today
  30x19), spanning tile x -7.96 to 20.46 and y -7.77 to 10.22.

### 1.6 Never measured

- **No emulator frame has ever been annotated.** Every calibration corner is on the phone frame
  (`calib_showdown_f060`).
- **The x remainder is unresolved.** It could be a real 1.1-1.8%, or an artifact of measuring
  through the lean.
- **The terrain classifier has never been scored under a new geometry.**
- **mu may move.** Nothing has checked whether it changes when H_top is refit.

## 2. Where the geometry lives

| piece | where | notes |
|---|---|---|
| calibration corners | `brawl_vision/data/frames/`: `calib_points_a.csv` (13, the fit), `calib_points_b.csv` (10, held out), `calib_height.csv` (6 wall-height pairs, pair 3 excluded) | phone frame; the PNGs are gitignored; the procedure is in `frames/ANNOTATION.md` |
| solver | `brawl_vision/camera.py`: `solve_camera_model`, `save_camera_model`, `load_camera_model` | `cv2.findHomography` on one corner CSV gives H_top. The height pairs give the vertical vanishing point. A planar homology with ratio mu then moves H_top to the ground. |
| model file | `brawl_vision/data/homography.json` (`config.HOMOGRAPHY_PATH`) | H, H_top, the 2002x1126 viewport, the fit report |
| rectify plan | `camera.py`: `RectifyPlan`, `build_rectify_plan(model, mask)` | M = window scale * H_inv, at 48 px per tile. The window is the viewport's tile corners snapped outward: today (-8,-8) 30x19, which is 1440x912 px. |
| per-HUD plans | `brawl_vision/terrain/labeling.py:184` `hud_plans` | one model, with one plan per mask in `config.HUD_MASKS` (`hud_mask.json`, `hud_mask_emulator.json`). The plans differ only in `valid`. |
| live plan | `brawl_deployment/loop.py`, in `VisionStack.build` | `build_rectify_plan(load_camera_model(), load_hud_mask(HUD_MASK_PATH))`, with the module-level `HUD_MASK_PATH = EMULATOR_HUD_MASK_PATH`. `scripts/probes/replay_clip.py` goes through the same build. |
| sim camera | `brawl_sim/config.py:100` `camera_quad`, `:103` `camera_clamp_onset` (12.1, 12.8, 8.9, 5.5); `brawl_vision/camera.py:118` `HERO_ANCHOR_TILES` (0.09, 0.80) | Measured off the shipped model. Pinned by `tests/test_sim_camera.py::test_quad_matches_the_shipped_homography` (to 0.05 tile) and by `tests/test_vision_camera.py::test_the_sim_view_fits_inside_what_the_camera_can_actually_see`. |

## 3. Consumers, and what a new emulator model moves

| consumer | what it takes from the geometry | under B2 (section 4) |
|---|---|---|
| odometry, `brawl_vision/terrain/odometry.py` | the median window shift in the rectified patch, divided by 48 | its scale near the patch centre; the 2% only if 6.1 shows it is real |
| terrain classifier and occupancy map, `terrain/classifier.py`, `terrain/occupancy.py` | rectified cells | cell registration, by up to about 0.3 tile at the edges; the classifier's input, slightly (D6) |
| known-map localizer and lattice, `brawl_deployment/perception/localize.py`, `lattice.py` | the occupancy grid, odometry, `camera_clamp_onset` | agreement, corrections and the lattice's +0.16 lean. The crate offset (-0.10, +0.41) was measured through today's geometry and should be re-measured. |
| loot, gas and entity projection: `perception/loot.py`, `perception/zone.py`, `terrain/zone.py`, `object_detection/project.py` | rectified positions, or the model's px-to-tile | positions away from the hero |
| `LabelGrid.check_plan`, `labeling.py:150` | origin, size and px per tile, nothing else | **nothing it can see**, if the window is kept (section 5) |
| emulator labels: 67 in `tests/fixtures/vision/labels/`, 7 in `labels_starfruit_test/` | drawn or projected on today's emulator plan | they point at world up to a few tenths of a tile away, at the edges |
| tools that go through `hud_plans()`: `vision_label.py`, `vision_train_terrain.py`, `vision_score_map.py`, `map_project_labels.py`, the `map_label_projection` probes | the plan for the label's HUD | they follow `hud_plans()` automatically |
| tools that call `load_camera_model()` directly: `vision_evaluate.py`, `vision_sample_palette.py`, `vision_truth.py`, `vision_watch.py`, `probes/zone_probe.py` | the one model | each must choose by HUD. A missed one silently rectifies emulator footage with the phone model. |
| `brawl_deployment/capture.py:81` | `viewport` only | nothing; both models share the 2002x1126 viewport |
| the sim: `camera_quad`, `camera_clamp_onset`, `HERO_ANCHOR_TILES`, `PLAYER_BOX_FROM_RING_TILES` | constants measured off the phone model | nothing under B2 (D7) |
| Bull's stats | measured in BlueStacks from wall px per tile, not through the model (BULL_HERO_PLAN.md §5.0) | nothing |

B1 moves more than this:

- the 35 phone labels, and the phone grid `showdown_alternate_map2.grid.*`;
- the sim quad, which fails its test, because its corners move 0.21-0.79 tile;
- `test_shipped_plan_matches_the_recorded_window`, because the window becomes 29x19;
- `test_annotated_corners_land_on_the_tile_lattice`, whose 2.5/5.0 px bounds on A need
  re-checking (the refit's mean on A is 2.48 px).

## 4. Options

### A. A per-HUD scale, or affine, in the rectify plan

`hud_plans()["emulator"]` gets M = window scale * diag(1/1.022, 1/1.004) * H_inv, or the full
measured 2x2.

- **Fixes:** the uniform part. That is odometry near the centre, and the view-to-map scale.
- **Leaves:** the lean, on both devices. The block residuals of 1.1 stay, and so does up to about
  0.3 tile at the edges.
- **Breaks:** two things.
  - It moves M without moving the window. The 74 emulator labels then re-point, and `check_plan`
    can't see it, so the fingerprint in section 5 has to come first.
  - It bakes in a number that was measured through the lean. A later model fix would then
    double-correct.
- **Verdict:** not on its own. As a last step on top of B, it is the right shape for a real x
  remainder (D4).

### B. Recalibrate

**B1: refit the shared model.** One `homography.json` serves both devices. It moves everything in
section 3, plus the phone labels, the sim quad and the window (see the B1 list there). Changing the
sim is a training-side change (D7).

**B2: a separate emulator model.**

- **What it is:** `homography_emulator.json`, next to today's file. One per-HUD lookup serves
  `hud_plans`, `VisionStack.build` and the direct callers.
- **What stays:** the phone model, the phone labels, the sim and their tests.
- **Why it's enough:** live play is emulator-only, so all the live benefit is here.
- **How it's picked:** by name, the way `terrain_model` picks the classifier. A replay can then run
  either model, and live can roll back with a config change.

Calibration sources, for either:

- **(i) Refit on A+B.** It needs no new clicks, has a 2.5 px mean residual, and predicts the
  footage gradients to within about 15%. But it is still a phone frame, so it can't contain an
  emulator-only x scale.
- **(ii) Annotate an emulator frame.** Follow ANNOTATION.md across the whole width of a BlueStacks
  match frame, not Training Grounds. These would be your clicks. It measures any device difference
  directly.
- **(iii) Fit from footage.** Solve for the correction that zeroes the shift field (the lean) and
  makes the Starfruit S equal 1 (the scale). It uses thousands of frames and no clicks. But it is
  new code, and the clips it fits on can't also be its check.

### C. Correct odometry only

Divide emulator odometry deltas by (1.022, 1.004).

- **Fixes:** drift between fixes.
- **Leaves:** the localizer's stretched view, every position away from the hero, the lean and the
  labels. A later model fix would apply the same correction a second time.
- **Verdict:** rejected.

### D. Let the localizer absorb it

Search over scale, or an affine, as well as translation.

- **Fixes:** agreement and the tent, and nothing upstream of them.
- **Costs:** tick time, in a stage that runs every tick (the live median is 75-81 ms).
- **Verdict:** rejected.

### E. Leave it

Every gate passes today. Living with it means:

- the margins of 1.2;
- the lean in every new map's projected labels;
- edge cells up to about 0.3 tile off, on both devices.

Capture-side changes (resolution, resize) are not an option. BRAWL_DEPLOYMENT_DESIGN.md §9 item 4
settled 1440p resized to 2002x1126, and no resize can remove a perspective error.

## 5. Labels

All stored labels were drawn at window (-8,-8), 30x19 tiles, 48 px per tile:

| folder | hud | files | labelled cells | clips |
|---|---|---|---|---|
| `tests/fixtures/vision/labels/` | emulator | 67: 57 drawn, 10 projected (Starfruit training) | 16,405 | 12 |
| `tests/fixtures/vision/labels/` | phone | 35 | 10,608 | 18 |
| `tests/fixtures/vision/labels_starfruit_test/` | emulator | 7, projected (Starfruit held out) | 1,346 | 1 |

**The trap.** `check_plan` compares origin, size and pixels per tile, and nothing else.

- **Taking the refit's own window is the safe case.** That window is 29x19, so a plain swap makes
  every emulator label fail loudly.
- **Keeping 30x19 is the dangerous case.** Section 7 recommends keeping it, so that array shapes
  and cell indices don't change. But then the check passes, while every emulator label points at
  world a few tenths of a tile from where it was drawn. The check's docstring warns about exactly
  this.

**Steps and options:**

1. **Add a fingerprint first, whatever else is chosen.**
   - Each label gains a `geometry` field: a short hash of the camera model it was drawn under.
   - A file without the field is legacy, meaning today's `homography.json`.
   - `check_plan` compares the field too.
   - Under B2, today's file stays on disk as the phone model, so no frozen copy is needed. Under
     B1, it is kept under a new name.
   - No existing label file is rewritten.
2. **Tag and keep the drawn labels.** The 57 drawn emulator labels stay byte-identical. The trainer
   and scorers rectify each label's frame with the plan for its (hud, geometry):
   - legacy emulator labels keep training on today's emulator plan;
   - new labels use the new one.

   The classifier then trains on two geometries, which differ by a few percent of local scale.
3. **Or resample the drawn labels.** Map each new cell's centre through the old and new plans into
   the old grid, and take that cell's label, or '?' where the cell straddles labels that differ.
   That gives one geometry everywhere, but it rewrites hand-made files and loses some edge cells.
4. **Regenerate the projected labels.** The 17 Starfruit labels are outputs of
   `scripts/map_project_labels.py`. The rebuild command is in
   `scripts/probes/map_label_projection/README.md`, and it writes to `runs/` first.
   - The tool refits S itself, so under the new plan it should find S ≈ (1.00, 1.00). That reading
     is itself part of 6.1.
   - The held-out frames (1840-2200) stay the same, so that set stays held out.

Recommended: step 1, then step 2 for the drawn labels and step 4 for the projected ones (D5).

## 6. Verification

6.1 gates everything after it. Everything runs on CPU (`CUDA_VISIBLE_DEVICES=-1`), and only on
footage that may be used:

- the three Dark Passage tuning clips (`bluestacks-dark-passage-{bottom-left,bottom-right,top-right}`);
- Starfruit `west-upper`;
- the BlueStacks clips in the 1.4 table.

The Dark Passage held-out pair and the K5 matches are spent, and stay out.

### 6.1 Closure: the new emulator model on its own

| check | today | gate |
|---|---|---|
| shift field on every emulator clip in 1.4 (`shift_field.py` on the new plan) | a1 +0.09 to +0.11, a2 -0.02 to -0.06, b1 +0.04 to +0.06, b2 -0.04 to -0.06 | each term within ±0.02 of 0 |
| Starfruit S, projection tool stages 1-2 | (1.022, 1.004) | within 0.01 of (1, 1) |
| Starfruit odometry-to-map cross terms | +0.011, -0.006 | under 0.005 |
| Starfruit 3x3 block residuals at S (1.1) | up to 0.14 tile | every block within 0.05 tile |
| Dark Passage top-right, best uniform scale (`scale_check.py`) | 1.01-1.03 | 1.00 ± 0.01 |
| calibration corners, sets A and B | B off by up to 0.12 tile | mean ≤ 3 px and max ≤ 7 px, on both sets |

What each outcome means:

- **Everything passes:** go on to 6.2.
- **The shift field passes, but S_x stays above 1.01:** the x remainder is real, and only the
  emulator has it. Fold it in with option A on top of B2, or recalibrate from an emulator frame,
  source (ii) (D4).
- **The shift field fails:** A+B's corners aren't enough. Go to source (ii) or (iii) (D3).

### 6.2 Dark Passage tuning replays

For each clip, run `scripts/probes/replay_clip.py --map dark_passage --device cpu --csv <out>`,
then `scripts/probes/map_localize_report.py`.

- **Run each clip twice:** once on today's plan and once on the new one, with the same classifier.
- **Pin the classifier, and record it.** `--terrain dark_passage` reproduces the old column, and
  `--terrain default` (today's `terrain.pt`) reproduces the shipped one. Note the weights' sha256
  next to every number, because a comparison that straddles a classifier swap is confounded.
- **Compare on the perception columns.** `meta.time_frac` reads the wall clock even in a replay
  (`loop.py`'s `elapsed_s`), so a near-tie decision can flip between runs.

| clip | K4: right / a tile off / no fix | 11-04: first fix, share | agreement median/5th/min, old → shipped | sub-tile steps a minute, old → shipped | crate spread, old → shipped | label check, old → shipped, best offset (0,0) |
|---|---|---|---|---|---|---|
| bottom-left | 95 / 5 / 0% | 0.00 s, 100% | 0.74/0.65/0.56 → 0.77/0.64/0.59 | 6.5 → 15.2 | 0.208 → 0.214 | 0.921 → 0.941 |
| bottom-right | 98 / 2 / 0% | 0.00 s, 100% | 0.79/0.65/0.55 → 0.82/0.65/0.58 | 10.9 → 8.5 | 0.186 → 0.182 | 0.950 → 0.968 |
| top-right | 100 / 0 / 0.3% | 0.10 s, 100% | 0.73/0.65/0.53 → 0.78/0.66/0.54 | 6.5 → 4.9 | no crates | 0.932 → 0.960 |

About the columns:

- "old" is `terrain_dark_passage.pt` (sha256 f7a788b2…). Its 11-04 replays reproduced K4's
  agreement exactly.
- "shipped" is `terrain.pt` (sha256 578f14d9…), the Starfruit classifier, copied from
  `runs/terrain_starfruit/candidate.pt`.
- The sources are `archive/KNOWN_MAP_LOCALIZATION_PLAN.md` (K4) and MAP_ONBOARDING.md, with
  telemetry in `runs/map_checks/dark_passage_regression/`.
- K4's "a tile off" was the report's own truth drifting, since odometry drifts up to half a tile in
  15 s. That is the drift this change should shrink.

Expected:

- the fix share stays at 100%;
- the agreement median rises;
- sub-tile corrections a minute fall;
- crate spread falls;
- the crate offset is re-measured;
- the label check stays at (0,0) and scores no worse.

A drop anywhere is a finding to explain before going on. It is not a threshold to retune.

### 6.3 Starfruit

Replay `bluestacks-starfruit-supernova-west-upper` as in MAP_ONBOARDING.md's Starfruit M3 run. That
run needed a scratch super-disc gate wrapper, because Shelly has no hypercharge.

| | today |
|---|---|
| first fix | 0.00 s, spawn (7,22) |
| agreement while fixed, median/5th/min | 0.79/0.74/0.73 |
| corrections | none |
| crate spread | 0.216 |
| label check | 0.946 over 1172 tiles, at (0,0) |
| the two fixes, after 14.7 tiles east | -0.24 tile in x, -0.60 in y |

- **The x difference should shrink toward 0.** Scale never explained the y difference.
- **Agreement should rise, and the label check should hold.**
- **The 27% unfixed from 50.8 s stays.** It is the gate reopening, not geometry.

### 6.4 Classifier

- **Re-score the shipped classifier.** Score `terrain.pt` (sha256 578f14d9…) on the regenerated
  held-out set, rebuilt in `runs/` first. Today it scores blocking F1 0.946 and 5-class F1 0.984.
- **Run the label-free score.** Run `scripts/vision_score_map.py` (the label-free hero-cell blocking
  score) on the tuning clips.
- **Retrain only on a measured drop (D6).** The training set would then be the tagged mix plus the
  10 regenerated labels.

### 6.5 Tests

**Unchanged under B2:**

- `tests/test_vision_camera.py`'s pins on the phone model;
- `test_quad_matches_the_shipped_homography`;
- the sim-view fit tests;
- `tests/test_label_projection.py`, once its f640 fixture is checked against the rebuild.

**New:**

- a window pin for the emulator plan;
- a fingerprint test: a label refuses a geometry it wasn't drawn under, and a legacy file maps to
  today's model;
- a shift-field test on an emulator pair with motion on both axes, scored by gradient. Today's plan
  should fail it (a1 +0.10) and the new one pass.
- a test that every offline entry point picks the emulator model for an emulator clip. A missed
  call site is B2's main risk.

### 6.6 Live

The live test is yours. Watch x drift between fixes, sub-tile corrections a minute, agreement while
fixed, and crate spread.

MAP_ONBOARDING.md §3 asks for a change to a shared threshold to be checked on Dark Passage's
held-out footage too. This is a geometry change, not a threshold change, and that pair is spent.
Whether to spend fresh footage on it is your call.

## 7. Recommendation

1. **Fingerprint the label check, on its own.** `LabelGrid` gains `geometry`, legacy files mean
   today's model, and `check_plan` compares the field. Nothing else changes, and every test still
   passes.
2. **Build B2 from source (i).**
   - **A short calibration script** around `solve_camera_model` and `save_camera_model`. It merges
     sets A and B (B's origin as a fitted translation), re-solves mu from `calib_height.csv`, and
     writes `brawl_vision/data/homography_emulator.json`.
   - **A per-HUD model lookup** next to `config.HUD_MASKS`. It serves `hud_plans`,
     `VisionStack.build` and the five direct callers, and can be selected by name.
   - **The emulator window kept at (-8,-8) 30x19,** through an explicit window argument to
     `build_rectify_plan`. That window still covers the whole viewport.
3. **Run the closure test (6.1).** If S_x stays above 1.01, go to D4.
4. **Migrate the labels.** Tag and keep the 57 drawn emulator labels. Regenerate the 17 projected
   ones into `runs/`, compare them, then replace them.
5. **Verify 6.2-6.4** against the baselines.
6. **Retrain the classifier only on a measured drop.**
7. **Leave the rest alone:** the phone model, the phone labels, the sim's camera constants and
   their tests.

**Why B2 over B1:**

- Live play is emulator-only.
- B2 is one new file and a lookup, and switching the name back undoes it.
- B1 moves the sim, which is a training-side change, and all 109 label files, for a phone pipeline
  that only feeds the classifier.

**Why source (i) first:** it needs no clicks, and 6.1 shows whether it's enough. Nothing here
touches the policy, the obs schema or the sim.

## 8. Decisions for you

- **D1. Fix it at all?** Fix it now (section 7), or leave it (option E) and accept the margins in
  1.2.
- **D2. Which model?** B2 (recommended), B1 (one shared refit), or A alone (scale only; the lean
  stays).
- **D3. Calibration source.** (i) the A+B refit (recommended first), (ii) your clicks on an
  emulator match frame, or (iii) a fit from footage (new code).
- **D4. If an x remainder survives 6.1:** add a per-HUD x scale on top (option A), or recalibrate
  from an emulator frame, source (ii).
- **D5. Drawn labels:** tag and keep (recommended), or resample.
- **D6. Classifier:** retrain only on a measured drop (recommended), or retrain on the new geometry
  regardless.
- **D7. The sim's camera constants.** `camera_quad` and `camera_clamp_onset` were measured off the
  leaning model too. Under the refit:
  - the quad's corners move 0.21-0.79 tile;
  - the 21x13 view's margin grows from +0.022 to +0.216 tile. 23x13 still doesn't fit (-0.534 to
    -0.294), and nor does 21x15 (-0.978 to -0.784), so the obs grid's size stands;
  - the clamp onsets move by up to 0.26 tile;
  - `HERO_ANCHOR_TILES` and `PLAYER_BOX_FROM_RING_TILES` move under 0.01 tile, so they stay.

  Changing the sim is a training-side change, and B2 doesn't need it.

## 9. Found on the way, not acted on

- **The capture decision's evidence is stale.** BRAWL_DEPLOYMENT_DESIGN.md §9 item 4 says the phone
  calibration transfers to BlueStacks with correct perspective and correct absolute scale. Its
  evidence is an odometry inlier ratio of 1.000, but phase correlation can't see a uniform scale
  error, and 1.4 contradicts the perspective claim. The decision itself (1440p, resized) stands;
  only the evidence sentence is stale.
- **The rigidity test can't see the lean.**
  `test_rectified_content_translates_rigidly_across_the_patch` uses `counted_walking` 286 to 298.
  - That pair moves only in y (mean |dx| 0.1 px), so it can't see the x lean.
  - Its scatter bound (std < 1 px) also misses a 5% y gradient at about 20 px of motion, which is
    about 1 px across the whole patch.
  - `test_the_rigidity_check_actually_discriminates` compares against a camera with no
    perspective at all, a far larger error.
- **Set B's tolerance is loose.** `test_annotated_corners_land_on_the_tile_lattice` holds set B to
  9.0/16.0 px, loose enough to pass a 0.12-tile lean.
- **The data README is out of date.** `brawl_vision/data/README.md` names
  `scripts/vision_calibrate.py`, which no longer exists. It also describes an older model schema:
  32 px per tile, 1920x1080.
- **One docstring stops being true under B2.** `load_label_dir`'s docstring says any of
  `hud_plans()` will do, because the plans "differ in `valid` and nowhere else".

**The scripts** are read-only, run as `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe <script>`.
They are kept outside the repo, in
`C:\Users\mateo\AppData\Local\Claude-Data\claude_pzs8sxrjxfjjc\side-session-notes\local_d5b0aebd-2f4f-4a07-bd80-087d3ab84f91\emulator_rectify_scripts\`,
together with the per-clip shift-field JSONs:

- `model_lean.py`: the lean table, the set A and B residuals, and the A+B refit.
- `predict_s.py`: the S the refit predicts (1.5).
- `refit_view.py`: the footprint, view margins and window under the refit.
- `shift_field.py CLIP HUD [start end stride]`: the shift field (1.4).
