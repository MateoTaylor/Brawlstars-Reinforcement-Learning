# Known-map localization (Dark Passage first)

**Status 2026-09-29: K1, K2 and K3 are DONE. The user signed off the Dark Passage label, the
localizer is built and passes its tests on damaged views, and the loop takes a map by name
(`--map dark_passage`, or `map.name` in configs/deployment.yaml). K4 has run: the three tuning
clips pass its gates at the crate gate's final 0.25 tile, and the held-out pair missed two gates.
The user then settled two of K4's three open items: after a match's first fix a commit needs only
the guard's agreement, and the crate gate is 0.25 tile. Both are built. The third, a new held-out
pair, was K5's first two dry runs: one passed every gate, one never landed, and in neither did the
hero move. The user then settled the landing's floor: a commit in the spawn windows needs only
the guard's agreement too, and it is built. A second held-out pair, played moving, then passed
every gate: both landed at the gate on the right spawn, neither lost its fix, and still frames
put the fixes within 0.4 tile. K5 with control has run: wall pushes were rare with and without
the map, six matches cannot separate their outcomes, and one match re-fixed 45 tiles wrong after a
drop. The user then chose the camera box (2026-09-29): a placement may commit, and a fix may stay,
only where the game's camera can be. It is built at the sim's clamp limits plus 5 tiles, not the
2 first proposed, because right fixes put the camera up to 2.9 tiles past those limits. Two new
held-out matches then passed it: the box never fired, with the camera up to 3.0 tiles past, and
nothing committed wrong. One missed the unfixed gate at 6.7%, because the older guard dropped a
right fix in bushes for 1.6 s, the second such drop (section 8, K5). Where a step changed this
plan's first draft, the text says so: "K2:" in sections 6.1, 6.3 and 6.4, "K3:" in section 7,
"K4:" in sections 3, 6.2 to 6.4, 6.6, 7 and 8, and "K5:" in sections 6.3, 6.4, 6.6 and 7. K4
moved the fix's sub-tile part from the crates to the terrain. A label holds no crates: power cube
boxes spawn semi-randomly each match, so a map image shows only the fixed terrain (user,
2026-09-28). Every crate item below was revised to match.**

Scope: a deploy match on a map the operator
names in advance and has hand-labelled. Terrain destruction is assumed away for now; the sim has
none either, so a wall that stays a wall in the label matches training. The attack-cadence
problem is separate and is not touched here.

## 0. The idea

Today every match builds its own map from nothing. Odometry gives a frame whose origin is
arbitrary and restarts at every cut, the crate lattice finds the sub-tile phase when it can, and
the classifier's votes fill an occupancy map that is thrown away at each restart.

With a preset, labelled map the loop instead **localizes inside a map it already has**:

1. It finds itself once, at landing.
2. It carries that fix with odometry.
3. It keeps checking the fix against what the classifier sees.
4. It draws the policy's terrain and position from the label.

That is what the sim does for terrain: it fills the whole 13 x 21 crop from the true map,
HUD-hidden cells included. Crates stay as today, from sightings, since boxes spawn semi-randomly
each match and no label can hold them.

## 1. What this rests on

### The current pipeline, measured

| fact | value | where |
|---|---|---|
| classified view per tick | 30 x 19 tiles = 570 cells with the emulator HUD mask: 380 at least half inside the footprint, 323 at the occupancy map's 0.9 share, which the score uses | `build_rectify_plan`, measured 2026-09-27 and 09-28 |
| policy grid crop | 13 x 21 | `grid.py` |
| HUD hole in the crop | bottom-left cells unknown on 40-77% of ticks; UNKNOWN median 8.8%, worst 17.2% (day12 BlueStacks) | `grid.py` |
| classifier map score, BlueStacks | .851 held out, .927 in-sample | `classifier.py` |
| crate lattice | phase error mean .129 tile, 72% within .15; an estimate exists 1/3/10/20 s into 44/56/85/87% of segments; 16% of ok time never locks | `lattice.py` |
| world-frame resets | about 1.5 per minute; each one drops occupancy, loot, gas and tracks | `lattice.py` |
| sim pads the view past the map edge with | WALL | `brawl_sim/maps/loader.py` |
| `MapFrame` | origin (30, 30), extent (60, 60) | `assemble.py` |
| occupancy and gas grids | 128 x 128, world (0, 0) at index (64, 64) | `occupancy.py`, `grid.py` |
| telemetry ring | 4000 rows, 333 s at 12 Hz: one whole match | `loop.py TELEMETRY_ROWS` |

### Dark Passage, measured from the supplied image

| item | value |
|---|---|
| image | 752 x 760 px, thin blue frame; the top ~8 px band is outside the play area |
| lattice | 12.488 x 12.479 px per tile; tile (0, 0)'s top-left corner at (0.962, 8.2), where pixel i spans [i, i+1); 60 x 60 tiles (K1's fit) |
| lattice check | all 10 spawn rings land on tile centres, fractional parts .44 to .54 |
| spawns, (col, row) | (6,5) (26,5) (43,6) (53,15) (6,24) (57,38) (6,47) (47,53) (34,54) (15,55) |
| candles, (col, row), all walls that break easily; K1's first pass took them for crates | (3,48) (7,2) (8,9) (9,22) (10,47) (13,15) (14,57) (19,24) (24,3) (25,14) (25,41) (27,50) (29,3) (31,14) (32,51) (35,57) (36,42) (39,18) (41,8) (43,36) (46,5) (47,18) (48,56) (49,14) (51,35) (53,17) (53,43) (53,50) (55,35) (57,42) |
| power cube boxes | none in the image: they spawn semi-randomly each match |
| layout | asymmetric; two lakes, upper right and left middle; wide bush bands; many single-tile stumps |
| outer ring | playable in places, bush and floor reach the edge, so the true label is not a legal sim CSV as it stands |

The positions were checked by drawing them back onto the image. K1's first pass reproduces the
spawns exactly, and `tests/test_known_map.py` holds it to that. The same check passed the candles
as crates: it could test where a sprite is, never what it is. The user caught it at sign-off.

## 2. The one decision that keeps this small: the map frame

With a map preset, the deploy world frame becomes **map tiles minus half the map size**:
world (x, y) = (col - 30, row - 30) on a 60 x 60 map, so tile (c, r)'s centre sits at
(c - 29.5, r - 29.5).

Three parts of the current code already assume a frame like this. They become true without
an edit:

1. `MapFrame.pos_norm` computes (world + 30) / 60, which is now col / 60: the sim's own
   `hero.pos_norm`. Its docstring measured a wrong origin as free; now there is no wrong origin.
2. `GasMap` and the zone estimator treat world (0, 0) as the map centre. Now it is.
3. The 128 x 128 occupancy and gas grids are centred on world (0, 0). The map lands on indices
   34 to 93, well inside.

So the new component only has to hand out `world.position_tiles` in this frame. Every consumer
downstream already speaks it. It does the job `LatticePhase` does today, turning raw odometry
into the world frame, behind the same members the loop already calls. The loop swaps one object.

## 3. How it fits the pipeline

Per perception tick, `DeployLoop._perceive` (`loop.py`, search for
`lattice = self.lattice.update(dets, self.vision.plan, odo)`):

```
today                                        with a preset map
-----                                        -----------------
odo -> LatticePhase.update(dets)             odo -> MapLocalizer.update(dets)
world = odo - crate phase                    world = odo + map offset          (map frame)
plan.registered(world); classify            plan.registered(world); classify  (unchanged)
OccupancyMap.update(cells, world)            OccupancyMap.update(cells, world) (unchanged, map-aligned)
                                             MapLocalizer.observe(cells, ...)  (new: the checks)
GridBuilder reads OccupancyMap.best()        GridBuilder reads KnownTerrain.best()  (the label)
LootMap from sightings                       LootMap from sightings, kept across confirmed cuts
```

| component | today | with a preset map | change |
|---|---|---|---|
| `LatticePhase` | makes the world frame | kept inside `MapLocalizer`, used while there is no fix | none |
| `MapLocalizer` | | makes the world frame | new |
| `KnownTerrain` | | the label painted into the world grid, WALL off the map | new |
| `OccupancyMap` | the policy's terrain | fallback terrain, and an after-match check of the label | none |
| `GridBuilder` | reads the occupancy map | reads `KnownTerrain` | constructor argument |
| `LootMap` | crates from sightings; a cut wipes them | crates from sightings; a confirmed cut keeps them | none |
| gas map, zone estimator | map centre assumed | map centre true | none |
| `MapFrame` | origin assumed | origin true | none |
| dead-bin move mask | reads the grid | reads the label-backed grid | none |
| trackers, decision history | world frame | map frame; kept across odometry cuts | none |
| telemetry | `lattice`, `phase_x`, `phase_y` | plus six `map_*` columns (section 7) | new columns |

**What the policy gains:**

- **Terrain from the true map.** The crop is filled the way the sim fills it. There is no HUD
  hole, no classifier error and no map lost at a cut.
- **Crates kept across cuts.** A crate sighted once stays in the box plane through a confirmed
  odometry cut, which wipes it today. A crate never sighted stays unknown. The sim shows every
  live crate in the crop, and that gap remains, since no label can hold semi-random boxes.
- **A true `pos_norm`.**
- **A dead-bin mask that works from real walls,** not from classifier mistakes.
- **A sub-tile phase that survives cuts.** Today the lattice relearns its phase after every cut,
  and 16% of ok time never has one. **K4:** the fix reads it from the terrain every second (check
  3 in 6.4), crates or not, and a confirmed cut carries it. Crates cannot give it: Dark Passage's
  sit 0.4 tile off the terrain's tile centres.

## 4. The label

Files in a new `brawl_deployment/data/maps/`:

- **`dark_passage.csv`:** the sim's map CSV format, 60 comma-separated rows of 60 cells, holding
  the TRUE game layout with no forced wall ring.
  - The legend is the sim's without `X`: `.` floor, `#` wall, `b` bush, `~` water, `S` spawn.
    Power cube boxes spawn semi-randomly each match, so the image shows none; the loader refuses
    an `X`.
  - `?` marks a cell not labelled yet. The tool saves it, so unfinished work is never lost; the
    loader refuses it.
  - Plus `f` for a fence that stops a body and passes a shot. The deploy grid already renders a
    fence that way (`_tile_lut`).
  - Dark Passage's stumps, blocks, tombstones, spiked fences and candles are all `#`, the
    transcription rule the screenshot maps used. Candles break easily, but they are walls.
- **`dark_passage.png`:** the source image.
- **`dark_passage.json`:** the lattice pitch and origin per axis in pixels, their convention, the
  image size and the date. Only the tool reads it, so a reopened label sits where it was drawn.
- **Coordinates as in the sim:** column = x to the right, row = y down, tile centre at +0.5.
- **Checks on load:**
  - 60 x 60, no `?`, legend characters only, at least one spawn.
  - The passable region is connected, using `validate_map`'s connectivity check. Its border
    and count rules bind sim maps only.

A sim copy for training on this map is a separate, later choice (L4 in section 9). It needs the
wall ring added, edge spawns moved in and crate spots chosen, since the image shows none.

## 5. The labeling tool

`scripts/map_label.py` is built from `scripts/vision_label.py`. It keeps the same matplotlib
window, colours (`TILE_COLORS`), class keys, drag-to-paint, cluster key, undo and save, but it
opens the top-down map image instead of a rectified frame.

```
.venv/Scripts/python.exe scripts/map_label.py dark_passage --image <path to the png>
```

1. **Grid fit.** It finds the tile pitch and origin from the floor checkerboard and prints them.
   - `--pitch` and `--origin` override the fit.
   - `--corners X0 Y0 X1 Y1` takes the map's two opposite corners as typed pixel numbers, for a
     map with a plain floor.
   - On Dark Passage the fit gives 12.488 x 12.479 px and origin (0.962, 8.2), with clarity 2.78.
     Clarity is the winning pitch's score over the best other pitch; under 1.5 the tool warns.
   - The first save writes the lattice to the JSON, and a reopened label uses it without refitting.
2. **First pass, when no CSV exists yet.**
   - Spawns by matching the ring template; this found all 10.
   - No crates. A crate finder once stamped 30 blue sprites; they were candles, so it is gone.
   - Terrain is proposed, never labelled: 12 colour clusters, left for the user to name. They are
     taken over each cell's upper-middle band, rows .18-.44 and columns .2-.8.
   - That band is where a cell's own content shows:
     - a raised sprite stands on its cell's bottom edge and rises about half a tile into the cell
       above;
     - the next row's sprite covers the bottom 40-50% of a cell;
     - a shadow darkens the top ~.2 of the cell below.
   - So a raised sprite is labelled at its base, the cell it stands on.
3. **Keys.**
   - Classes: `1` floor, `2` wall, `3` bush, `4` water, `5` fence, `6` spawn, `0` clear.
   - Click or drag paints the held class.
   - `c` gives the held class to the whole cluster under the cursor, except its spawns.
   - `g` cycles the overlay: label, clusters, bare image.
   - `u` undoes the last stroke or the last whole cluster fill.
   - `s` saves, `q` saves and quits.
   - The toolbar zooms and pans; clicks paint only while no toolbar tool is active.
   - `--review` opens the label read-only.
4. **Save** runs the section 4 checks. It prints the verdict, the count for each class and the
   spawn list. A label that fails is still written.

The user signed off Dark Passage on 2026-09-28. The 12 clusters were named as:

| named | clusters | what they hold |
|---|---|---|
| floor | 3 | the two checker shades, and cells the spawn rings cross |
| bush | 1 | bush |
| water | 2 | the lakes, and their edge cells |
| wall | 6 | stumps, tombstones, pink caps and candles among them |

8x crops of seven regions against the image found no wrong cell. At sign-off the user turned the
30 candles that the first pass had stamped as crates into walls.

## 6. The localizer

### 6.1 One score

Everything below uses a single score for "the view says the offset is d".

- **Per cell.** Each classified cell in the view is carried into map tiles with d and compared
  with the label cell it lands on: **+1 if the classes agree, -1 if they differ, 0 if the cell is
  not scored**.
- **Not scored:**
  - cells outside the footprint, under the HUD or gassed;
  - cells under a crate or brawler box; these are the masks the occupancy map already gets;
  - cells with classifier confidence under 0.5;
  - cells off the map;
  - **K2:** cells where the view and the label are both floor. Floor is two thirds of Dark
    Passage, so it says almost nothing about where a view is. Counted, an all-floor view agreed
    0.72-0.86 inside every spawn window and reached the commit margin within 5 ticks at 6 of the
    10 spawns. A view's floor on a label wall still counts -1.
- **Counted as agreeing:** wall and fence count as one class; label spawn counts as floor.
- **Many offsets at once:** one `cv2.matchTemplate` per class over one-hot planes, 0.2 ms for the
  whole 60 x 60 map at every whole-tile offset, and exact once rounded.

+1/-1 is close to the log evidence ratio for a classifier that is right about 85% of the time, so
totals add across ticks. Masked and off-map cells count zero, so an offset near the map edge is
not penalised for what it cannot see.

**A mirrored or rotated map is not a trap.** The camera never rotates or mirrors. The view from a
spawn and the view from its mirror spawn are therefore mirror images of each other, and the
score tells them apart. Real ambiguity comes only from locally repetitive terrain such as open
floor. The spawn prior and accumulating over ticks handle that.

### 6.2 Two states

| state | world frame handed out | terrain the policy sees |
|---|---|---|
| no fix | the lattice frame, as today | the CV map, as today |
| fixed | odometry + offset: the map frame | the label |

A match starts with no fix, and a match that never gets one plays exactly like today's. The
localizer never hands out a status the odometry did not produce, so an unsure localizer falls
back to today's pipeline instead of freezing the consumers.

A change of offset is always a **new epoch**, the signal every consumer already treats as "the
frame moved, drop your state". That is how a lattice rebase works today.

**K4:** except check 3's sub-tile steps (section 6.4). They come about 10 times a minute, and an
epoch drops every track, the loot, the gas map and the history. A step under half a tile is what
odometry's own drift already does to every consumer between epochs, and they are built for it.

### 6.3 Getting the fix at landing

- **Anchor.** `_begin_match` resets the localizer on the gate tick. The fix is anchored on the
  match's first ok odometry tick. On the gate tick itself the hero is still on its spawn, because
  the loop's first command comes after `_perceive`. A later first ok tick can be up to a tile off,
  which the search window below covers.
- **Spawn prior.** For each of the 10 spawns, the offset that puts the hero's ring on that spawn's
  centre. The ring's position is the viewport centre through the plan plus `HERO_ANCHOR_TILES`:
  `EntityTracker._nominal_tile` without its box term.
- **K2: the camera clamp.** The game camera stops following the hero near an edge
  (`camera.clamp_onset`), and P1 found nearly every clamp run starting at spawn. Nine of Dark
  Passage's ten spawns sit inside the sim's default band, seven by more than 2 tiles and one by
  10.3. So each spawn's window runs from the offset that puts the ring on the spawn to the one
  that puts it where the sim's camera stops, 2 tiles past either end. The onsets are unmeasured
  geometry defaults; the tests move the game's camera 1-1.5 tiles off them on every side.
- **K5: the camera box.** Here and in every search after a drop or a cut, a placement may lead or
  commit only while the camera's nominal point it implies is within 5 tiles of the sim's clamp
  limits (`CAMERA_TILES`; the user, 2026-09-29). The game's camera goes further than the sim's:
  right fixes put it up to 2.9 tiles past on the north and south edges, 2.5 on the east and 1.9
  on the west. So at spawn it sits between the spawn and the sim's clamped point, which the
  windows above already span (section 8, K5).
- **Candidates.** Every whole-tile offset in the ten windows, 465-504 in all.
- **Accumulate.** Every ok tick adds each candidate's score to its running total. Odometry is
  continuous within a segment, so the unknown offset is constant and the evidence adds up.
- **Commit** when the best total leads every candidate outside its own 3 x 3 neighbourhood by
  `commit_margin`, and its agreement rate is at least `min_agreement`. The provisional values are
  60 cells and 0.65; K4 sets the real ones. **K4:** `min_agreement` holds only until the match's
  first fix. After it, a commit needs only `guard_agreement`, the rate that keeps a fix (6.6).
  **K5:** a commit in these windows needs only `guard_agreement` too, so `min_agreement` now holds
  only in the whole-map search below, before a first fix (the user, 2026-09-28; section 8, K5).
- **On commit:**
  - a new epoch, so every consumer resets once;
  - the terrain source switches to the label.
- **No spawn prior.** If the loop joined mid-match, there is no spawn to anchor on. The same
  accumulator then runs over every offset on the map, with a larger margin, `whole_map_margin`
  (provisionally 120). The sub-tile part comes from the lattice phase, if it has one. **K2:** a
  landing search that has not committed after 5 s widens to the whole map the same way.
  **K4:** after a fix is dropped, the search runs in the lost fix's frame, not the lattice's. That
  frame's sub-tile part is the terrain's, and a whole-map search in a frame half a tile off
  locked 48 tiles wrong on a recording.
- **Wrong map.** If the best agreement is still under 0.55 after 5 s, the loop logs
  `map mismatch: expected dark_passage` and stays unfixed for the match.
  - **K2:** judged once, at 5 s, over the whole map, and only before the match's first fix.
  - **K2:** the whole-map leader counts only when it averaged 20 cells a tick. Views of other
    maps put it on as few as 4 cells over 5 s, at up to 1.0 agreement.
  - Over three seeds of every sim map scored against Dark Passage, 112 of 114 runs got the
    verdict. The other 2 (vine_springs, leader at 0.56) stayed in search. None committed.
    Staying in search is the cheap way to be wrong, since the verdict holds for the match.

### 6.4 Keeping it: checks through the match

The offset stays fixed between corrections, and odometry carries the position. Three checks run
on every ok tick:

1. **Agreement.** The agreement rate at the current offset, smoothed over about 1 s. It is
   logged, and it feeds the guard below.
2. **Neighbours.** The score at the current offset and at its 8 whole-tile neighbours, each
   smoothed over about 1 s. If a neighbour leads by `slip_margin` for 1 s, the offset moves one
   tile that way; the provisional margin is 15. This catches drift that has passed half a tile.
   - **K2:** the first draft's 2 s and 30 cells were too slow. After 240 seeded one-tile odometry
     slips on the K2 test views:

     | smoothing, margin | moved once | dropped by the guard | never moved | time to move, median / worst |
     |---|---|---|---|---|
     | 2 s, 30 cells | 160 | 50 | 30 | 4.3 s / 7.8 s |
     | 1 s, 15 cells | 240 | 0 | 0 | 2.2 s / 3.5 s |

   - With the fix right, no neighbour's smoothed lead rose above -8 cells in 80 walks of 12 s.
     The weakest real slip led by 22.
3. **K4: Terrain sub-tile.** The first draft read the sub-tile part from crates: each full-height
   sighting matched to the nearest tile centre, and the offset moved by the residuals' mean past
   0.15 tile. The recordings ruled that out. Dark Passage's crates read (-0.10, +0.41) tile off
   the tile centres the terrain agrees best on, the same to 0.02 on both clips with crates, so the
   check moved right fixes 0.4 tile the wrong way. And odometry drifts: the terrain's best
   sub-tile placement moved up to half a tile in 15 s, crates or not.
   - Now: every 1 s, per axis, a tent is fitted to the agreement rates at the fix and a tile either
     side of it, summed over that second. An estimate past 0.2 tile moves the offset by it.
   - A step is not an epoch (section 6.2). Nothing moves while a neighbour agrees better than the
     fix: that is a whole tile, check 2's.
   - Crate residuals are still logged (`map_crate_resid`), as telemetry only.

**Guard.** Smoothed agreement under 0.6 for 2 s drops the fix, the same as an unconfirmed cut.
**K2:** the guard reads the best agreement within a tile of the fix, not at it. A fix one tile off
is check 2's to move, and a guard read at the fix dropped it first. A silent 5-tile odometry jump
is the guard's case: in 30 of 30 runs it dropped the fix and the whole-map search found the right
one. **K5:** before the checks, a fix whose camera sits more than 5 tiles past the clamp limits
drops on that tick. The guard could not drop map 2's wrong fix, because too few of its cells were
on the map to rate (section 8, K5).

### 6.5 Odometry cuts

- **Carry.** A cut restarts odometry at a new origin. The localizer assumes the camera did not
  move across the cut: offset = last world position minus the new odometry position. In practice
  a cut leaves odometry's position where it was (`Odometry._cut`), so the carried offset is
  normally the old one, and a camera that moved inside the cut shows up as a confirm step.
- **Keep the epoch.** Today every cut resets occupancy, loot, gas, tracks and history. With a
  map, a confirmed cut resets nothing.
- **Confirm.** The accumulator from 6.3 runs over offsets within 4 tiles of the carried one.
  - A commit on the carried offset keeps the epoch.
  - A commit on any other is a correction, so it takes a new epoch.
- **Why local.** The search stays local while a recent fix exists. That is not about mirrored
  twins, which 6.1 rules out; it keeps open-floor lookalikes elsewhere on the map from competing.
- **Give up.** No commit within 5 s drops the fix.

### 6.6 Losing it

The loop goes back to no fix, which is today's pipeline. The localizer keeps a whole-map search
running on the current segment and returns to fixed, with a new epoch, when it commits.

- **K4: the floor after a first fix.** That commit, and a cut's confirm in 6.5, needs the guard's
  0.6 instead of `min_agreement`'s 0.65 (the user, 2026-09-28).
  - Why: a held-out match dropped a right fix where the view read under 0.6 at every placement.
    The search after it led by `whole_map_margin` within 0.8 s. But its running agreement began
    in that stretch and peaked at 0.648, so 39% of the match went unfixed. A fix could be kept
    at 0.6 but not found again under 0.65.
  - What stays: the margins, which are what rule out a wrong placement, and the landing's 0.65,
    which there also stands between a confident fit and the wrong map. **K5:** the landing's
    spawn windows now need only 0.6 as well; its whole-map search keeps 0.65 (section 8, K5).
  - The risk: in a stretch reading near 0.6, a fix could drop and return over and over, each
    return an epoch. The report counts drops and re-fixes after the first fix.
  - **K5:** the failure seen live was another: after a drop, a placement in the map's corner, its
    view mostly off the map, out-scored the right one and committed at 0.604 (section 8, K5). The
    camera box (6.3, 6.4) now stops it: that placement's camera sat 12 tiles or more past the
    limits.
  - Tests: a poor view (0.62-0.64) never lands, re-fixes after a drop, and confirms a cut. Five
    wrong versions of the rule each fail at least one of them. **K5:** the first became two: a
    poor view lands in the spawn windows, and not in the whole-map search however far it leads.
    The camera box has three more, and six wrong versions of it each fail at least one.

## 7. Code shape

- **`brawl_deployment/perception/known_map.py`**
  - `KnownMap` loads the CSV and runs the section 4 checks; the JSON sidecar is the tool's alone.
  - Built in K1:
    - `tiles`, the label as `Tile` ids with SPAWN kept;
    - `spawns`, the centres in world tiles.
  - Built in K2: `classes`, the label in `CLASSES` indices, with `S` as FLOOR.
  - Built in K3: `world_grid(height, width)`, the label painted into the 128 x 128 frame, WALL
    off the map.
  - `KnownTerrain` has `height`, `width`, `origin` and `best()` with the occupancy map's meanings,
    so `GridBuilder` and `GasMap.from_occupancy` take it unchanged. `best()` returns the label
    while fixed and `occupancy.best()` otherwise.
  - **K3:** "while fixed" became "while `world` handed out the map frame this tick"
    (`MapLocalizer.in_map_frame`). That covers a fix carried across a cut while it is confirmed,
    whose loot and tracks are kept, so the terrain does not flicker to the occupancy map for those
    ticks. It follows the tick's frame rather than the state because on the tick `observe`
    commits, the hero and every track are still in the lattice frame, and the label read there
    would be the map at the wrong place.
- **`brawl_deployment/perception/localize.py`**
  - `score_placements(view, scored, label)`: section 6.1, pure numpy and cv2. It returns the score
    and the counted cells for every whole-tile placement that overlaps the map.
  - `MapLocalizer(known, clamp_onset, ...)` owns a `LatticePhase` and the one epoch counter.
    `clamp_onset` is the sim's `camera_clamp_onset`, for the landing windows and, **K5**, the
    camera box, whose reach past it is the module constant `CAMERA_TILES`. Its members:
    - `reset(segment)`, `update(detections, plan, odometry)`, `world(odometry)` and `epoch`, the
      four the loop already calls on `LatticePhase`;
    - the new `observe(t, cells, confidence, plan, world, zone, occluded)`.
  - `observe` runs after classification, and any correction it makes applies from the next tick.
    The offset only moves in jumps, so one tick of lag costs nothing.
  - `MapResult` carries the state, offset, agreement, margin, crate residuals and a one-line
    event for the log.
- **`LatticeResult.sightings`** (K2): the tick's full-height crate sightings in odometry's frame,
  which check 3 reads against the fix instead of the lattice. **K4:** they now feed only the
  `map_crate_resid` telemetry, since check 3 reads the terrain.
- **`DeployLoop.__init__`** builds `MapLocalizer` and `KnownTerrain` when a map is named, otherwise
  `LatticePhase` and the occupancy map as now.
  - `_perceive` keeps the classifier's confidence, which it discards today with `cells, _ =`.
  - It calls `observe` right after `occupancy.update`.
  - **K3:** `self.terrain` is what the grid and the loop's bush and gas lookups read, and
    `self.localizer` is the localizer, or None without a map. The occupancy map still takes every
    deposit, as the fallback and for K4's label check.
- **Config:** a `map:` block in `configs/deployment.yaml`, with matching `DeploymentConfig` fields.
  - `name: null`, where null means today's behaviour.
  - `commit_margin`, `whole_map_margin`, `slip_margin`, `min_agreement`, `guard_agreement`,
    `wrong_map_agreement`, `search_tiles`, `search_seconds`: `MapLocalizer`'s keyword
    arguments, with its defaults.
- **CLI:** `deploy_run.py --map dark_passage` and `scripts/probes/replay_clip.py --map dark_passage`
  override the config for one invocation, the way `--run` does.
  - **K3:** `deploy_run.py` loads and checks the named label before it builds anything, and on a
    typo lists the labelled maps, as it does for a missing checkpoint.
- **Telemetry:** new `TickRow` columns `map_state`, `map_dx`, `map_dy`, `map_agree`, `map_margin`
  and `map_crate_resid`. The last one is this tick's crate-phase residuals; it is empty when no
  full-height crate is in view.
  - **K3:** a float column with no value holds NaN and a text column is empty; without a map every
    row is that way. Each localizer event is also logged, as `map: <event>`.
  - **K5:** `map_lead_dx` and `map_lead_dy`, the search's leader as the offset a commit on it
    would take, NaN while fixed; `MapResult.leader` carries it.
- **Rendering:** anything that draws the deploy grid for a human reads `KnownTerrain.best()`, so
  the render and the policy's grid cannot disagree.
  - **K3:** nothing in the deploy stack draws the grid for a human yet, so there was nothing to
    rewire. `scripts/vision_watch.py` draws the occupancy map, but it runs no loop and names no
    map. A deploy overlay added later reads `loop.terrain.best()`.

## 8. Steps

### K1. Labeling tool and the Dark Passage label

- **Build:**
  - `scripts/map_label.py` (section 5);
  - `KnownMap`'s loader and checks (section 7), since the tool's save runs them.
  - Copy the image to `brawl_deployment/data/maps/dark_passage.png`.
- **Label:** Claude runs the first pass; the user corrects it in the tool.
- **Tests, in `tests/test_known_map.py`:**
  - The checks reject a 59-row file, a stray character, an `X` and a sealed-off pocket.
  - The loader round-trips the file, with spawn centres at +0.5.
  - On the checked-in image, the spawn finder returns exactly the 10 spawns in section 1.
- **Done when:** `dark_passage.csv` passes the checks with 10 spawns and no crates, and the user
  has signed off on its overlay.
- **Status 2026-09-28: DONE.** The user signed off, and 17 tests pass. Beyond the rejections
  above, they cover:
  - a ragged row, a long cell, `?` and a missing spawn;
  - a bush ring, which seals nothing;
  - the tool run headless end to end: paint, cluster fill, undo, save, reopen;
  - the guard that stops a different image replacing a labelled one.

### K2. Score and localizer, pure

- **Build:** `localize.py`, with `score_placements`, `MapLocalizer` and `MapResult`. It does not
  depend on the loop.
- **Test views:** cut from the label, but damaged realistically, never clean copies:
  - the HUD hole masked;
  - a gassed band;
  - the classifier's errors at its held-out BlueStacks F1 (.851): 15% of the non-floor cells read
    as floor, and as many floor cells beside them read as a neighbour's class, a third of them
    under 0.5 confidence. **K2:** the first draft's "15% of cells" meant 15% of the footprint,
    which is about 40% of the cells the score counts. The leader was still the true placement at
    every spawn on the first tick, but its agreement sat at 0.56-0.67, under the commit's 0.65;
  - a few walls turned to floor as if destroyed;
  - a brawler box masked.
- **Cases:**
  1. Every spawn is picked right within 3 ticks.
  2. An all-floor view never commits.
  3. A whole-tile drift injected mid-match is corrected once.
  4. A cut with no motion keeps the epoch.
  5. A cut with a 3-tile jump is found locally, with a new epoch.
  6. A view cut from `brawl_sim/maps/csv/ghost_point.csv`, scored against Dark Passage, ends in
     the wrong-map log.
- **Done when:** all six pass.
- **Status 2026-09-28: DONE.** `tests/test_deployment_localize.py`, 28 tests, all pass. It holds
  the six cases (1 and 2 at all ten spawns) plus:
  - the score against a direct count, and fence scored as wall;
  - a silent 5-tile odometry jump, which the guard drops and the whole-map search finds again;
  - check 3: crates 0.3 and -0.2 tile off their centres move the fix by exactly that. **K4:**
    replaced by terrain tests when check 3 moved to the terrain; the file now holds 30.
- Seeded sweeps beyond the tests, on the same views:
  - 200 of 200 spawn landings were fixed right, in 1 tick (142) or 2 (58);
  - the one-tile slip, the silent jump and the wrong-map numbers are in 6.3 and 6.4;
  - no false alarm in 80 walks of 12 s with the fix right.

### K3. Loop wiring

- **Build:** `KnownTerrain`, the `map:` config, the two CLI flags and the telemetry columns
  (section 7).
- **Tests, in `tests/test_deployment_loop.py` with its existing fakes:**
  1. With a map set and the pose known:
     - the grid's static planes equal the label crop;
     - off-map cells are WALL;
     - `pos_norm` equals col / 60.
  2. With `map.name: null`, the output is unchanged from today's.
  3. A crate sighted before a confirmed odometry cut is still in the box plane after it.
- **Done when:** these pass, and the vision-free suite passes apart from the known stage-walk
  pin failure.
- **Status 2026-09-28: DONE.** The three tests pass. Each of five deliberate breaks in the code
  failed at least one of them:
  - the terrain following the localizer's state instead of this tick's frame;
  - the terrain never reading the label;
  - the label painted one column east;
  - a confirmed cut opening a new epoch;
  - the offset columns left unwritten.
- The vision-free suite: 2775 passed, 4 failed. The failures are the stage-walk pin and the three
  auto-aim real-checkpoint policy tests, which failed before K3 too (found in K2).
- The loop tests' fake plan puts the viewport centre 11 tiles from the hero, so the landing prior
  would miss there. The map tests use `_MapPlan`, whose viewport centre projects onto the hero's
  anchor, as the real camera's does.
- The first suite run also caught a K1 leak: the label tool's tests left matplotlib's global
  keymaps trimmed, and `test_play_manual` failed after them. `test_known_map.py` now restores
  rcParams after each test.

### K4. Offline gate on recorded Dark Passage matches

- **Record.** Record 3 matches on Dark Passage in BlueStacks, from different spawns if possible,
  noting which spawn each one started on.
  - The recording must include the match start, since the fix anchors on the spawn.
  - Either the current model plays through `deploy_run.py`, or the user plays by hand while OBS
    records.
- **Footage, recorded 2026-09-28.** All five start in the loading screen, were played by hand in
  BlueStacks and recorded with OBS, and sit in `brawl_vision/data/training_videos/`. The
  `bluestacks-` prefix is what gives them the emulator HUD mask. Spawns were checked by eye
  against the map image.

| clip | spawn (col, row) | length | use |
|---|---|---|---|
| `bluestacks-dark-passage-bottom-left.mp4` | (6, 47) | 42.1 s | tuning, a wall walk |
| `bluestacks-dark-passage-bottom-right.mp4` | (47, 53) | 61.1 s | tuning, a wall walk |
| `bluestacks-dark-passage-top-right.mp4` | (43, 6) | 44.5 s | tuning, a wall walk |
| `bluestacks-dark-passage-holdout-left-center-bottom.mp4` | (6, 47) | 37.4 s | held out, a whole match |
| `bluestacks-dark-passage-holdout-left-center-top.mp4` | (6, 24) | 48.6 s | held out, a whole match |

  - The held-out pair is replayed once, after the thresholds are set, to check the result. Nothing
    is tuned on it. Its frames are not exported for vision labeling either, since
    `vision_export_frames.py` with no clip names takes every .mp4 in that folder.
  - The first held-out match landed on the same spawn as the first tuning clip, so only its
    landing is not new. Everything after the gate is.
  - First replays of the tuning clips, with the provisional thresholds, already miss the gates.
    The landing was right on all 3, at the gate tick or 0.5 s after it. Right fixes then dropped:
    the true map's agreement sat at a median of 0.67-0.74 and a 5th percentile of 0.43-0.57,
    against the guard's 0.6. The crate-phase check moved two right fixes 0.37-0.49 tile the wrong
    way in y. On top-right the whole-map search committed 48 tiles off, for 1.7 s. Time without
    a correct fix was 34%, 25% and 71%.
  - Replaying the clips back to back on one `VisionStack` changed the result: top-right never
    fixed. `deploy_run.py --matches` above 1 keeps one stack the same way, so K4 should find what
    carries over between matches.
- **Replay.** Replay each through the real loop with `replay_clip.py --map dark_passage --device
  cpu --csv <out>`.
- **Report.** A small `scripts/probes/map_localize_report.py` reads each CSV and reports:
  - time to commit and the spawn picked;
  - agreement over time and corrections per minute;
  - crate residual RMS and time in each state. **K4:** the crate figure became their spread about
    their own circular mean, the map's crate offset, since crates no longer move the fix. The user
    signed it off at 0.25 tile. The report also counts drops and re-fixes after the first fix.
- **Label check.** After each replay, crop the occupancy map to the map's 60 x 60 window and run
  it through `compare_to_truth` against the label.
  - Best offset (0, 0) means the classifier's own map lands on the label.
  - Cells the classifier contradicts in most views are listed for the user to re-check in the
    tool. Each is either a label error or a classifier error.
- **Thresholds.** Set `commit_margin`, `slip_margin`, `min_agreement` and `wrong_map_agreement`
  from these numbers. Record them in the yaml comments with the date. The true map's agreement at
  the fix is the number to watch. On K2's damaged views its smoothed value was 0.78 at the median,
  0.72 at the 5th percentile and 0.65 at worst over 11,520 ticks, against the guard's 0.6. The
  classifier's real errors could put it lower than any of these. **K4:** they did, to 0.73-0.79 at
  the median, 0.65 at the 5th percentile and 0.53 at worst on the tuning clips, but no right fix
  was dropped there, so all eight values stayed. The evidence is in the yaml comment.
- **Clamp onsets.** Each recording starts on a spawn, most of them inside the clamp band, so the
  replays show where the camera really stops. If the landing windows miss by more than their 2
  tiles, the fix still comes from the whole-map search after 5 s, only later.
- **Gates:**

| gate | target |
|---|---|
| spawn picked right | 3 of 3 matches |
| time to commit after the gate | under 2 s |
| crate residual RMS after the fix; **K4:** their spread about the map's crate offset | at most 0.2 tile; **K4:** 0.25, the user's |
| time without a fix | under 5% of the match |
| label check best offset | (0, 0) |

- **Only if the numbers ask for it.** Crate-free stretches late in a match might show sub-tile
  drift over 0.2 tile. If they do, add a sub-tile terrain check: the same score, with the label
  shifted by quarter tiles. **K4:** they asked, crates or not. Built as check 3 in 6.4, a tent
  fitted to the whole-tile neighbours' agreement, which needs no shifted labels.
- **Status 2026-09-28: RUN. The tuning clips pass, and the held-out pair misses two gates.**
  - **What changed**, all marked "K4:" above:
    - check 3 reads the sub-tile part from the terrain, not the crates, and its steps open no
      epoch (6.2, 6.4);
    - a search after a dropped fix runs in the lost fix's frame (6.3);
    - the report's crate figure is their spread about the map's crate offset.
  - **Why.** The first replays' four problems had one cause, the fix's sub-tile part. They were
    the dropped right fixes, the crate check's wrong-way steps, the whole-map fix 48 tiles off and
    the change back to back. A fix made without crates took odometry's arbitrary sub-tile phase,
    and that phase was what carried over from one match to the next. On real views the tent reads
    0.3-0.85 of a sub-tile error, so check 3 closes one over a few steps. Run closed loop on the
    tuning clips' views, it lifted the fix's median agreement from 0.66 to 0.76 and cut the time
    under the guard's 0.6 from 19% to 6.5%.
  - **Tuning clips**, each replayed fresh:

| clip | match after the gate | right fix | a tile off | no fix | agreement while fixed, median / 5th pct / min | sub-tile steps a minute | crate spread | label check best offset |
|---|---|---|---|---|---|---|---|---|
| bottom-left | 27.6 s | 95% | 5% | 0% | 0.74 / 0.65 / 0.56 | 6.5 | 0.208, 0.008 over | (0, 0) |
| bottom-right | 49.7 s | 98% | 2% | 0% | 0.79 / 0.65 / 0.55 | 10.9 | 0.186 | (0, 0) |
| top-right | 36.9 s | 100% | 0% | 0.3% | 0.73 / 0.65 / 0.53 | 6.5 | no crates | (0, 0) |

  - Every landing committed on the right spawn within 0.1 s of the gate. No fix was dropped or
    wrong, and none moved a whole tile. Back to back on one stack the three gave the same, right
    95-100% with a 5th percentile of 0.64-0.65, so nothing carries over now.
  - The "a tile off" stretches are the report's truth drifting, not the fix. The truth is the hero
    on its spawn at the gate, and odometry drifts up to half a tile in 15 s. No neighbour's lead
    came within 22 cells of `slip_margin` in them.
  - **Held out**, replayed once after the thresholds were set:

| match | spawn picked | commit after the gate | right fix | a tile off | no fix | agreement while fixed, median / 5th pct | crate spread | label check best offset |
|---|---|---|---|---|---|---|---|---|
| left-center-bottom | (6, 47), right | 0.0 s | 72% | 28% | 0% | 0.81 / 0.70 | 0.237, MISS | (0, 0) |
| left-center-top | (6, 24), right | 0.0 s | 61% | 0% | 39%, MISS | 0.71 / 0.58 | 0.126 | (0, 0) |

  - left-center-top dropped a right fix 23 s after the gate, near tile (27, 23) in the middle of
    the map. Its agreement fell from 0.72 to 0.58 in about 4 s while every neighbour still
    trailed, so the view read poorly at every placement there. The whole-map search then led by
    the 120 cells it needs within 0.8 s, but its agreement rose only to 0.648 in the 15 s left,
    under `min_agreement`'s 0.65, and it never committed. Its leader's offset is not logged, so
    whether it was right is unknown.
  - left-center-bottom's agreement never fell under 0.70 and no neighbour led. Its tile-off
    stretches read at most 0.63 from the gate's truth, the same drift as on the tuning clips as far
    as the log can tell. Its crate spread missed by 0.037.
  - Both held-out matches are now used, so a change made for either miss can only be checked on
    new footage.
  - **Label cells to re-check,** where most views on two tuning clips each disagreed with the
    label:

| cell (col, row) | label | the classifier's map |
|---|---|---|
| (28, 17) | floor | wall |
| (31, 5), (32, 5) | floor | wall |
| (22, 25) | floor | wall |
| (19, 28) | floor | wall |
| (34, 33) | floor | wall |
| (31, 14), (39, 18) | wall, both candles | floor |

  - The two candles stay walls by the user's rule; the classifier missed them, broken or not.
  - **The user checked the six floor cells (2026-09-28): each is the top half of a wall that
    overlaps the floor tile north of it in perspective. The label is right; nothing changes.**
  - **Perspective, unmeasured.** Raised sprites rise north on screen, so walls and bushes may
    pull the terrain fit south. On the two tuning clips with water in view, each class's best
    sub-tile fit, from the replays pinned to the spawn truth:

| clip | x: wall / bush / water | y: wall / bush / water |
|---|---|---|
| bottom-left | -0.36 / -0.32 / -0.35 | +0.38 or more / +0.30 / +0.21 |
| bottom-right | +0.08 / +0.07 / -0.06 | +0.10 / +0.26 / -0.40, a flat peak |

  - In x the three classes agree within 0.13; in y the raised ones fit 0.1-0.7 tile south of
    flat water. The combined fit sat 0.03 south of water on one clip and 0.49 on the other, so
    the fix may lean south, by an amount not yet known. The crate figure cannot see it.
  - **Settled by the user, 2026-09-28, and built:**
    1. The crate gate reads the spread about the map's crate offset, at most 0.25 tile. Check 3
       lets the fix wander about 0.2 tile between steps, and the detector adds its own scatter.
       And in perspective a crate box's centre is rarely its tile's centre (the user), by an
       amount that changes across the screen, which the offset cannot take out.
    2. After a match's first fix, a commit needs the guard's 0.6, not `min_agreement`'s 0.65 (6.6).
       By the log, left-center-top would then have committed 1.5 s after its drop, on a tile the
       log cannot name.
    - Both were set after the held-out pair's misses: at 0.25 its 0.237 would pass. So only the
      new pair tests them. The tuning clips never searched after their first fix, so item 2
      leaves their results unchanged, and at 0.25 all three pass.
  - **Open:** a new held-out pair, 2 more whole Dark Passage matches, recorded by the user. K5's
    dry run can be them, since its live CSVs go through the same report.

### K5. Live

- **Dry run.** Run `deploy_run.py --map dark_passage --dry-run --telemetry <csv>` while the user
  plays a Dark Passage match by hand. It builds and decides but sends no touches. Run the K4
  report on the live CSV.
  - **First attempts, 2026-09-28:** both fixed on Dark Passage at the gate, and both lost their CSV
    because its folder did not exist; `write_csv` now makes it. The first also stopped at the gate
    on the 1.0 s capture-stall guard: the gate's tick paid the detectors' warm-up as well as the
    landing, 1.22 s in all. The warm-up now runs on the run's first tick, before the gate, and is
    left out of the next stall check. The second ran its 15 s match through, and gave the first
    live timing with the localizer on: in-match ticks median 74.8 ms, p95 87.9, max 94.7, against
    the 83 ms budget at 12 Hz.
  - **The new held-out pair, 2026-09-28, reported once.** Neither match tests tracking while
    moving: in both the hero never left its spawn.

    | | match 1 | match 2 |
    |---|---|---|
    | spawn | (57, 38) | (34, 54) |
    | length after the gate | 32.9 s, killed by the gas | 5.4 s: the client loaded after the match began and the hero died where it stood |
    | result | fixed right at the gate, right all match | never fixed |
    | agreement | 0.80 until the gas reached the view at ~20 s, then 0.62-0.68, lowest 0.59 | leader 0.62-0.65 throughout |
    | crate spread | 0.111 | no crates |
    | in-match tick, median / p95 / max | 78.5 / 89.9 / 102.3 ms | 76.4 / 86.6 / 93.1 ms |

    Match 2's leader cleared the landing margin on the first tick (62, then 3946 by 5 s) and was
    held off only by `min_agreement`. Its offset is not logged, so whether it was the right
    placement is not known.
  - **Synthetic check of the landing floor**, no held-out footage: every sim map, up to 10 spawns
    each (378 wrong-map runs), and Dark Passage's 10 spawns at K2's damage and at `POOR`, 7 s from
    the gate each. "As built" is the rule the user chose below, re-run on the built code.

    | | 0.65 everywhere, before K5 | as built: 0.6 in the windows | 0.6 everywhere |
    |---|---|---|---|
    | wrong-map commits inside the spawn windows | 0 | 0 | 0 |
    | wrong-map commits in the whole-map search after 5 s | 4 | 4 | 7 |
    | Dark Passage at K2's damage, landed right | 10 / 10 | 10 / 10 | 10 / 10 |
    | Dark Passage at `POOR`, landed right | 4 / 10 | 9 / 10 | 10 / 10 |

    The tenth `POOR` spawn found no clear leader in its window and landed only in the whole-map
    search, 6.4 s in, which as built needs 0.65. At `POOR` the window landings took up to 4.9 s.

  - **The wrong-map guard is weaker than K2 measured on 12 maps.** With the 38 current sim maps, 4
    of 378 wrong-map runs commit at today's floor, all in the whole-map search, on placements that
    score 23-40 cells a tick. Right placements on Dark Passage score 37-231, median 77, so a
    minimum of scored cells cannot separate them. It matters only when `--map` names the wrong map.
  - **Decided by the user, 2026-09-28, and built:** a commit inside the spawn windows needs only
    the guard's 0.6, and the whole-map search before a first fix keeps 0.65. The search's leader
    is logged as well, in `map_lead_dx` / `map_lead_dy`: the offset a commit on it would take. The
    report prints it at the end of every stretch without a fix, so a match that never fixes can
    still be scored against its spawn. Tests: a poor view lands in the windows and not in the
    whole-map search; the leader is checked in the lattice's frame, a lost fix's frame and a cut's
    confirm, and is NaN while fixed. Six wrong versions of the change each fail a test.
  - **The second held-out pair, 2026-09-28, reported once**, played moving: both heroes started
    running within 0.4 s of the HUD showing, before the gate opened.

    | | match 3 | match 4 |
    |---|---|---|
    | spawn | (26, 5) | (6, 47) |
    | length after the gate | 46.1 s | 31.7 s |
    | landing | at the gate, right spawn | at the gate, right spawn |
    | time without a fix | 0% | 0%, with 0.2 s confirming after a cut |
    | fixes dropped | 0 | 0 |
    | agreement while fixed, median / 5th percentile / min | 0.69 / 0.62 / 0.55 | 0.81 / 0.67 / 0.57 |
    | corrections | 6 sub-tile, no whole-tile | 5 sub-tile, no whole-tile |
    | crate mean / spread | (+0.180, +0.483) / 0.184 | (-0.001, -0.493) / 0.239 |
    | in-match tick, median / p95 / max | 76.0 / 84.7 / 92.6 ms | 81.3 / 88.1 / 91.9 ms |

    - All four K4 gates pass in both. Match 3's fix read 0.64 on the tick after it landed, under
      the old 0.65 floor, so the window floor likely let it land at the gate.
    - The report called both fixes a tile off for 90% and 100% of the match. That was its truth,
      not the fixes: the gate opens about 0.5 s after the HUD shows (6 ring samples at 12 Hz), so
      at the first odometry reading the heroes were running at 2.9 and 2.8 tiles/s, about 0.7 and
      1.0 tile from the spawn. Along the axis each hero did not run, x in match 3 and y in match
      4, the truth holds: there the landing tick was 0.7 and 0.6 tile off, within 0.4 after the
      first check 1.1 s later, and within 0.15 from 10.9 s and 2.2 s on.
    - Still frames, read by eye against the map image to about 0.2 tile:

      | still moment | the fix put the hero at | the frame shows | fix minus frame |
      |---|---|---|---|
      | match 3, 10.7 s after the gate | (21.39, 15.81) | (21.8, 15.65) | (-0.4, +0.15) |
      | match 3, 32.7 s | (40.70, 20.48) | (40.65, 20.3) | (+0.05, +0.2) |
      | match 4, 18.5 s | (19.36, 41.02) | (19.3, 41.3) | (+0.05, -0.3) |

      The crate means agree with match 1's right fix to 0.15 tile modulo whole tiles, and the
      frames rule out a whole-tile error.
    - Neither match had a stretch without a fix, so the leader columns went unused.
    - The report now warns when the hero moves between its first two odometry readings. A
      held-out match needs a second standing still at the start and can be played moving after.
  - **Next:** with control, below.
- **With control.** Then run with control on, and compare wall-push decisions and match outcomes
  against the same checkpoint without `--map`.
  - **Run 2026-09-28, reported once:** six matches with the agent in control, the first three with
    `--map dark_passage` and the last three without, so the order was not alternated.

    | | map 1 | map 2 | map 3 | no map 1 | no map 2 | no map 3 |
    |---|---|---|---|---|---|---|
    | rank of 10 | 6 | 3 | 6 | 10 | 8 | 5 |
    | alive after the gate, s | 18.5 | 60.1 | 16.1 | 10.8 | 11.2 | 26.7 |
    | decisions | 70 | 232 | 59 | 37 | 41 | 75 |
    | stuck moves / moves scored | 1 / 61 | 8 / 195 | 1 / 41 | 0 / 29 | 0 / 32 | 2 / 55 |
    | longest stuck run, s | 0.25 | 1.0 | 0.25 | 0 | 0 | 0.5 |
    | decisions with a bin masked | 0% | 6% | 10% | 5% | 10% | 15% |
    | in-match tick median / p95, ms | 72.8 / 79.0 | 73.6 / 79.8 | 73.1 / 79.2 | 72.4 / 79.7 | 80.1 / 90.6 | 76.5 / 82.9 |

    - Stuck: the hero moved under 0.15 tile over the decision interval after the command's own,
      the one the command best matches (mean cosine 0.86); full speed is about 0.7. Intervals with
      odometry or the lattice unsure are skipped. A scratch measure: the sim's wall-push probe
      cannot read a live CSV.
    - Wall pushes were rare either way, 10 of 297 moves with the map and 2 of 116 without, and no
      run stalled for 2 s, so there was nothing here for the map to fix.
    - Mean rank 5.0 with the map and 7.7 without, 31.6 s alive against 16.2 s. At three a side the
      permutation p is 0.15 and 0.25, and every map match came first, so this does not show the map
      helping. The agent died within 11-27 s in five of the six, so live samples stay small.
    - The map costs no tick time.
    - Maps 1 and 3 pass every K4 gate: first fix 0.17 s and 0.09 s after the gate, 0.9% and 0.5% of
      the match without a fix, crate spread 0.122 and 0.241, and spawns (6, 5) and (15, 55) checked
      by eye against the gate's frames.
    - **Map 2 re-fixed 45 tiles wrong, the first wrong commit seen live.** It landed at the gate on
      (6, 47), and still frames show that fix right until it dropped at 47.1 s: its agreement fell
      from 0.67 at 44 s to 0.54 as the hero went into the bush patch at the top of the L wall. The
      search led on the right placement by up to 304 but read 0.53-0.58, under the 0.6 floor. From
      52.2 s, as the hero ran north toward the lake, that lead fell from 254 to 36 in 0.6 s, and at
      53.0 s a placement 45 tiles east and 36 north committed at 0.604 with a lead past 120. It sits
      in the map's north-east corner: the lake and the whole view east of the hero fall off the map
      there, where nothing counts, and what is left, the bush band under the lake and the bush strip
      beside the wall, lands on the bush band along the north edge and the bush column down the east
      edge. Odometry never cut.
    - The wrong fix then read 0.73-0.86 as its view slid further off the corner, and from 57.5 s too
      few of its cells were on the map to rate, under 20 a tick, so the guard could never drop it.
      It held for the last 7.1 s, 12% of the match. From 58.0 s the hero pushed north into the lake
      for 1 s with every bin legal, while the wrong fix placed it off the map at (68, -0.3); at
      136 HP, a bot across the lake killed it at 59.6 s.
    - Map 2 misses the unfixed gate, 9.8%, with the box below as well, because the right placement
      never read 0.6. Why it read low is open: the lake, or the bushes turning see-through around the
      hero, are the first suspects, and gas covered the view's west third from 41 s.
  - **Decided by the user, 2026-09-29, and built: the camera box.** A placement may lead and
    commit, in the spawn windows and the whole-map search alike, and a fix may stay, only while
    the camera's nominal point it implies is within `CAMERA_TILES` of the sim's clamp limits. A fix
    past it drops on that tick, with no 2 s wait, since no misread puts the camera there.
    - `CAMERA_TILES` is 5, not the 2 first proposed. The game's camera goes further than the
      sim's, and right fixes on the live matches and the tuning clips put it this far past the
      sim's limits:

      | edge | furthest past, right fixes | where |
      |---|---|---|
      | west | 1.9 | live and clips |
      | east | 2.5 | bottom-right clip |
      | north | 2.9 | live map 1; top-right clip 2.8 |
      | south | 2.9 | bottom-right clip; live map 3 2.0 |

      At 2 the box would have dropped right fixes on three edges. Five covers 2.9, a fix a tile off
      that check 2 has yet to move, and a tile to spare. Map 2's corner fix sat 12.2 to 21.3 tiles
      past the east limit and up to 8.3 past the north, so 5 still stops it with 7 to spare.
    - The first proposal said the spawn windows use the same 2 tiles. They do not: their 2 tiles
      is how far each window reaches past the spawn and past the sim's clamped point, not a margin
      on the camera. At spawn the game's camera sits between those two points, so the windows need
      no change.
    - Tests: the three in 6.6. A corner placement 20 tiles off, with its view mostly off the map,
      never commits after a drop, and the right one re-fixes. A fix moved off the map drops on
      that tick. A camera 3 tiles past the limits on three edges keeps its fix. Each has a check
      that it fails without the box, or with 2 tiles. Six wrong versions of the box each fail one
      of them. The deployment and map suites pass, 539 tests.
    - The loop tests' fake camera sat 5.6 tiles past the west limit, where no game camera goes, so
      three of them lost their fix. The fixture now stops the camera 1.6 tiles past, with the hero
      still on the spawn.
    - The three tuning clips, replayed with the box: all landed right at the gate and nothing
      dropped. Their fixes stayed within 1.9, 2.9 and 2.8 tiles of the limits, so the box never
      fired on them. Replayed again with the box off, every map, odometry and hero column came out
      the same row for row.
    - On map 2, the box rules out the corner. The right placement read 0.53-0.58 there, under the
      0.6 floor, so the last 7 s would likely have gone unfixed rather than wrong.
    - **Next:** new held-out matches with `--map dark_passage`, played by hand, standing still
      for about 1 s at the start. Runs along the map's edges and into its corners test the box
      most. It passes if no right fix drops on the box, nothing commits wrong, and the K4 gates
      pass. The K5 control matches motivated the box, so they cannot test it.
  - **The camera box's held-out pair, 2026-09-29, reported once** (`k5_box1`, `k5_box2`, played by
    hand): the box passed. It never fired, the fixes put the camera up to 3.0 tiles past the sim's
    limits, and nothing committed wrong. Match 2 missed the unfixed gate at 6.7%, because the
    older agreement guard dropped its right fix in bushes for 1.6 s.

    | | match 1 | match 2 |
    |---|---|---|
    | spawn | (34, 54) | (6, 5) |
    | length after the gate | 17.8 s | 25.0 s |
    | landing | at the gate, right spawn | 0.09 s after the gate, right spawn |
    | time without a fix | 0% | 6.7% |
    | fixes dropped | 0 | 1, by the guard, found again 1.6 s later on the same placement |
    | fixes dropped by the box | 0 | 0 |
    | furthest past the limits, west / east / north / south | not near / 2.5 / not near / 3.0 | 2.0 / 2.6 / 3.0 / not near |
    | agreement while fixed, median / 5th percentile / min | 0.77 / 0.65 / 0.55 | 0.77 / 0.58 / 0.56 |
    | corrections | 1 sub-tile, no whole-tile | 2 sub-tile, no whole-tile |
    | fix minus the video's truth, in order | (-0.11, -0.18), (-0.13, +0.27) | (-0.65, -0.63), (-0.38, -0.41), (-0.58, -0.49) |
    | crate mean / spread | (-0.169, +0.445) / 0.206 | (+0.292, +0.468) / 0.113 |
    | in-match tick, median / p95 / max | 75.2 / 83.8 / 94.4 ms | 73.5 / 81.0 / 94.2 ms |

    - The truth is the videos'. Both heroes had moved before the gate opened, (+0.51, +0.24) and
      (+0.31, -0.15) tiles, so the report's own truth, the hero on its spawn at the gate, was off
      again: by it, match 1 read a tile off for 94% of the match. The video truth takes the hero's
      path from the first frame with the HUD, when it stands on its spawn, and fits it to the
      hero's odometry over the first 4 s after the gate. Fits over 2.5 s and 6 s move it 0.15
      tile or less.
    - Furthest past is read on decision ticks, the only ones that log the hero's offset. Each edge
      reached 0.1 tile further than the table above, with 2 tiles still to spare.
    - Match 2's drop came as the hero ran east through the bush band along the north edge, from
      map (34, 1.9) to (44.6, 3.3), with the camera 2.8 tiles past the north limit. Agreement slid
      from 0.66 to 0.56 over 2.8 s, the last 2 s under 0.6. The search led on the same placement
      from its first tick, its lead reaching 667, and committed when its agreement reached 0.6;
      the fix then read 0.79. It is the second right fix lost in bushes, after map 2's with
      control. While unfixed the grid reads the occupancy map, so the cost is map-free terrain for
      those seconds, not wrong terrain.
    - Match 1's fixes sat within 0.3 tile of the truth. Match 2's sat 0.4 to 0.65 tile off on both
      axes, well under the 1.5 that counts as wrong: its first sub-tile check, at 1.2 s, moved it
      from 0.65 to 0.4 off, and one at 19.2 s moved it to 0.6. That is the size of the terrain
      against hero disagreement K4 left open, and a truth read off the hero cannot say which is
      off.
    - The pair is spent.
    - **Next:** the box is done. The one gate missed is the guard's drop in bushes. Whether to
      change the guard there is the user's call, and a change would need new held-out matches.

## 9. Later, not in this plan

| item | note |
|---|---|
| L1 terrain destruction | Walls stay walls in the label and the sim. A destroyed wall already shows in the checks as disagreement that stays in one cell; a later step can act on that. Dark Passage's 30 candles, listed in section 1, are the walls that break most easily. |
| L2 gas from symmetry | The map centre is now known, so gas fronts seen on one side could be mirrored to the other. |
| L3 automatic map recognition | Scoring the landing view against every labelled map would pick the map without `--map`. |
| L4 training on Dark Passage | Needs a sim copy of the label, with the wall ring added, edge spawns moved in and crate spots chosen. The user's call. |
| L5 more maps | Each one is K1's tool plus a replay through K4's gate. |

**A gap this plan keeps on purpose:** crates block movement in the game but are walkable in the
sim, which was the user's decision. The label-backed grid follows the sim.
