# Map CSVs

Each `csv/<name>.csv` is a plain-text grid: one row per map row, comma-separated single
characters, no header. `blank.csv` is 20x20 (paired with the `debug_tiny` preset, which
overrides `world.map_h`/`map_w` to 20 -- see `configs/presets/debug_tiny.yaml`). Every other map
is 60x60, matching `configs/default.yaml`'s `world.map_h`/`map_w`. All maps referenced together
by one `EnvConfig` must share the same dimensions, since a single per-env `map_id` selects among
them and the padded `MapBank` tensors (Step 6) assume one shape.

## The maps

| Map | Character |
|-----|-----------|
| `blank` | 20x20, almost empty. Tests only. |
| `open` | Sparse wall clusters, very little bush. Long sightlines. |
| `bushy` | ~24% bush tiles. Concealment everywhere. |
| `walled` | Dense walls, a water channel. No bush at all. Not in the rotation -- see below. |
| `skull_creek` | A creek of separate ponds on the diagonal with bush banks, rock formations inland. Modeled on the Solo Showdown map. |
| `feast_or_famine` | A thick bush field in the centre holding 10 of the 16 boxes, with bare outskirts. Modeled on the Solo Showdown map. |
| `scorched_stone` | Cluttered: ~40 small crate clusters, wall stubs everywhere (its barrel litter, since 2026-09 walls where they touched rock and floor elsewhere), grass bands running in off every edge, and a central grass keep whose crate core is ringed by 8 of its 44 boxes. Modeled on the Solo Showdown map. |
| `island_invasion` | One island in open water: a 36%-bush ring with eight floor alcoves punched into it, an open arena in the middle, water biting into the coast between eight headlands. Modeled on the Solo Showdown map. |

`skull_creek` and `feast_or_famine` were added to fill two gaps: water as a hazard that actually
shapes movement (only `walled` used it), and a map where the loot is concentrated somewhere
dangerous, so approaching the middle is a real decision rather than a default.

`scorched_stone` and `island_invasion` fill two more. `scorched_stone` is the cluttered one:
crate clusters and wall stubs everywhere, so no sightline runs far. And `island_invasion` is the
first map where water is the BOUNDARY rather than an inland hazard, which makes its coast the one
place on any map where you can be shot from a direction you cannot walk to. It is also the
bushiest map in the rotation at 35.9%, against `bushy`'s 24%.

**Fences are not a sim tile** (since 2026-09). They are rare enough in real Solo Showdown maps
that none of the reference layouts has one, so the map vocabulary
(`constants.MAP_CHAR_TO_TILE`) no longer accepts `f` and the loader refuses a CSV that carries
it. `scorched_stone`'s 152 barrel tiles and `feast_or_famine`'s 73 became walls where they
touched an existing wall and floor elsewhere; `walled`'s 27 became walls. `Tile.FENCE` itself
survives only because `brawl_vision`'s terrain classes and label files index it.

## The ten generated maps (2026-09, SIM_OVERHAUL_PLAN.md Step M3)

Ten more 60x60 maps, made by `brawl_sim/maps/generate.py` (plan Step M2) rather than by hand.
Shares are interior (rows/cols 1..58, so 3364 cells; a moat counts as water); every one has 16
spawns; "attempts" is always 1 because the seed in the table IS the seed that passed, so
`generate(seed, family, symmetry=symmetry)` returns the shipped grid byte-for-byte
(`tests/test_maps.py::test_generated_map_is_reproduced_by_its_seed_at_the_first_attempt`).

| Map | Family | Seed | Symmetry | Wall | Bush | Water | Boxes |
|-----|--------|-----:|----------|-----:|-----:|------:|------:|
| `broken_wall` | standard | 2 | point | 8.4% | 24.4% | 5.3% | 26 |
| `stone_fort` | standard | 3 | point | 10.0% | 27.1% | 3.2% | 30 |
| `twin_ponds` | standard | 4 | point | 9.3% | 22.9% | 6.2% | 28 |
| `cross_creek` | standard | 100 | point | 9.4% | 23.4% | 6.6% | 32 |
| `split_river` | standard | 303 | point | 9.2% | 21.8% | 5.2% | 26 |
| `narrow_pass` | standard | 502 | mirror | 9.6% | 25.3% | 6.4% | 22 |
| `dry_gulch` | open | 702 | point | 4.6% | 15.2% | 2.2% | 18 |
| `thorn_field` | dense | 800 | mirror | 10.6% | 34.5% | 4.3% | 24 |
| `reed_marsh` | water_border | 901 | point | 7.5% | 23.0% | 15.4% | 26 |
| `hollow_ring` | water_border | 904 | point | 8.3% | 23.1% | 13.0% | 20 |

What each one reads as in the ASCII render:

- `broken_wall`: 6-wide wall slabs chopped into stubs by bush, small ponds between them, crate
  pairs along the edges.
- `stone_fort`: 5x5 wall forts with grass skirts, a 10x2 channel top and bottom, the most crates
  of the standard picks.
- `twin_ponds`: ponds in mirrored pairs (a 5x5 pair on the east and its twin on the west), few
  channels, walls as 4x4 and 6x3 blocks.
- `cross_creek`: straight 1-wide channels running both ways so the lanes cross; 32 crates.
- `split_river`: two parallel 2-wide, 12-long channels down the middle with a bush strip between
  them, 10x2 channels near the top and bottom edges.
- `narrow_pass` (the one mirror standard map): twin vertical canals at the top centre leave a
  2-wide floor lane between them, a bridge you can be shot on from both sides; the halves face
  each other left-right.
- `dry_gulch` (the open outlier of the ten): 3x4 and 4x4 blocks spread thin, one pond pair, a
  bush field in the north-west corner; 2.2% water, hence the name.
- `thorn_field` (dense, mirror): 34.5% bush everywhere, a 2-wide central river, wall pairs
  facing each other across it.
- `reed_marsh` (water border): the 2-tile moat with its four gaps, and a marsh of five small
  ponds strung across the middle rows.
- `hollow_ring` (water border): the moat around a ring of 4x4 blocks and 5x3 slabs, small ponds
  in the centre, a 1-wide channel down the middle.

Picked from 26 candidates (`runs/maps_candidates/`, regenerable with `scripts/gen_maps.py`),
judged on the ASCII render against the six reference screenshots (plan §1.4: wall clusters
with grass skirts, small ponds and one or two narrow channels, crates singly and in pairs,
spread bush, point symmetry that reads as a Showdown layout) and the two design rules below.
Every rejected seed, and why:

- standard 1: look-alike of 303 with weaker features (single straight channels, one L-shaped
  slab); 303 kept instead.
- standard 101: crates stacked in columns along the east and west border rows -- nothing in
  the screenshots puts crates on the edge like that.
- standard 102: two 6x3 slabs stacked on the west side (rows 40-45) read as one 6-tall barrier.
- standard 103: two 7x7 ponds dominate; the same pond motif as 4 with less else going on.
- standard 300: two 2-wide vertical channels stacked at cols 13-15 (rows 3-19) read as one
  17-long barrier.
- standard 301: no distinct motif; the closest look-alike of 2.
- standard 302: generator-rejected, water 2.5% (below the 3% band).
- standard 304: the north-west bush field is the only feature; look-alike of 4.
- standard mirror 500: generator-rejected, water 2.8%.
- standard mirror 501: bare top and bottom rows, bare corners, and the water band across rows
  50-54 reads as a river across the bottom.
- standard mirror 503: a 1-wide water column 20 tall through the centre (rows 19-38) plus a
  bare band across rows 39-41.
- standard mirror 504: 9x5 ponds in all four corners and two bare rows top and bottom.
- open 700: the 6x4 slab in the north-west and its mirror are the only walls that matter; the
  rest is stubs.
- open 701: a bare 40x7 band across rows 7-13 and its mirror; the emptiest of the three.
- dense 801: a 7x5 wall block dead centre (rows 16-20) and its mirror turn the middle into a
  corridor.
- dense 802: four channels along row 54 read as a broken second moat.
- water_border 900: generator-rejected, water 12.0% (just under the band's floor, after
  rounding).
- water_border 902: the lowest pocket ratio of the batch (0.29) and a 1-wide channel running
  parallel to the west moat, which makes a 2-lane corridor.
- water_border 903: a 9x2 channel two rows inside the north moat (rows 12-13) doubles the moat.

**Human review is still pending.** The picks above are one reader's judgment of the ASCII
renders; the plan's acceptance step is a look at the ten renders next to the screenshots. To
swap one: run `scripts/gen_maps.py --family <family> --seed <n> --preview` (add
`--symmetry mirror` for the mirror maps) until a render reads right, copy the CSV over
`csv/<name>.csv`, and update the row above and the seed table in `tests/test_maps.py`
(`GENERATED_MAPS`). Keep the name: it is what `tests/test_map_csvs.py` (`DIMENSIONS`),
`brawl_sim/config.py` (`_KNOWN_MAP_NAMES`) and `world.maps` refer to.

**In the rotation since 2026-09-18.** `world.maps` in `configs/default.yaml` lists all sixteen
(the six originals plus these ten), and `brawl_sim/config.py`'s `_KNOWN_MAP_NAMES` registers
them, which `validate` checks every `world.maps` / `fixed_map` name against. A brand-new map
therefore needs its CSV, a `_KNOWN_MAP_NAMES` entry, a `world.maps` entry, a `DIMENSIONS` row
and a `GENERATED_MAPS` row. The flip made checkpoints trained on the six-map rotation
mismatched (see below); the plan accepts that.

## What actually trains

`configs/default.yaml`'s `world.maps` is the training rotation, and it is NOT every file in
`csv/`. Two maps are excluded on purpose:

- `blank` is 20x20 and only loadable under the `debug_tiny` preset.
- `walled` is the only 60x60 map with no bush at all, so nothing on it exercises concealment --
  the mechanic the rest of the rotation is built around. It is kept as a fixture
  (`tests/test_map_csvs.py::test_walled_has_water` is the only coverage of a
  bush-free map surviving the loader) and stays pinnable via `scripts/watch.py --map walled`,
  but no training run selects it.

Everything else in both tables (the six originals and the ten generated maps) is in the
rotation, drawn uniformly per episode. Note that
`cfg.map_names` is part of what a policy trains against, so changing this list makes existing
checkpoints mismatched -- see the note in the repo-root `README.md`.

## Designing a map for THIS simulator

Two rules that are not obvious, both learned by building maps that violated them:

- **No long solid barriers, and no enclosed pockets.** Bots steer; they do not path-find
  (`bots/steering.py` plus `terrain.resolve_move`'s axis-separated slide). A creek spanning the
  map with three crossings, or wedge-shaped pockets behind radial wall spurs, both produce bots
  that grind along an obstacle instead of routing around it. `skull_creek`'s water is a chain of
  separate ponds for exactly this reason -- a local hazard you slide past, never a wall.
- **Spread bush across the map, not into one blob.** `bots.hunt_cell_tiles` subsamples one hunter
  waypoint per 10x10 cell, so a map whose bush all sits in two cells gives hunters only two places
  to look. `skull_creek` yields 24 waypoints and `feast_or_famine` 30, against 36 possible cells;
  `scorched_stone` and `island_invasion` both reach all 36, the first from its edge-to-edge grass
  bands and the second because its bush ring wraps the whole island.

## Legend

| Char | Tile    | Unit | Projectile |
|------|---------|------|-----------|
| `.`  | FLOOR   | pass | pass |
| `#`  | WALL    | block | block |
| `b`  | BUSH    | pass | pass (hides occupants -- see `bots/perception.py`, not a tile rule) |
| `~`  | WATER   | block | pass |
| `S`  | SPAWN   | pass | pass |
| `X`  | BOX     | pass | pass |

Grids are row-major with `origin="upper"`, so **y increases downward**, matching CSV row order.
`loader.validate_map` enforces four things on load: the outer border is entirely wall, 8-32
spawn markers, 8-64 box markers, and the unit-passable region is a single connected component.

See `BRAWL_SIM_BUILD_PLAN.md` Step 2 / Notice 4 for the source-of-truth pass/block table and
why there's no separate vision column (the camera is bird's-eye; only bush-hiding restricts
sight, computed in `bots/perception.py`, not here).

`S` and `X` are placement markers, not distinct physical tiles -- `maps/loader.py` reads their
coordinates into `spawns`/`box_spots` and treats the cell itself as passable like `FLOOR`.

## Per-map requirements

- Border ring entirely `#`.
- At least 12 `S` markers (`blank.csv` is the stated exception, at 8), at even angular
  intervals around the perimeter -- the loader sorts them by angle from map center and the
  spawner depends on that order.
- At least 16 `X` markers (`blank.csv` again the exception, at 8).
- Every unit-passable tile (`.`, `b`, `S`, `X` -- i.e. everything except `#`/`~`) forms a
  single connected region.
- No `f`. The loader refuses it; see "Fences are not a sim tile" above.

## Editing by hand

CSVs are committed as plain text specifically so they're hand-editable -- open one in any
editor, count columns/rows against the header comment below, and edit characters directly.
After editing, re-run the loader's tests (Step 6) to re-check dimensions, connectivity, and
marker counts before committing.

The ten generated maps are the exception to everything in this section: they come from the
committed `brawl_sim/maps/generate.py`, and a hand edit to one of them fails the
reproduced-by-seed test above, so edit the generator or the seed, not the CSV.

Every OTHER file here was produced by a one-off generator script (not committed -- the plan only
requires the CSVs themselves to be plain text). The original four lay a wall border, stamp
map-specific terrain (sparse wall clusters for `open`, large bush patches for `bushy`, symmetric
wall clusters + a water channel + fence segments, since turned to wall, for `walled`), place `S`/`X` rings at even
angular intervals, then flood-fill the passable region and carve a connecting path between any
disconnected components as a safety net.

`skull_creek` and `feast_or_famine` were generated differently, and the difference is worth
repeating if you add another: features are authored EXPLICITLY as a list of (position, shape) for
half the map and then stamped twice, the second time rotated 180 degrees about the centre. The
first attempt swept features procedurally and cleared full-width bands to make river crossings,
which left three bare corridors straight across the map and big empty quadrants. Point symmetry
also gets fairness for free -- every spawn's surroundings are equivalent -- without the
mirror-image feel of a pure reflection.

`scorched_stone` and `island_invasion` were built the same way, with one change worth knowing:
the symmetry group is chosen to match the reference layout instead of always being 180 degrees.
`scorched_stone` lists each feature next to its left-right mirror and then stamps the whole list
rotated 180 degrees, which is what the real map has (it reads as a mandala, and so does the
original). `island_invasion` stamps a single feature list at all four 90-degree rotations, so its
four approaches into the arena are identical up to a quarter turn. Its coastline is the one part
of either map that is not a feature list: the island is a square clipped by eight ellipses cut
along the 22.5-degree diagonals, which makes the eight bays deeper (7 tiles) than they are wide
(4) -- circular bays scalloped the coast so hard the corner alcoves fell off the island.
