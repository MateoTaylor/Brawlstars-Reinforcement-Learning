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
| `walled` | Dense walls, a water channel, fences. No bush at all. Not in the rotation -- see below. |
| `skull_creek` | A creek of separate ponds on the diagonal with bush banks, rock formations inland. Modeled on the Solo Showdown map. |
| `feast_or_famine` | A thick bush field in the centre holding 10 of the 16 boxes, with bare, fenced outskirts. Modeled on the Solo Showdown map. |
| `scorched_stone` | Cluttered: ~40 small crate clusters, barrel (fence) litter everywhere, grass bands running in off every edge, and a central grass keep whose crate core is ringed by 8 of its 44 boxes. Modeled on the Solo Showdown map. |
| `island_invasion` | One island in open water: a 36%-bush ring with eight floor alcoves punched into it, an open arena in the middle, water biting into the coast between eight headlands. Modeled on the Solo Showdown map. |

`skull_creek` and `feast_or_famine` were added to fill two gaps: water as a hazard that actually
shapes movement (only `walled` used it), and a map where the loot is concentrated somewhere
dangerous, so approaching the middle is a real decision rather than a default.

`scorched_stone` and `island_invasion` fill two more. Fences were a footnote before -- `walled`
had a few segments and nothing else used them at all -- so `scorched_stone` makes them a real
terrain type: 152 tiles of it, mostly in 1-3 tile clumps, cover that stops a body but not a
shot. And `island_invasion` is the first map where water is the BOUNDARY rather than an inland
hazard, which makes its coast the one place on any map where you can be shot from a direction
you cannot walk to. It is also the bushiest map in the rotation at 35.9%, against `bushy`'s 24%.

## What actually trains

`configs/default.yaml`'s `world.maps` is the training rotation, and it is NOT every file in
`csv/`. Two maps are excluded on purpose:

- `blank` is 20x20 and only loadable under the `debug_tiny` preset.
- `walled` is the only 60x60 map with no bush at all, so nothing on it exercises concealment --
  the mechanic the rest of the rotation is built around. It is kept as a fixture
  (`tests/test_map_csvs.py::test_walled_has_water_and_fences` is the only coverage of a
  bush-free map surviving the loader) and stays pinnable via `scripts/watch.py --map walled`,
  but no training run selects it.

Everything else in the table is in the rotation, drawn uniformly per episode. Note that
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
| `f`  | FENCE   | block | pass |
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
- Every unit-passable tile (`.`, `b`, `S`, `X` -- i.e. everything except `#`/`~`/`f`) forms a
  single connected region.

## Editing by hand

CSVs are committed as plain text specifically so they're hand-editable -- open one in any
editor, count columns/rows against the header comment below, and edit characters directly.
After editing, re-run the loader's tests (Step 6) to re-check dimensions, connectivity, and
marker counts before committing.

Every file here was produced by a one-off generator script (not committed -- the plan only
requires the CSVs themselves to be plain text). The original four lay a wall border, stamp
map-specific terrain (sparse wall clusters for `open`, large bush patches for `bushy`, symmetric
wall clusters + a water channel + fence segments for `walled`), place `S`/`X` rings at even
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
