# Map CSVs

Each `csv/<name>.csv` is a plain-text grid: one row per map row, comma-separated single
characters, no header. `blank.csv` is 20x20, for the `debug_tiny` preset
(`configs/presets/debug_tiny.yaml` sets `world.map_h`/`map_w` to 20). Every other map is 60x60,
matching `configs/default.yaml`. All maps one `EnvConfig` loads must share one size: a per-env
`map_id` selects among them, and the padded `MapBank` tensors assume one shape.

## The maps

| Map | Character |
|-----|-----------|
| `blank` | 20x20, almost empty. Tests only. |
| `open` | Sparse wall clusters, very little bush. Long sightlines. |
| `bushy` | ~25% bush tiles. Concealment everywhere. |
| `walled` | Dense walls, a water channel. No bush at all. Not in the rotation -- see below. |
| `skull_creek` | A creek of separate ponds on the diagonal with bush banks, rock formations inland. Modeled on the Solo Showdown map. |
| `feast_or_famine` | A thick bush field in the centre holding 10 of the 16 boxes, with bare outskirts. Modeled on the Solo Showdown map. |
| `scorched_stone` | Cluttered: ~40 small crate clusters, wall stubs everywhere, grass bands running in off every edge, and a central grass keep whose crate core is ringed by 8 of its 44 boxes. Modeled on the Solo Showdown map. |
| `island_invasion` | One island in open water: a bush ring with eight floor alcoves punched into it, an open arena in the middle, water biting into the coast between eight headlands. Modeled on the Solo Showdown map. |

`skull_creek` makes water a hazard that shapes movement, and `feast_or_famine` puts the loot
somewhere dangerous, so approaching the middle is a decision rather than a default.
`scorched_stone` is the cluttered one: no sightline runs far. `island_invasion` makes water the
boundary rather than an inland hazard, so on its coast you can be shot from a direction you
cannot walk to; at 35.9% bush it is the bushiest map in the rotation.

**Fences are not a sim tile.** None of the reference layouts has one, so
`constants.MAP_CHAR_TO_TILE` has no `f` and the loader refuses a CSV that carries it.
`Tile.FENCE` survives only because `brawl_vision`'s terrain classes and label files index it.

## The ten generated maps

Made by `brawl_sim/maps/generate.py` rather than by hand. Shares are interior (rows/cols 1..58,
so 3364 cells; a moat counts as water); every one has 16 spawns. The seed in the table passes at
the first attempt, so `generate(seed, family, symmetry=symmetry)` returns the shipped grid byte
for byte (`tests/test_maps.py::test_generated_map_is_reproduced_by_its_seed_at_the_first_attempt`).

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
- `stone_fort`: 5x5 wall forts with grass skirts, a 10x2 channel top and bottom.
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

These readings are one reader's judgment of the ASCII renders; the look at the ten renders next
to the screenshots is still open (listed in `BRAWL_SIM_DESIGN.md`). To swap one, run
`scripts/gen_maps.py --family <family> --seed <n> --preview` until a render reads right
(`--symmetry` overrides the family's default; `narrow_pass` needs `--symmetry mirror`), copy
`runs/maps/<family>_<seed>.csv` over `csv/<name>.csv`, and update the row above and
`GENERATED_MAPS` in `tests/test_maps.py`. Keep the name: the lists under "Adding a map" refer to
it.

## The five screenshot maps

Transcribed tile by tile from top-down screenshots of four Solo Showdown maps and one unnamed
screenshot (`crescent_lakes` is a placeholder name: two water crescents face each other around
the centre). Shares as above.

| Map | Source | Wall | Bush | Water | Boxes | Spawns |
|-----|--------|-----:|-----:|------:|------:|-------:|
| `hot_maze` | Hot Maze | 16.1% | 10.3% | 5.6% | 28 | 10 |
| `ghost_point` | Ghost Point | 10.3% | 15.7% | 8.3% | 16 | 10 |
| `shadow_spirits` | Shadow Spirits | 14.1% | 7.5% | 1.1% | 24 | 10 |
| `crescent_lakes` | unnamed fifth screenshot | 10.9% | 12.6% | 9.1% | 22 | 10 |
| `twisting_vines` | Twisting Vines | 9.9% | 19.6% | 5.6% | 24 | 10 |

How a screenshot becomes a grid:

- Teal grass is bush, purple ground is floor, green is water. Purple and gray walls are walls,
  and so is everything red (cacti, crates, barrels) and every gray fence or chain. Raised
  sprites are drawn about half a tile above their cell, so each is read at its base. A blue
  icon with a yellow-green face is a power cube box.
- The screenshot's play area runs to the image edge, so the sim's wall border ring covers its
  outer row. Spawns there move one tile in, and a 2-wide gap by the edge that the ring leaves
  1 wide is closed by running the solid beside it out to the border (or, where a spawn or box
  stands in the gap, by trimming that solid instead). No map has a 1-wide passage or a dead end.
- Boxes: where a screenshot shows none (`shadow_spirits`, `twisting_vines`), the generator's crate
  pass places 24 with the map's symmetry (mirror on `shadow_spirits`, point on
  `twisting_vines`), at least 3 tiles off the border.
  `crescent_lakes` showed 11; each has its 180-degree twin, 22 in all.
- **Ten spawns, not sixteen**: one per player, as in the game. The spawner needs one per entity
  (the hero plus the nine `entities.n_enemies`), so every spawn is used every episode, and
  `tests/test_map_csvs.py` holds these five to 10 instead of 12.
- **`hot_maze` breaks the first design rule below on purpose.** Its fence lines stay as long as
  the screenshot has them (up to 19 tiles), and its most enclosed cell reaches 20% of what an
  open field offers within six steps, where the generator demands 28%. If bots grind along those
  fences, breaking the longest runs is the fix.

These are hand-transcribed, not generated: edit the CSV directly.

## The fifteen maps from the screenshot families

`generate.py` has one family per screenshot map. Each lays that map's skeleton first, most of it
drawn in one quarter and reflected across both axes, then runs the usual wall, water, bush, spawn
and crate passes and every check the ten above pass: share bands, 16 spawns, no straight wall run
past 8, the pocket test, bush in 80% of the hunt cells, at most 63 waypoints, and a 2-tile margin
around every solid stamp so no passage is 1 wide. All five families default to mirror symmetry
(left-right).

| Family | Modeled on | Skeleton | Wall stamps |
|--------|------------|----------|-------------|
| `maze` | `hot_maze` | two broken wall rings, a broken bush border, a bush plus in the centre | 1-wide fence lines and stubs |
| `lake_ring` | `ghost_point` | L-shaped lakes and straight bars on a ring, bush on their inner side, a bush-ringed pond in the centre | stubs and clusters |
| `branches` | `shadow_spirits` | a broken bush border and a 2-wide water bar below the centre | branches: a trunk with a side branch and a bush crook |
| `crescent` | `crescent_lakes` | two water crescents with bush outer rims around the centre, fully bush-rimmed corner ponds | stubs and clusters |
| `vines` | `twisting_vines` | a winding bush vine down the middle and one down each side, water pockets along their edges | stubs and clusters |

Three things differ from the four original families (`standard`, `open`, `dense`,
`water_border`), all switched on by a family's `layout`, so the ten maps above still reproduce
byte for byte: water is placed before walls, since a pond needs a 9x9 hole and the stub-heavy
mixes leave none; a stamp may sit next to bush, which the skeleton lays first; and crates keep 3
tiles off the border, which otherwise counts as cover and draws crate columns along the edge.

All fifteen are mirror, 16 spawns, reproduced at the first attempt by `generate(seed, family)`:

| Map | Family | Seed | Wall | Bush | Water | Boxes |
|-----|--------|-----:|-----:|-----:|------:|------:|
| `pond_maze` | maze | 2 | 11.3% | 12.8% | 2.6% | 32 |
| `canal_maze` | maze | 3 | 11.8% | 12.8% | 2.6% | 32 |
| `picket_maze` | maze | 7 | 12.1% | 10.5% | 1.8% | 30 |
| `lagoon_ring` | lake_ring | 2 | 8.9% | 15.7% | 9.4% | 26 |
| `square_lakes` | lake_ring | 3 | 8.7% | 17.4% | 8.6% | 18 |
| `bush_halo` | lake_ring | 6 | 9.1% | 17.2% | 7.5% | 20 |
| `bramble_ponds` | branches | 1 | 12.5% | 10.4% | 2.9% | 20 |
| `bramble_bend` | branches | 4 | 11.4% | 9.3% | 2.5% | 28 |
| `bramble_stars` | branches | 6 | 11.2% | 10.6% | 3.0% | 26 |
| `moon_gate` | crescent | 3 | 8.1% | 12.7% | 9.1% | 20 |
| `half_moon` | crescent | 5 | 7.7% | 14.4% | 10.2% | 20 |
| `moon_pools` | crescent | 7 | 7.4% | 14.6% | 8.7% | 26 |
| `vine_springs` | vines | 2 | 7.4% | 21.6% | 4.5% | 20 |
| `vine_canal` | vines | 4 | 8.7% | 20.7% | 4.6% | 24 |
| `vine_hollow` | vines | 6 | 8.1% | 22.1% | 4.5% | 22 |

What each one reads as:

- `pond_maze`: twin ponds flank the bush plus in the centre; `canal_maze`: long canals either
  side of the centre, bush in the corners; `picket_maze`: round ponds at the sides, the most
  fence of the three.
- `lagoon_ring`: a bush band across the centre with two big ponds below it; `square_lakes`: the
  cleanest ring of lakes and bars, wall blocks inside it; `bush_halo`: a bush ring round the
  centre pond with ponds outside it.
- `bramble_ponds`: big ponds flank the central water bar; `bramble_bend`: L-shaped lakes near the
  centre, channels along the bottom; `bramble_stars`: channels down both sides and star-shaped
  bush clumps.
- `moon_gate`: the crescents stand upright, like `( )`, with ponds all round and a bush band at
  the bottom; `half_moon`: the crescents lie flat, with L-shaped channels; `moon_pools`: upright
  crescents and the most ponds.
- `vine_springs`: ponds at the top, thin channels; `vine_canal`: channel bars across the top,
  ponds at the bottom; `vine_hollow`: ponds low on both sides.

## What actually trains

`configs/default.yaml`'s `world.maps` lists 36 maps: every CSV except two.

- `blank` is 20x20 and loads only under the `debug_tiny` preset.
- `walled` is the only 60x60 map with no bush, so nothing on it exercises concealment. It stays
  as a fixture (`tests/test_map_csvs.py::test_walled_has_water` is the only coverage of a
  bush-free map surviving the loader) and can be pinned with `scripts/watch.py --map walled`.

A training run draws uniformly per episode from 34: `configs/train.yaml`'s
`run.env_overrides.world.maps` is the 36 minus `split_river` and `hollow_ring`, the
`eval.holdout_maps` pair. Only the second eval rollout plays those two and nothing trains on
them, so `eval/holdout_gap` (`eval/win_rate_mean` minus `eval/holdout_win_rate`) measures map
overfitting; `best_model.zip` is chosen on the 34 alone.

Changing either list does not break a checkpoint: no agent observation spec reads
`meta.map_id`, so the policy's inputs keep their shape. `map_id` does index the run's own list,
which is why `scripts/watch.py` rebuilds the env from the run's own `train.yaml` and a saved match
carries its map bank for the viewer's `--preset`. The observation spec is the real contract:
changing the spec that `run.agent_obs` names in `configs/train.yaml` (currently
`configs/agent_obs_deploy5.yaml`) mismatches every checkpoint trained on the old one.

## Adding a map

A new CSV needs an entry in each of these, or it fails validation or is silently left out:

- `_KNOWN_MAP_NAMES` in `brawl_sim/config.py`; `validate` checks every `world.maps` and
  `fixed_map` name against it.
- `world.maps` in `configs/default.yaml` AND `run.env_overrides.world.maps` in
  `configs/train.yaml`. In `default.yaml` alone the map is loadable and watchable but never
  trained on, which looks like training on it until the win rates disagree.
- A `DIMENSIONS` row in `tests/test_map_csvs.py` (the CSV tests cover only the maps listed
  there), plus its `GENERATED_MAPS` or `SCREENSHOT_MAPS` list if it is one of those.
- `THIRTY_SIX_MAP_ROTATION` in `tests/test_maps.py` (and `GENERATED_MAPS` there for a generated
  map), and `TRAINING_MAPS` in `tests/test_training.py`.

## Designing a map for THIS simulator

- **No long solid barriers, and no enclosed pockets.** Bots steer; they do not path-find
  (`bots/steering.py` plus `terrain.resolve_move`'s axis-separated slide), so a creek spanning the
  map, or wedge-shaped pockets behind radial wall spurs, leaves bots grinding along an obstacle
  instead of routing around it. `skull_creek`'s water is a chain of separate ponds for this
  reason: a local hazard you slide past, never a wall.
- **Spread bush across the map, not into one blob.** `bots.hunt_cell_tiles` subsamples one hunter
  waypoint per 10x10 cell, so a map whose bush all sits in two cells gives hunters only two
  places to look. `skull_creek` yields 24 waypoints and `feast_or_famine` 30, of 36 possible
  cells; `scorched_stone` and `island_invasion` reach all 36.

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

The pass/block tables are `constants.TILE_BLOCKS_UNIT` and `constants.TILE_BLOCKS_PROJ`. There
is no vision column: the camera is bird's-eye, and only bush-hiding restricts sight, computed in
`bots/perception.py`.

`S` and `X` are placement markers, not distinct physical tiles -- `maps/loader.py` reads their
coordinates into `spawns`/`box_spots` and treats the cell itself as passable like `FLOOR`.

## Per-map requirements

- Border ring entirely `#`.
- At least 12 `S` markers (`blank.csv` is the stated exception, at 8, and the five screenshot
  maps at 10, the real maps' one per player; never fewer than the hero plus
  `entities.n_enemies`), at even angular intervals around the perimeter -- the loader sorts
  them by angle from map center and the spawner depends on that order.
- At least 16 `X` markers (`blank.csv` again the exception, at 8).
- Every unit-passable tile (`.`, `b`, `S`, `X` -- i.e. everything except `#`/`~`) forms a
  single connected region.
- No `f`. The loader refuses it; see "Fences are not a sim tile" above.

## Editing by hand

CSVs are plain text so they can be edited in any editor. Check rows and columns against the
map's `DIMENSIONS` row in `tests/test_map_csvs.py`, then re-run `tests/test_map_csvs.py` and
`tests/test_maps.py` to re-check dimensions, connectivity and marker counts.

The twenty-five generated maps are the exception: a hand edit to one fails the
reproduced-by-seed test above, so change the generator or the seed, not the CSV. The one-off
scripts that made the original eight are not kept; edit those CSVs directly.
