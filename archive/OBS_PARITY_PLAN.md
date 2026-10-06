# Observation parity: the sim shows the hero only what the camera shows

**Status: rev 3, 2026-09-24; progress 2026-09-25.** Phase 1 is built (C1–C11) and its run
finished: `mortis_deploy5` passed its offline gates and `configs/deployment.yaml` names it; the
live session is the lead's. Phase 2 is measured (Z1) and configured for the next run (Z2 adapted,
Z3); the retrain is Z4. Both §6 decisions are made (the quad; camera
offset > 2 tiles). Both phases are split into delegable chunks in `OBS_PARITY_TASKS.md` (Phase 1:
C1–C12 and P1; Phase 2: Z1–Z4), which is the file to build from; where the two disagree the tasks
file wins (its header lists the three deliberate deviations).

## 0. TL;DR

The deploy4 policy was trained on an observation the live loop cannot supply. Measured on
10 107 elite-tier decisions of the deploy4 checkpoint (`scripts/probes/obs_leak_measure.py`,
`scripts/probes/obs_quad_measure.py`):

| what the sim handed the policy, per decision | measured |
|---|---|
| at least one enemy revealed / revealed but NONE on screen | 0.956 / 0.482 |
| revealed enemy rows: total / on screen / off screen | 3.55 / 0.73 / 2.82 |
| of the on-screen rows, inside the 21×13 grid crop / above it / below / sides | 0.558 / 0.117 / 0.012 / 0.044 |
| top-12 projectile slots wasted on off-screen projectiles | 2.5 of 12 |
| `zone.active` | constant 1.0 every step; live it is 0 until gas is seen |

**Phase 1** (one retrain, `deploy5`):

1. A camera model in the sim: the measured ground window, placed around a camera that stops at
   the map edge the way the game's does.
2. Enemy reveal, history rings and projectile `in_view` limited to that window. **The urgent one.**
3. `hero.near_edge`: a 0/1 flag that the hero is off-centre on screen, from the same camera model
   in the sim and from the player box's screen offset live.
4. Enemy slots assigned the way the live tracker assigns them.
5. `zone.active` truthful (latched on the first gas on screen).
6. `configs/agent_obs_deploy5.yaml`, retrain, validate.

**Phase 2**: the zone schedule timed from recordings, then randomised ±10 %.

Dropped from rev 1: the sim-side dead-bin move mask (the deployment mask `policy.dead_bin_mask`
stays; the sim gets the edge flag instead), the projectile dropout augmentation, the per-side
zone latch.

## 1. What the audit found and what was decided

| # | finding | decision (2026-09-24) |
|---|---|---|
| 1 | `bots/perception.visibility` has no range term; the fairness mask uses it, so the hero is told about every enemy not in a bush | limit to the camera window; Phase 1 |
| 2 | zone rect exact and a function of the clock; `active` constant | `active` truthful in Phase 1; schedule timed from the game in Phase 2, ±10 % ranges |
| 3 | terrain exact under the HUD hole in the sim | not an issue: YOLO and HP reads are unmasked live, and the occupancy map is world-frame memory, so only never-seen ground is unknown |
| 4 | projectiles complete in the sim; off-screen ones consume top-12 slots | `in_view` on the camera window; ranking quirk and detector gap kept as noise |
| 5 | slot identity permanent in the sim | mirror the live tracker's promote / coast / reuse rules |
| 6 | wall-push habit; the agent has no edge signal | `hero.near_edge` flag from the camera clamp; no sim-side move mask |
| 7 | bush constants | correct as they are |
| 8 | ammo / HP / super lag | out of scope for this round |
| 9 | live tracker not window-limited either | intended: the reveal region is the whole screen, so nothing needs limiting live |

Settled elsewhere and not reopened: `hero.pos_norm` anchored at the map centre
(BRAWL_DEPLOYMENT_DESIGN.md 9.17), `meta.time_frac` as wall-clock ÷ 150 s clamped (9.6),
`zone.hero_margin_local` as the deployable zone shape (9.16), UNKNOWN terrain read as floor
(`perception/grid.py`). Procedural maps stay a separate, still open decision (§5).

## 2. The camera model

### 2.1 The window

The live detectors run on the whole frame. The game renders with a tilted perspective camera
(Terrain_Perception_Build_Plan.md Phase C: `|H[2,:2]|` three orders above the affine threshold,
confirmed by two independent annotation sets), so the ground the screen shows is a trapezoid:
28.9 tiles across the top row of the screen, 23.3 across the bottom, 18.4 tall. Pushing the
normalised viewport's corners through the homography and subtracting the hero's screen anchor
(`configs/default.yaml`: +0.09 tiles right of the viewport centre, +0.80 below) gives it relative
to the hero, in tiles:

| corner | dx | dy |
|---|---|---|
| top-left | −14.11 | −10.59 |
| top-right | +14.77 | −10.93 |
| bottom-right | +11.73 | +7.45 |
| bottom-left | −11.55 | +7.01 |

The policy's 21×13 grid is the largest hero-centred rectangle inside it and stays as it is (the
network's input shape). The window is only the *reveal* region: which enemies and projectiles
reach the entity groups at all. Grid planes keep their crop, history planes their 4-tile radius.

**Shape (§6 Q1).** Default: the quad itself, four half-plane tests. Alternative: the 21×13
rectangle, which then also has to be enforced live (drop tracker detections outside it) and
discards 23.6 % of on-screen sightings, mostly the four rows above the hero (§0 table).

### 2.2 The clamp

The game camera stops following the hero near a map edge (Terrain_Perception_Build_Plan.md
"Camera displacement is NOT hero displacement": `counted_walking` moved the player without moving
the camera; EDGAR_HERO_PLAN.md E0.1: the camera clamped mid-jump). Near an edge the hero is
therefore off-centre on screen and sees further into the map than usual. Model:

```
cam = clamp(hero, lo=(onset.west, onset.north), hi=(map_w - onset.east, map_h - onset.south))
```

`onset.<side>` is how far from that map edge the camera stops (tiles). Geometry defaults, taking
the user's "about 2 tiles of off-map background" allowance off the window's half extents:
west 12.1, east 12.8, north 8.9, south 5.5. They are asymmetric because the window is: the screen
shows 14 tiles sideways, 10.9 up and only 7.45 down. When the camera is not clamped `cam == hero`.

Everything in §2.1 is relative to `cam`, not to the hero: `rel = pos - cam`. A hero hugging the
west edge sees ~26 tiles east, as on the real screen.

### 2.3 Measuring the onsets (probe, runs alongside Step 1)

`scripts/probes/edge_probe.py` (chunk P1 in the tasks file), on the replay harness (`open_source`, the rectify plan, odometry):
per frame, the player box through `to_tiles` gives its camera-relative tile; minus the nominal
anchor that is the hero's screen offset. Print, per clip, every run where the offset exceeds
1 tile for more than 0.5 s, with its plateau value and direction, and odometry's camera motion
over the run (it should be ~0 while the hero walks). The plateau when the hero hugs an edge *is*
`onset` for that side. Clips: the 2026-09-23 session recording (13 matches), the 9-10 training
videos, `showdown_alternate_map*.mp4`, `bluestacks-example*.mp4`. If no clip reaches an edge on
some side, that side keeps its geometry default and the yaml comment says so.

**Measured 2026-09-24** (`scripts/probes/edge_probe.py` on the replay harness: deploy4 checkpoint,
mask on, detectors on the CPU; 19 emulator clips = the 2026-09-23 recordings, `9-10_new1..8`,
`bluestacks-example*`; `2026-09-23 17-30-14.mp4` is black and unreadable). A run is ≥ 0.5 s of
|offset| > 1 tile on one axis; it counts as a *clamp* when ≥ 3 of its decisions moved the camera
less than 0.2 tiles along that axis (odometry, the loop's own camera track), otherwise it is a
dash's camera lag or a tracker lag and is listed but not used.

| side | clamp runs | plateau median (tiles) | peak | geometry default | result |
|---|---|---|---|---|---|
| west | 8 | 3.34 | 10.13 | 12.1 | keeps the default (onset ≥ 10.1) |
| east | 7 | 3.42 | 7.18 | 12.8 | keeps the default (onset ≥ 7.2) |
| north | 2 | 1.39 | 2.18 | 8.9 | keeps the default (onset ≥ 2.2) |
| south | 5 | 1.33 | 2.31 | 5.5 | keeps the default (onset ≥ 2.3) |

All but three clamp runs start at the match gate: the hero spawns inside the clamp band and the
camera sits still until it walks out (16 s and 23 s in the 20-18-01 and 20-20-47 sessions, with
`near_edge` set on 65/65 and 89/90 decisions). No clip has the hero at a map edge, so a plateau is
a lower bound on that side's onset, never the onset itself; the four geometry defaults stand and no
peak contradicts one. The north and south values carry the `hero_offset` bias described next
(−0.78 tiles in y): corrected, the two north runs sit 0.5–0.7 tiles off, a hero standing still or
walking east–west rather than a clamp, and the south plateaus are 1.8–3.0 tiles. The one clear
mid-match clamp is 20-20-47 west (1.9 tiles); the three 20-42-11 south runs all start within 8 s
of spawn. The full per-run table is in the probe's output; rerun it on any new recording with
`--out`.

**A bias in `hero_offset` (C7, the lead's call).** The signed y offset sits at −0.78 tiles while
the hero simply walks (`bluestacks-example3.mp4`, 110 decisions: median −0.78,
IQR −0.82 to −0.75; x −0.15), the size of `brawl_vision.camera.HERO_ANCHOR_TILES[1]` (0.80).
`EntityTracker._nominal_tile` adds that constant to the viewport centre's tile, but the player
box's tile comes out of `to_tiles` at `detector.anchor_frac` (0.30 of the box height), a point
the constant does not describe; the sim side (`camera:` block, `tests/test_sim_camera.py` parity)
uses the same constant, so the sim hero and the live box disagree by 0.8 tiles in y. With
`camera.edge_flag_tiles 2.0` the live flag fires at 1.2 tiles of true northward offset and 2.8
southward. In the 19-clip sweep every flag the loop set outside a clamp (8 decisions in 3 runs)
was northward. Fix, live side only: keep `HERO_ANCHOR_TILES`, which is right for what it measures
(the green ground ring under the player) and from which the sim's `camera.quad` is derived, and
give `_nominal_tile` the player box's own nominal point: a second measured offset of about
(−0.15, −0.78) tiles from the ring, pinned by a replay test on recorded footage rather than by the
constant itself. Changing `HERO_ANCHOR_TILES` instead would move the sim's reveal window by
0.8 tiles and fail `tests/test_sim_camera.py`. The trainer never imports the tracker, so none of
this touches the running deploy5 run.

### 2.4 Config and code

`configs/default.yaml`:

```yaml
camera:                     # measured, Terrain_Perception_Build_Plan.md Phase C/K; tiles relative
  quad: [[-14.11, -10.59], [14.77, -10.93], [11.73, 7.45], [-11.55, 7.01]]   # to the camera's
  clamp_onset: {west: 12.1, east: 12.8, north: 8.9, south: 5.5}   # nominal hero position (§2.2)
  edge_flag_tiles: 2.0       # hero.near_edge = 1 when |hero - cam| exceeds this on either axis
slots:                      # EntityTracker's defaults, counted in decisions (Step 5)
  promote_hits: 2
  max_misses: 3
```

`brawl_sim/core/camera.py` (new; `core/` must not import `bots/`, and observation, history, zone
and env all need these):

- `camera_centre(hero_pos, cfg) -> (N, 2)`: the clamp above (`cfg.map_w`, `cfg.map_h`).
- `in_camera(rel, cfg) -> bool`: convex-quad test, vectorised over any leading shape. If §6 Q1
  picks the rectangle, the same function tests `|dx| <= 10.5, |dy| <= 6.5` and nothing else changes.
- `hero_view(state, vis, cfg) -> (N, E) bool`:
  `vis[:, 0, :] & in_camera(state.ent_pos - camera_centre(hero)[:, None], cfg)`. `visibility()`
  itself is untouched: it is the concealment rule, and bots and combat read it. Bots keep
  `sight_tiles: 14`.

A test in `tests/test_sim_camera.py` recomputes the quad corners from
`brawl_vision.camera.load_camera_model()` and the yaml's hero anchor and asserts each is within
0.05 tiles of the config, so the sim cannot drift from the calibration deployment uses.

**Assumption A1.** The quad was measured on the phone recordings' camera model. BlueStacks
capture is normalised to the same 2002×1126 viewport, so the same quad should hold; the probe in
§2.3 confirms it on one BlueStacks clip (the hero's offset must sit near 0 mid-map).

## 3. Phase 1 steps (execution order)

### Step 1 — camera model; enemy reveal and history limited to it (sim)

*Files:* `brawl_sim/core/camera.py` (new), `brawl_sim/bots/perception.py` (docstrings),
`brawl_sim/core/observation.py`, `brawl_sim/core/history.py`, `brawl_sim/env.py`,
`brawl_sim/config.py`, `configs/default.yaml`.

1. §2.4 config and functions.
2. `observation.build_obs` line 268: `revealed_to_hero = hero_view`, not `vis[:, _HERO, :]`. This
   changes `entities.revealed_to_hero` (the fairness mask), the `enemy_revealed` grid plane
   (line 202) and `hidden_by_bush`. `entities.in_view` (line 372) moves onto the window in Step 2
   with the other `in_view` fields; nothing in the deploy specs reads it.
3. `env._build_observation` caches `self._obs_hero_view` (N, E) instead of the (N, E, E) matrix,
   and `history.push` takes that row: `seen = ent_alive & hero_view`. The rings then hold only
   sightings a camera could have made, which is what the `enemy_hist` planes claim to be.
4. `n_enemies_alive` unchanged: the HUD counter is on screen.

*Tests* (`tests/test_observation.py`, `tests/test_history.py`): an enemy 12 tiles east is
revealed in `entities` and absent from the `enemy_revealed` plane (inside the window, outside the
crop); 16 tiles east is not revealed; 9 tiles south is not revealed while 9 tiles north is; an
enemy in a bush 1.5 tiles away is revealed and one 3 tiles away in a bush is not; with the hero
2 tiles from the west edge an enemy 20 tiles east is revealed (clamped camera); a sighting outside
the window is not pushed into the history.

*Acceptance.* `scripts/probes/obs_leak_measure.py` re-run with the new predicate: "revealed but none on
screen" is 0 by construction; mean revealed rows per decision drop from 3.55 to about 0.73.

### Step 2 — projectiles: `in_view` on the window (sim)

`observation.build_obs` line 431: `projectiles.in_view = in_camera(prj_pos - cam)`; same for
`boxes.in_view` and `pickups.in_view` (lines 445, 458) for consistency. The top-K ranking stays as
it is (chosen before the mask), by decision: the empty rows it sometimes leaves are noise in the
same direction as the live detector's misses (held-out AP50 0.20), and the user wants that noise
kept. *Test:* a projectile 20 tiles away with `time_to_closest = 0.1` produces a zero row.

### Step 3 — `zone.active` as a latch (sim)

`state.zone_seen` (N,) bool: set on any tick where the zone has shrunk at least once and the
window's bounding box around `cam` (x −14.1..+14.8, y −10.9..+7.5) crosses the rect edge, i.e.
gas is on screen; never cleared until reset. `build_obs` reports it as `zone.active`. That is
`ZoneEstimator.active` ("has any gas been observed") on the sim side. `hero_margin_local` stays
exact (three of its four horizons lie inside the window; Phase 2's randomised schedule removes
the timing leak). *Tests:* 0 before the first shrink and while the edge is beyond the window; 1
once an edge enters it; still 1 after the hero walks away.

### Step 4 — `hero.near_edge` (sim + deployment)

*Sim.* `full_obs["hero"]["near_edge"] = (|hero - cam|.amax(-1) > cfg.camera.edge_flag_tiles)`.
Schema row `("hero.near_edge", ("N",), "bool", "bool", None, ...)` like `hero.in_zone`; deploy5's
`self` group lists it (27 floats). With the geometry defaults and the 2-tile threshold it fires
about 10 tiles from the west and east edges, 7 from the north edge and 3.5 from the south: when
the screen visibly shows it, which is why the threshold is on the offset and not on "5 tiles from
the edge" (§6 Q2).

*Deployment.* `EntityTracker.update` already has the player box's camera-relative tile before it
adds odometry (`tracker.py` line 177-182). It subtracts the nominal hero tile (the centre of
`plan.viewport` through the same plan's `M` and `rect_to_tile`, plus
`brawl_vision.camera.HERO_ANCHOR_TILES` = (+0.09, +0.80)) and exposes `TrackerResult.hero_offset`
(tiles, or the last value while the hero box coasts, `None` before the first sighting).
`Assembler._put_self` writes `near_edge = max(|offset|) > cfg.camera_edge_flag_tiles`, 0 when
`None`; the threshold is the sim config the assembler already holds, so `configs/deployment.yaml`
gets no camera keys and there is one source of truth. Nothing else in the loop changes; the
tracker stays un-gated.

*Tests.* Sim: hero at (30, 30) → 0 and `cam == hero`; at (3, 30) → 1, `cam.x == 12.1`; at (30, 57)
→ 1, `cam.y == 54.5`; at (11.5, 30) → offset 0.6 → 0; at (10.5, 30) → 1. Deployment: a player box
1.5 tiles left of the anchor → 1; 0.5 → 0; no box → previous value; the nominal anchor recomputed
from the homography within 0.05 tiles.

### Step 5 — tracker-style enemy slots (sim)

*Why.* Live, an enemy gets a slot after two consecutive detections, keeps it through three missed
decisions, loses it on the fourth, and on return takes the lowest free slot, which may be a
different one (`tracker.py`: `promote_hits=2`, `max_misses=3`, `_free_slot`). In the sim slot k is
entity k forever, so the policy could learn that "slot 3 is the one I hurt earlier".

*State* (`brawl_sim/core/state.py`, new `_SLOT_FIELDS` table, zeroed by `zero_`):

| field | shape | meaning |
|---|---|---|
| `slot_ent` | (N, K) i64 | entity index + 1 held by slot k; 0 = empty. K = E − 1 = 9 |
| `ent_slot` | (N, E) i64 | slot + 1 of entity e; 0 = none |
| `ent_hits` | (N, E) i32 | consecutive decisions seen while unslotted |
| `ent_misses` | (N, E) i32 | consecutive decisions unseen while slotted |

Stored +1 so a freshly reset row (all zero) means "no slots", which `zero_` gives for free.

*Update* (`brawl_sim/core/slots.py`, `update(state, hero_view, cfg)`, called from
`env._build_observation` before `build_obs`, once per decision: live, the tracker updates on the
current decision's detections and the assembler reads the slots after that update):

1. `seen = hero_view & ent_alive`, hero column False.
2. Slotted: `misses = where(seen, 0, misses + 1)`; `misses > 3` frees the slot. A dead brawler is
   simply unseen and coasts out, as a box that stops appearing does live.
3. Unslotted: `hits = where(seen, hits + 1, 0)`; `hits >= 2` means consecutive.
4. Promotion: `need = unslotted & (hits >= 2)`, `free = slot_ent == 0`; rank needing entities by
   index and free slots by index (`cumsum - 1`), entity e takes slot k where the ranks match: an
   (N, K, E) boolean and an argmax. Entities beyond the free count stay pending with hits intact.

*Observation.* `full_obs["slots"] = {"entity": slot_ent, "valid": slot_ent > 0}`. In `obs_select`
a per-entity group whose yaml says `slots: tracked` gathers its rows by `slot_ent - 2` (entity
e ≥ 1 sits at row e − 1 once the hero row is dropped; stored +1), multiplies by `valid`, and
gathers the fairness mask by the same index. A coasting row is zeroed by the fairness mask because
`revealed_to_hero` is 0 for it, exactly as `_put_entities` + fairness does live.

*Deployment.* `_put_entities` already writes track k into entity index k + 1, so the assembler
supplies the identity permutation. No tracker change.

*Constants.* `slots.promote_hits: 2`, `slots.max_misses: 3` in `configs/default.yaml`; a test in
`tests/test_deployment_tracker.py` asserts they equal `EntityTracker`'s defaults.

*Tests.* Hand-stepped: seen once → no row; twice → slot 0; two promoted together → slots 0 and 1
in entity order; away 3 → slot held, rows zero; away 4 → freed; return → two decisions later,
lowest free slot; ten candidates for nine slots → one pending. Parity: the same sighting sequence
fed to `EntityTracker` as synthetic detections gives the same slot table.

*Accepted difference.* The sim reports an exact `rel_vel` on the promotion tick; the tracker a
finite difference.

### Step 6 — `deploy5`: spec, retrain, validation

1. `configs/agent_obs_deploy5.yaml` = deploy4 + `hero.near_edge` in `self` + `slots: tracked` on
   `enemies`; `docs/AGENT_OBS_DEPLOY5.md`; `scripts/dump_obs_schema.py` output checked in.
2. Retrain with `scripts/train.py` on the deploy4 recipe (GPU is shared: check for a running
   `train.py` first).
3. Gates, in order: the acceptance measurements of Steps 1-3 on the new checkpoint; the deploy4
   wall-push protocol (TierEvaluator, elite, 96 episodes, wall-push fraction and win rate, training
   and holdout maps) at or better than deploy4 + mask; the 2026-09-23 stuck clip replayed through
   the live path picks a legal bin; a live BlueStacks session (the user runs it).
4. BRAWL_DEPLOYMENT_DESIGN.md gets a §9 entry for the window, the clamp and the flag.

## 4. Phase 2 — zone schedule timed from the game

**Status:** measured 2026-09-24 (Z1); configured 2026-09-25 for the next run at 1.3× the game's
pace in a 185 s episode, as a run setting in `configs/train.yaml` rather than in
`configs/default.yaml` (Z2, adapted), with the ±10 % overlay; retrain pending (Z4).

*Today.* `zone.start_fraction 0.08` (12 s), `step_seconds 1.5`, `tiles_per_step 1`, all fixed
scalars, so the rect is a pure function of `time_frac` and the map size.

1. **Measure.** `scripts/probes/zone_probe.py` on the replay harness (`open_source`, odometry,
   `detect_zone`, `GasMap`): per second the gassed-cell count and the *advance events*, fresh gas
   on cells the sticky world-frame map had already seen clear. That distinction is what
   BRAWL_DEPLOYMENT_DESIGN §9 entry 14 said a moving camera could not make; the `seen` map makes it.
   Outputs: first-gas time after the match gate (the loop's own clock, so the constant lands in
   the deployed `time_frac` frame), step period, advance depth per step. Clips:
   `zone_grows_from_east.mp4` (absolute lattice), `bluestacks-example-zone.mp4` (gas from ≈21 s
   after the gate), the 2026-09-23 session recording.
2. **Set and randomise.** `zone_step_seconds` and `zone_tiles_per_step` are per-env parameters
   already (`brawl_sim/config.py` lines 545-546) and `zone_start_time` derives per env from
   `start_fraction` (line 685), so ranges need no code: `start_fraction: {low: 0.9x, high: 1.1x}`,
   `step_seconds: {low: 0.9y, high: 1.1y}` in `configs/default.yaml`. Config validation must still
   refuse `dt >= step_seconds` at the low end. *Superseded in one detail by chunk Z2 of
   `OBS_PARITY_TASKS.md`: the measured scalars stay in `default.yaml` and the ±10 % ranges go into
   `configs/randomization.yaml` (multiplicative), because `TierEvaluator` deliberately skips the
   overlay and `tests/test_zone.py` reads the base value as a scalar.*
3. Retrain or fine-tune from the Phase 1 checkpoint; the Phase 1 gates apply (chunk Z4; Z1–Z3
   are the probe, the config and the docs pass).

**Measured 2026-09-24** (Z1; `scripts/probes/zone_probe.py`, CPU only, on the loop's own
`GasMap` and odometry). Clock origin = the loop's gate (`enter_samples` 6 at 12 Hz, so about
0.5 s after the HUD ring holds steady); on the OBS recordings `scan_gameplay` mis-gates or finds
no span, so the gate came from the replay harness (`--t0-frame`). The probe forgets the gas map at
the gate (the lobby UI reads as gas), then tracks the *front per line*: for every row (east and
west sides) and column (north and south) observed in a frame, the innermost gassed cell; a
counted step is that front moving inward by ≥ 1 cell onto a cell seen clear within the last
second, so a pan that reveals gas already there cannot step. Cells are tiles.

*T0, first gas after the gate* (8 matches; the sim starts its gas at 12 s):

| match | gate (clip s) | first gas after gate | evidence |
|---|---|---|---|
| `bluestacks-example-zone.mp4` | 2.70 | 18.1 s | E, W and N lines all appear 18.2–19.0 s |
| `2026-09-23 20-18-01.mp4` | 14.50 | 19.0–19.5 s | west border on screen from spawn (clamp), zero gas through 18 s, first cells in second 19 (clip ends 19.5 s) |
| `2026-09-23 20-20-47.mp4` | 14.90 | 19.0 s | west border on screen from spawn (clamp), zero gas through 18 s, 4 cells in second 19 |
| `2026-09-23 17-29-12.mp4` | 15.27 | 18–19 s | 3 static cells at 11 s never grow (a false positive); real growth from second 18 |
| `2026-09-23 17-30-59.mp4` | 15.03 | 19.2 s | S then W lines; 3 E cells at the gate frame are the transition, not gas |
| `2026-09-23 17-32-09.mp4` | 14.27 | 19.7 s | E lines (the border was not on screen earlier) |
| `2026-09-23 20-42-11.mp4` | 12.60 | 19.2 s | S, E and W lines |
| `9-10_new5.mp4` | 7.10 | 18–19 s | 1 cell in second 18, 9 in second 19 (N and W) |

**T0 = 19 s after the gate** (median 19.0, range 18.1–19.7). The two clamp clips bracket it
without any assumption about what was on screen: the border was in view the whole time, and
nothing was gassed until second 19.

*Advance rate per side* (well-observed east/west fronts: ≥ 10 lines seen ≥ 5 s and ≥ 14 counted
steps; north/south fronts are biased low because tiles under the HUD are never observed, so
their visible front is the mask edge as often as the gas; the sim moves 0.667 tiles/s per side):

| match | side | lines | steps sharp / raw | sharp rate (tiles/s) | raw rate | bands (+1 tile) |
|---|---|---|---|---|---|---|
| `zone_grows_from_east.mp4` (phone) | E | 18 | 45 / 58 | 0.179 | 0.195 | 7, P_band 2.25 s |
| `bluestacks-example-zone.mp4` | W | 16 | 39 / 51 | 0.180 | 0.182 | 7, P_band 3.87 s |
| `2026-09-23 20-20-47.mp4` (camera static) | W | 13 | 28 / 28 | 0.169 | 0.169 | 3, starts 26.2, 34.2, 39.8 s |
| `2026-09-23 17-32-09.mp4` | E | 23 | 34 / 47 | 0.153 | 0.153 | 3 (two are odometry jumps of +11) |
| `2026-09-23 17-32-09.mp4` | W | 14 | 32 / 32 | 0.118 | −0.001 (pans) | 0 |
| `2026-09-23 17-29-12.mp4` | E | 11 | 15 / 17 | 0.144 | 0.144 | 1 |
| `2026-09-23 17-34-14.mp4` | E | 24 | 110 / 211 | 0.140 | 0.238 | 13, P_band 2.80 s |
| `2026-09-23 17-34-14.mp4` | W | 16 | 17 / 32 | 0.094 | 0.090 | 1 |
| `9-10_new4.mp4` | E | 10 | 14 / 14 | 0.116 | 0.116 | 1 |

**Rate = 0.14 tiles/s per side** (median of the nine fronts 0.144, IQR 0.118–0.169; the two
cleanest clips give 0.18), so **one tile every 6–7 s per side**. The one clip whose camera never
moved (20-20-47, spawned in the west clamp band) shows the same thing directly: after the first
ring at 19 s its gassed-cell count jumps at 26, 34 and 39–40 s, intervals of 7, 8 and 5 s.

**D = 1 tile.** Every counted line step is +1; the band medians are 1.0 in every clip that has
bands. The +4, +6 and +11 bands (17-32-09 E at 78.7 and 81.1 s, 9-10_new5 N at 38.7 s) coincide
with odometry jumps and are excluded.

**P as a burst period is not measurable from these clips, and does not need to be.** The burst
and band starts come 2–4 s apart with 3–12 lines each: the detector sees a soft edge, so the same
one-tile advance crosses threshold on different lines at different times and one real step
prints as several partial bands (the Z1 spec's `P` from bursts is superseded by the per-line
rate). Nothing drifts: the raw and sharp rates agree wherever the camera watched a front for
40 s or more (17-32-09 E over 80 s), so the sim's single-period model stands, with P = 1/rate.

*Usability.* `2026-09-23 20-27-44.mp4` no sharp bursts (three reveals only); `9-10_new2.mp4` 7 s,
no gas; `9-10_new1/3/6/8` mid-match with one or two lines each, bound nothing; `9-10_new7.mp4` S
front 0.170 over 8 s (consistent, too short to table); `2026-09-23 17-30-14.mp4` black;
`showdown_alternate_map.mp4` 0 fresh cells (the false-positive check passes).

*What Z2 takes from this* (nothing in `configs/default.yaml` changes while the deploy5 run
trains; the trainer re-reads that file at every evaluation): `start_fraction = 19 / 150 ≈ 0.127`
(today 0.08, so the sim's gas starts 7 s early), `step_seconds ≈ 6.5–7` with `tiles_per_step 1`
(today 1.5 s, 4.4× too fast; the ±10 % overlay then spans about 6–7.7 s, inside the measured
IQR). The consequence for Z4 is large: today the gas closes the whole map by about 60 s into a
150 s episode, whereas at the measured rate it moves 19–20 tiles per side by the end, so the
end-game pressure the deploy4 and deploy5 policies learned from is mostly gone and the
fine-tune has something real to learn. Measured afterwards (2026-09-24, deploy4, 64 elite
episodes): at this pace a 150 s episode truncates 30 % of matches before the end-game, where no
win can be paid, and a 240 s episode truncates none. So Z2 also needs a 240 s episode, set in the
new run's recipe rather than in `configs/default.yaml` (old runs name that file by path and would
silently get a 240 s live `time_frac`); with it, today's `start_fraction 0.08` already gives
19.2 s. Table and knock-ons in `OBS_PARITY_TASKS.md`, pending decision 3. The lead's call
(2026-09-25): train at 1.3× the game's pace in a 185 s episode instead, a little shorter than a
real match (5.0 s per tile, first gas at 14.8 s); the episode-length check at that pace is in the
tasks file's Status block.

## 5. Not in this plan

- Sim-side dead-bin move mask: dropped; `policy.dead_bin_mask` stays live (design doc §6.16).
- Projectile detection-dropout augmentation: no; the noise is wanted as is.
- Per-side zone "never looked there" latch: no.
- Procedural / augmented maps (memorisation, win 0.53 training vs 0.37 holdout): still open,
  separate decision; not blocked by anything here.
- Ammo / HP / super observable lag: later.

## 6. Decisions (closed 2026-09-24)

- **Q1 window shape: the quad.** The alternative, the 21×13 rectangle, would have to be enforced
  live too and discards 23.6 % of on-screen sightings, mostly the rows above the hero.
- **Q2 edge flag semantics: camera offset > 2 tiles** (`camera.edge_flag_tiles: 2.0`; the user
  chose 2 over 1 "to be extra safe"). Readable on all four sides; fires at ~10 / 10 / 7 / 3.5 tiles
  from W / E / N / S. The alternative, "within 5 tiles of any edge", needs a per-side threshold
  live and is unreadable for the south edge, where the hero is only ~0.5 tiles off-centre at 5
  tiles.

## 7. Files touched

| area | files |
|---|---|
| sim config | `configs/default.yaml` (`camera:`, `slots:`), `brawl_sim/config.py` |
| sim camera | new `brawl_sim/core/camera.py` (`camera_centre`, `in_camera`, `hero_view`, `window_bbox`); `bots/perception.py` docstrings only |
| sim obs | `brawl_sim/core/observation.py`, `obs_schema.py`, `obs_select.py`, `history.py`, `state.py`, `zone.py`, new `slots.py`, `env.py` |
| deployment | `brawl_deployment/perception/tracker.py` (`hero_offset`), `assemble.py` (`near_edge`, identity slots), `loop.py` (pass-through, telemetry columns); `brawl_vision/camera.py` (`HERO_ANCHOR_TILES`) |
| specs | `configs/agent_obs_deploy5.yaml`, `docs/AGENT_OBS_DEPLOY5.md`, BRAWL_DEPLOYMENT_DESIGN.md §9 |
| tests | new `test_sim_camera.py`, new `test_slots.py`, `test_config.py`, `test_observation.py`, `test_history.py`, `test_zone.py`, `test_obs_select.py`, `test_obs_schema.py`, `test_deployment_tracker.py`, `test_deployment_assemble.py`, `test_deployment_loop.py`, `test_configs_files.py` |
| probes (`scripts/probes/`) | `obs_leak_measure.py`, `obs_quad_measure.py` (in the repo since 2026-09-24), `edge_probe.py` (§2.3, new), `zone_probe.py` (§4, Phase 2) |
