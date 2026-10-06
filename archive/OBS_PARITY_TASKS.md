# Observation parity: delegable task chunks (Phase 1 and Phase 2)

**Status 2026-09-25: Phase 1 chunks C1–C11 are BUILT and verified (the Status block below is authoritative); C12's run `mortis_deploy5` FINISHED and passed its offline gates, `configs/deployment.yaml` names it, and only the live BlueStacks session is left (the lead's); the three pending decisions are resolved and built (the tracker offset is live now; the dash stop and the gas at 1.3× the game's pace train in the next run); Z1–Z3 are done and Z4 is the lead's next run; the cube economy (every crate spot filled, farming bots, `cube_pickup` 1.0) is built for that same run, "Done 2026-09-25" below.** This is the execution breakdown of both phases of
`OBS_PARITY_PLAN.md` (read its §0–§2 and §4 first for the *why*; this file is the *how*): Phase 1
is chunks C1–C12 plus the probe P1, Phase 2 (the zone schedule) is Z1–Z4 at the end. Each chunk is
sized for one agent session, states its inputs and outputs, and ends with an acceptance check
that does not depend on the delegate's judgement. Line numbers are as of 2026-09-24 and drift as
earlier chunks land, so every anchor also quotes the code to search for.

Three deviations from the plan's rev 2, all deliberate (details in the chunks):

- The camera functions live in a new `brawl_sim/core/camera.py`, not in `bots/perception.py`:
  `core/observation.py`, `core/history.py`, `core/zone.py` and `env.py` all need them and `core`
  must not import `bots`.
- The `near_edge` threshold live is the sim config's `camera.edge_flag_tiles`, which the
  assembler already holds; `configs/deployment.yaml` gets no camera keys (C7).
- The slot update runs in `env._build_observation` before `build_obs`, not in `env.step`, so a
  sighting on the current decision counts the way the live tracker's does (C8).

## Status (built 2026-09-24, one session, CPU only)

Every chunk landed as written; the deviations a later reader needs are listed per chunk.

| chunk | delivered | deviation from the text |
|---|---|---|
| C1 | `camera:` and `slots:` blocks in `configs/default.yaml`, `EnvConfig` fields and validation | none |
| C2 | `brawl_sim/core/camera.py` (clamp, quad test, `hero_view`), calibration parity in `tests/test_sim_camera.py` | none |
| C3 | `hero_view` wired into `build_obs`, the grid planes, `env` and `history.push` | the 13 × 21 grid crop stays hero-centred (only reveal uses the quad) |
| C4 | `*.in_view` on the camera window; projectile fairness mask reads it | ranking quirk kept, as decided |
| C5 | `state.zone_seen`, `zone.mark_seen`, `zone.active` latched | none |
| C6 | `hero.near_edge` in `build_obs` and the schema | none |
| C7 | `TrackerResult.hero_offset`, `assemble._near_edge`, loop telemetry `hero_offset_x/y` + `near_edge` | threshold is the sim config's `camera.edge_flag_tiles`; no deployment.yaml key |
| C8 | `state.py` `_SLOT_FIELDS`, `brawl_sim/core/slots.py`, `env._build_observation` wiring, `tests/test_slots.py`, tracker parity over a 15-decision script | promotion tie order and the assembler's HP-commit delay are not modelled (module docstring) |
| C9 | `slots: tracked` in `obs_select`, `obs["slots"]` bookkeeping + schema rows, assembler identity slot table | `slots.*` is refused in a spec's `fields` |
| C10 | `configs/agent_obs_deploy5.yaml`, `docs/AGENT_OBS_DEPLOY5.md`, pins in `tests/test_configs_files.py`, design doc §9.19, README + SKILL sentences | `scripts/dump_obs_schema.py` prints a tracked-slot note; deploy5 joined `test_deployment_grid.py`'s load-both-ways list only (its grid is deploy4's); `tests/test_deployment_shadow.py` `_NOT_SHADOWED` gained `hero.near_edge` (found by C11) |
| C11 | below | none |

**C11 results.** Full suite `pytest tests -q -m "not vision"`: `1 failed, 2492 passed, 2 skipped,
85 deselected in 1088.57s (0:18:08)`. The failure was
`test_deployment_shadow.py::test_observe_covers_every_self_field_the_deploy_spec_asks_of_it[agent_obs_deploy5.yaml]`
(the new bit is the tracker's, not the shadow's); after the exemption above that file re-ran
`66 passed`. `scripts/smoke_test.py --device cpu --n-envs 4 --steps 200`: all four config variants
passed with the invariant checks on. `scripts/benchmark.py --help` runs. `graphify update .` done.

Probes, deploy4 checkpoint rolled in the NEW sim on CPU (`64` episodes, 9 048 decisions) against the
§2 baseline (old sim, 10 107 decisions):

| measure | baseline | built |
|---|---|---|
| ≥1 enemy revealed / revealed but NONE on screen | 0.956 / 0.482 | 0.633 / 0.000 |
| revealed rows per decision: total / on screen / off screen | 3.55 / 0.73 / 2.82 | 0.78 / 0.78 / 0.00 |
| top-12 projectile slots held by off-screen projectiles | 2.5 | 2.66 (quirk kept by decision; the rows the policy sees are masked) |
| decisions where an on-screen projectile was dropped by the top-12 | 0.024 | 0.027 |
| share of on-screen sightings the 21 × 13 rectangle would discard | 23.6 % | 21.8 % |
| zone margin under the horizon while that edge is off screen | 0.170 | 0.186 |
| `zone.active` | constant 1 | latched (0 until gas was on screen) |

**Open.** C12's run `runs/mortis_deploy5-20260924-090436` finished 2026-09-25 and passed gates
1–3; `configs/deployment.yaml` named it (gate 4) until 2026-09-26, when the lead moved the target
to `runs/mortis_ppo-20260925-194025`, the run trained after it on deploy5's spec. Gate 5, a live
BlueStacks session, is the lead's (`scripts/deploy_run.py --dry-run`, then without the flag).
Results in "Done 2026-09-25" below. `configs/train.yaml` names deploy5's spec since 2026-09-25
(the mortis_ppo run trained on it).

**Done concurrently with C12 (2026-09-24, CPU only; nothing `TierEvaluator` re-reads from disk
was touched).**

*Wall-push baseline for gate 3.* `scripts/probes/wall_push_measure.py` is the §6.16 protocol as
a script (96 elite episodes per cell, deterministic, the deployed mask computed from the agent's
own `blocks_unit` plane through `legal_move_bins`) with one definition change: a wall-push is now
any chosen move bin that `terrain.resolve_move` would leave in place (sim truth over all 16
bins), where §6.16's scratch scripts counted only "moving, moved < 0.05 tiles, wall cell
0.75 tiles ahead". The deploy4 checkpoint under the NEW sim (camera-limited reveal, latched
zone, `near_edge`, tracked slots; mask = `policy.dead_bin_mask`):

| maps | mask | wall-push decisions | of which wedged | stalls ≥ 2 s / episode | longest stall | win | mean rank | decisions |
|---|---|---|---|---|---|---|---|---|
| training | off | 0.056 | 0.030 | 0.32 | 81 | 0.20 | 3.11 | 14 108 |
| training | on | 0.036 | 0.036 | 0.19 | 81 | 0.20 | 3.11 | 14 197 |
| holdout | off | 0.065 | 0.035 | 0.30 | 83 | 0.17 | 3.23 | 14 572 |
| holdout | on | 0.031 | 0.031 | 0.22 | 83 | 0.15 | 3.21 | 14 626 |

A stall is ≥ 8 consecutive wall-push decisions (2 s at 4 Hz); the SE of a win rate near 0.2 over
96 episodes is ≈ 0.04. Deploy5 must land at or better than the "on" rows under the same sim.
The mask removes every wall-push on free ground (0.000 in both "on" cells); what remains is a
sim trap the mask cannot see:

| maps | mask | wedged decisions | stuck (all 16 bins dead) | wedge onsets | onsets within 1 decision of a dash | mask ≠ sim truth |
|---|---|---|---|---|---|---|
| training | off | 0.032 | 0.022 | 41 | 41 | 0.032 |
| training | on | 0.039 | 0.023 | 40 | 40 | 0.039 |
| holdout | off | 0.041 | 0.028 | 65 | 65 | 0.041 |
| holdout | on | 0.036 | 0.022 | 57 | 57 | 0.036 |

*The dash wedge (resolved 2026-09-25: a dash now stops at the wall, "Done 2026-09-25" below).* `hero.start_dash`
marches only the CENTRE against `blocks_unit` and backs off by `los_step_tiles + unit_radius`
along the dash direction alone, so a dash along or beside a wall face lands the 0.4-tile body
inside the wall (hero local x 10.81 against a wall at column 11 was one case). From there
`resolve_move` rejects every bin, the "one-way trap" hero.py's own comment (lines 337–355)
describes for the head-on case, until the next dash. 100 % of wedge onsets follow a dash; 3–4 %
of decisions are wedged, 2.2–2.8 % have no legal bin at all, and the longest stalls (81–83
decisions ≈ 20 s) are all wedges. The deployed `legal_move_bins` shrink rule (written for the
real game, where the map is an estimate) calls those bins legal, so no mask fixes it, and it caps
what any retrain can show on this metric. The fix proposed here (resolve each dash tick
through `resolve_move`, so a dash slides along a wall) was not taken: the lead chose to have a
wall stop a dash's momentum rather than redirect it. §6.16's 0.000 with the mask on was the narrower definition, not a clean sim.

*Stuck-clip replay through the live path.* `scripts/probes/replay_clip.py` feeds a recording
through the real `DeployLoop` (`ClipCapture` in place of `Capture`, `NullBackend` controls, the
loop's own gate; the stall and stop logic patched to record instead of exit; detectors on the
CPU). `2026-09-23 20-20-47.mp4`, deploy4: the gate opens at 14.8 s; in the 22–32 s window the
unmasked policy picks east on 37/41 decisions (§6.16 counted 40/45 live); with the mask on it is
vetoed on 22/41 there and 49/171 over the clip, the bin changes on 63, and it never picks a dead
bin. The wall east of the hero reads as blocked on only about half of those decisions (the CV
position jitters across the 0.4-tile margin), so live the agent alternates E/ESE rather than
stopping; gate 3's replay of deploy5 should therefore count legal bins per decision, not give one
verdict.

Run lines (all CPU, safe beside a training run):

```
.venv/Scripts/python.exe scripts/probes/wall_push_measure.py --run runs/<run> --episodes 96 --json <out.jsonl>
.venv/Scripts/python.exe scripts/probes/replay_clip.py "C:/Users/mateo/Videos/2026-09-23 20-20-47.mp4" --mask both --window 22 32
.venv/Scripts/python.exe scripts/probes/zone_probe.py <clip> --hud emulator --out <report.md>
.venv/Scripts/python.exe scripts/probes/edge_probe.py <clips or globs> --out <report.md>
```

*P1, measured (table and method in `OBS_PARITY_PLAN.md` §2.3).* 19 emulator clips through the
replay harness; a clamp run is ≥ 0.5 s of |offset| > 1 tile with the camera still along that
axis. Plateaus: west 3.3 (peak 10.1), east 3.4 (peak 7.2), north 1.4 (peak 2.2), south 1.3
(peak 2.3) tiles as measured; the y bias below moves north and south by 0.78, so corrected the
two north runs are 0.5–0.7 tiles (not clamps) and the south ones 1.8–3.0. No clip reaches a map
edge, so these are lower bounds and all four geometry
defaults (12.1 / 12.8 / 8.9 / 5.5) stand. Nearly every clamp starts at spawn: the hero appears
inside the band and the camera holds until it walks out (16–23 s in the 20-18-01 and 20-20-47
sessions, `near_edge` set throughout). One finding for C7: the live `hero_offset` carries a
constant y bias of −0.78 tiles (110 walking decisions, IQR −0.82..−0.75), the size of
`HERO_ANCHOR_TILES[1]`, because `_nominal_tile` adds that constant to the viewport centre while
the player box's tile is taken at `detector.anchor_frac`; with `camera.edge_flag_tiles 2.0` the
flag fires at 1.2 tiles north and 2.8 tiles south of true. Every flag the loop set outside a
clamp in the sweep (8 decisions, 3 runs) was northward.

*Z1, measured (tables in `OBS_PARITY_PLAN.md` §4).* Eight matches. **T0 = 19 s after the loop's
gate** (18.1–19.7; the two clips that spawned in the west clamp band had the border on screen
from spawn and no gas through second 18). **D = 1 tile.** **Rate 0.14 tiles/s per side** (nine
well-observed east/west fronts, IQR 0.12–0.17; the cleanest two give 0.18), i.e. one tile every
6–7 s; the static-camera clip's cell-count jumps come 7, 8 and 5 s apart. Today's sim: 12 s and
0.667 tiles/s, so its gas starts 7 s early and closes 4.4× too fast; it has shut the map by about
60 s into a 150 s episode, which the game never does. `P` as a burst period is not measurable
(a soft edge turns one advance into several partial bands 2–4 s apart) and the per-line rate
replaces it; nothing drifts, so the single-period model stands. `zone_probe.py` now forgets the
gas map at the gate, measures per-line fronts (sharp and raw rates, band steps) and takes
`--t0-frame` from the harness's gate because `scan_gameplay` mis-gates the OBS recordings.

**Pending decisions (the lead's; all three resolved 2026-09-25, results in "Done 2026-09-25" below).**

1. **Resolved: stop at the wall, not slide; built.** The dash wedge. Proposed, not taken:
   resolve dash ticks through `resolve_move` for the run after deploy5, then
   re-run `wall_push_measure.py` (above). Mechanism confirmed by an angle sweep on the fixture of
   `tests/test_hero.py::test_no_dash_approach_angle_can_leave_an_entity_stuck_against_a_wall`
   (wall column at x = 30, 300 start offsets per angle, CPU): share of wall-meeting dashes that end
   with the body inside the wall, uncharged / charged, is 0 / 0 % head-on, 2 / 3 % at 75°,
   12 / 12 % at 60°, 28 / 29 % at 45°, 57 / 67 % at 30°, 76 / 87 % at 20°, 100 / 100 % at 10°.
   That test sweeps only the head-on direction despite its name; the fix's regression test is the
   same sweep over angles, asserting zero at every angle.
2. **Resolved as proposed; built.** The `hero_offset` y bias: give `_nominal_tile` the player box's own nominal point, a second,
   live-only measured offset of about (−0.15, −0.78) tiles from the ring, pinned by a replay test
   on recorded footage. Do NOT change `HERO_ANCHOR_TILES`: it is the green ground ring's position
   and the sim's `camera.quad` is derived from it (`tests/test_sim_camera.py`), so changing it
   would move the sim's reveal window by 0.8 tiles. Live side only; the trainer never imports the
   tracker, so it is safe during the run. Timing: deploy5 is the first spec that reads
   `hero.near_edge` (deploy4's does not), so the fix must land before deploy5's first live session.
3. **Resolved: 1.3× the game's pace in a 185 s episode; built for the next run.** Z2's numbers (`start_fraction ≈ 0.127`, `step_seconds ≈ 6.5–7`, `tiles_per_step 1`, ±10 %
   overlay in `configs/randomization.yaml`): applied after the run finishes, never while
   `TierEvaluator` re-reads `configs/default.yaml`.

   **Episode length (measured 2026-09-24):** deploy4's `best_model.zip`, 64 elite episodes per
   row through `TierEvaluator` with the schedule passed as env overrides, CPU, nothing on disk
   edited.

   | schedule | wins | mean rank | episode s, median / mean | hit the time cap |
   |---|---|---|---|---|
   | today: gas at 12 s, 1.5 s per tile, 150 s episode | 0.20 | 3.36 | 40 / 35 | 0 % |
   | measured: gas at 19 s, 6.5 s per tile, 150 s episode | 0.12 | 3.17 | 122 / 93 | 30 % |
   | measured: gas at 19.2 s, 6.5 s per tile, 240 s episode | 0.25 | 2.72 | 124 / 104 | 0 % |

   At the game's pace the gas reaches 2×2 at about 200 s, so a 150 s episode truncates 30 % of
   matches before the end-game, and a truncated match can never pay `win_bonus`. Recommendation:
   pair the schedule with a 240 s episode (`sim.max_episode_steps: 4800`), where today's
   `start_fraction 0.08` already gives 19.2 s and only `step_seconds` changes. Put the episode
   length in the NEW run's recipe (`--set sim.max_episode_steps=4800`, recorded in its
   `train.yaml`), never in `configs/default.yaml`: runs name `default.yaml` by path, and
   `DeployedPolicy` rebuilds the live `time_frac` denominator from it plus the run's overrides, so
   editing the default would silently turn deploy4's and deploy5's live 150 s into 240 s. The rule
   of design doc §9 entry 6 (wall-clock over the sim's episode, clamped) is unchanged; the new run
   carries its own denominator. Knock-ons for Z4: about a third as many matches per million steps
   (its budget was sized on 40 s episodes); a fine-tune from deploy5 relearns what each
   `time_frac` value means; the zone damage ramp is per shrink step, so each stage of the close
   is as lethal as today, only later (Z1 did not measure damage; leave it); re-baseline deploy5
   under the new sim before judging a fine-tune, since deploy4's own numbers move above from the
   schedule alone. Caveat: deploy4 never trained on slow gas or a 240 s denominator, so the wins
   and rank columns are indicative; the 30 % truncation is mechanical, and a policy that survives
   longer only raises it.

**Done 2026-09-25 (CPU only; nothing trains).**

*C12's gates.* The run finished (`best_model.zip` 01:44, `final_model.zip` 02:47). Every gate is on
`best_model.zip`, the checkpoint `configs/deployment.yaml` names.

| gate | result |
|---|---|
| 1. C3's leak numbers | ≥1 enemy revealed 0.627, revealed but none on screen 0.000; revealed rows per decision, total / on screen / off screen, 0.78 / 0.78 / 0.00; top-12 projectile slots held by off-screen projectiles 2.56 (quirk kept); on-screen projectile dropped by the top-12 0.024; zone margin under the horizon while that edge is off screen 0.178 |
| 2. the wall-push protocol against deploy4 + mask, same sim | better on the training maps, level within noise on holdout (table below) |
| 3. the 2026-09-23 stuck clip through the live path | east picked on 0/41 decisions in the 22–32 s window (deploy4: 37/41); the mask vetoes 3/171 over the clip (deploy4: 49/171); no dead-bin pick; the bin changes on 45 |
| 4. `configs/deployment.yaml` `run.dir` | deploy5, `best_model.zip` |
| 5. a live BlueStacks session | the lead's |

Gate 2, both checkpoints with the mask on, 96 elite episodes per cell, before the dash fix below:

| maps | run | wall-push decisions | stalls ≥ 2 s / episode | longest stall | win | mean rank |
|---|---|---|---|---|---|---|
| training | deploy4 | 0.036 | 0.19 | 81 | 0.20 | 3.11 |
| training | deploy5 | 0.022 | 0.12 | 32 | 0.30 | 2.79 |
| holdout | deploy4 | 0.031 | 0.22 | 83 | 0.15 | 3.21 |
| holdout | deploy5 | 0.030 | 0.23 | 27 | 0.14 | 3.38 |

Every deploy5 wall-push with the mask on was a dash wedge (48–67 onsets per cell), which the
next item removes.

*Pending decision 1, built: a dash stops where the body first touches a wall.* The lead's call:
a wall stops a dash's momentum rather than redirecting it, so a glancing dash keeps its line and
stops short, with no slide. `terrain.body_travel` tests the whole body (`circle_blocked`, the test
walking uses) every `los_step_tiles` along the line, bisects 4 times (`_BODY_REFINE_STEPS`) inside
the step where it first fails, and backs off `_WALL_CLEARANCE` only at a contact; `hero.start_dash`
calls it with `params.dash_ray_tiles` as the ray budget. A body that starts overlapping stays where
it is rather than tunnelling out. Tests: four `body_travel` cases in `tests/test_terrain.py`;
`tests/test_hero.py::test_a_glancing_dash_stops_on_its_line_at_the_wall_instead_of_sliding` and
`::test_a_dash_alongside_a_wall_with_the_body_clear_is_not_shortened`. The wall-push protocol on
deploy5 under the fixed sim:

| maps | mask | wall-push decisions | of which wedged | stalls ≥ 2 s / episode | longest stall | win | mean rank | decisions |
|---|---|---|---|---|---|---|---|---|
| training | off | 0.026 | 0.000 | 0.12 | 26 | 0.36 | 2.72 | 14 696 |
| training | on | 0.000 | 0.000 | 0.00 | 0 | 0.31 | 2.90 | 14 556 |
| holdout | off | 0.034 | 0.000 | 0.17 | 33 | 0.15 | 3.40 | 14 037 |
| holdout | on | 0.000 | 0.000 | 0.00 | 0 | 0.17 | 3.17 | 14 542 |

Stuck (all 16 bins dead) 0.000 and mask ≠ sim truth 0.000 in every cell; wedge onsets 0, 0, 2 and
1, against 48–67 per cell before. A sweep over the real map bank (16 maps × 512 random clear
starts × 16 bins × both dash lengths, 262 144 dashes) lands no body inside a wall; 433 (0.17 %)
clip a corner between two samples mid-dash and land clear, which is where the three one-decision
onsets come from (a decision boundary fell mid-dash). Left as is: brief and harmless, and closing
it would mean sampling every dash more finely. This sim change reaches only the next run; deploy5
trained under the old dash.

*Pending decision 2, built: the tracker's player-box offset.* `tracker.PLAYER_BOX_FROM_RING_TILES
= (−0.08, −0.78)`, added in `_nominal_tile`, measured through the live path on five BlueStacks
recordings (1 113 single-box sightings while tracking; per-clip medians −0.15 to −0.04 in x,
−0.92 to −0.71 in y). `HERO_ANCHOR_TILES` is unchanged. Pinned by
`tests/test_deployment_tracker.py::test_recorded_player_boxes_read_near_zero_offset_while_tracking`
on `tests/fixtures/vision/player_boxes_tracking.json`. Live only, so deploy5 gets it now. Found
with it: `tests/test_deployment_policy.py`'s end-to-end checkpoint test had never met a spec that
reads `hero.near_edge` (it skipped until deploy5's checkpoint existed) and now passes
`hero_offset` the way the loop does.

*Pending decision 3, built for the next run: the gas at 1.3× the game's pace.* The lead's call:
1.3× Z1's pace, in an episode a little shorter than a real match. `configs/train.yaml`
`run.env_overrides` sets `sim.max_episode_steps: 3700` (185 s, 740 decisions) and
`zone.step_seconds: 5.0` (6.5 s ÷ 1.3); the existing `start_fraction 0.08` then puts the first gas
at 14.8 s (19 s ÷ 1.3 is 14.6). `run.randomization` names `configs/randomization.yaml`, which now
ships exactly the two ±10 % lines (first gas 13.3–16.3 s, 4.5–5.5 s per tile, redrawn per env at
every reset); evaluation stays at the nominal schedule. Pins:
`tests/test_configs_files.py::test_randomization_file_ships_only_the_gas_jitter` and
`::test_the_shipped_run_trains_the_gas_at_1_3x_the_games_pace`,
`tests/test_zone.py::test_the_run_jitters_each_envs_gas_schedule_and_redraws_it_at_reset`. Design
doc §9 entry 20; the plan's §4 status line.

Two deviations from Z2 as written. The schedule is a run setting in `configs/train.yaml`, not new
scalars in `configs/default.yaml`: runs name `default.yaml` by path, so the reason item 3 above
gives for the episode length holds for the whole schedule, and every run trained before keeps its
150 s. `check_zone_ranges` is not built: the shipped draws (4.5 s at the least, first gas at 16.3 s
at the latest) sit far inside `validate`'s bounds, so it would guard only a hand-edited range.

Episode length at 1.3× (deploy5 `best_model.zip`, 64 elite episodes per row through
`TierEvaluator` with the schedule passed as overrides, CPU):

| schedule | wins | mean rank | episode s, median / mean | hit the time cap | longest |
|---|---|---|---|---|---|
| as trained: gas at 12 s, 1.5 s per tile, 150 s episode | 0.36 | 2.42 | 44.6 / 38.9 | 0 % | 55.2 s |
| 1.3× nominal: gas at 14.8 s, 5.0 s per tile, 185 s episode | 0.19 | 3.33 | 81.2 / 76.3 | 0 % | 155.8 s |
| 1.3× slow end: gas at 16.3 s, 5.5 s per tile, 185 s episode | 0.20 | 3.41 | 79.8 / 82.1 | 0 % | 171.2 s |

No episode reaches the cap even at the slow end of the jitter, 14 s to spare. Episodes run about
twice as long as deploy5 trained on, so a run gets about half as many matches per million steps.
deploy5 never trained on slow gas, so its wins here are indicative only.

*Next run (Z4).* Everything above is in `configs/train.yaml` as the lead left it; only the spec is
passed, as for deploy5:

```
.venv/Scripts/python.exe scripts/train.py --set run.agent_obs=configs/agent_obs_deploy5.yaml
```

Add `--set run.name=<name>` to name it. A fine-tune adds `--resume
runs/mortis_deploy5-20260924-090436/best_model.zip`, which builds the new recipe and loads the
weights; it relearns what each `time_frac` value means, since the denominator moves from 150 s to
185 s.

*Z3.* Design doc entry 20 and the plan's §4 status line are written; §10 item 9 stays as written
(no burst period was measurable). Schema proof, rendered in memory because
`scripts/dump_obs_schema.py` always rewrites `docs/OBSERVATION.md`: `docs/OBSERVATION.md` and
`docs/AGENT_OBS_DEPLOY5.md` are identical to the committed files. Suite
`pytest tests -q -m "not vision"`: `4 failed, 2499 passed, 2 skipped, 85 deselected in 1486.94s
(0:24:46)`. Three were `tests/test_deployment_grid.py::test_the_grid_matches_the_sim_channel_for_channel`
for deploy, deploy3 and deploy4, "episode ended mid-comparison" with no plane mismatched: the dash
stop moved that test's seeded, crowded run into two bot rifles firing point-blank, and the hero
died at decision 19 of 40. The test now zeroes the bots' damage after the reset, so survival no
longer rides on the seed; the file re-ran `57 passed`. The fourth,
`tests/test_training.py::test_shipped_stage_walk_is_the_plan_walk`, pins
`curriculum.max_timesteps_at_stage` at 75M against the lead's local `configs/train.yaml` retune
(100M per stage, 300M total), and is left for the lead: deploy5 force-advanced at the cap on all
four transitions (`curriculum.json`: win rates 0.25–0.30 against the 0.35 gates), so at 100M per
stage in 300M the next run would end as `veteran` finishes and never train against the expert or
elite stages. Run as deploy5 did (75M, 450M), it spent its last 150M at elite.

*Cube economy, built for the next run (the lead's request of 2026-09-25: the agent loses live to
7+ cube bots and never builds cubes early).* Measured first, deploy5 `best_model.zip` at elite
under the 1.3× gas, 64 episodes per cell: with the 8 crates deploy5 trained on, the richest bot
held 2.6 cubes at 60 s and the hero's killer 1.0, and the hero collected 32 % of every cube picked
up; the sim had the pressure backwards. Built: `configs/default.yaml` fills every marked crate spot
(`n_boxes 48`, `max_boxes 48`, `max_pickups 64`; 16 to 44 crates per map, 24.7 on average, where
real maps carry 20 to 30); `brawl_sim/bots/policy.py` pulls bots to crates at 20 tiles × 3.0
unless an enemy target is within 8, and to cubes at 12 tiles × 4.0 with no enemy gate, both gated
on a clear straight walk (`_walk_clear`, a `terrain.march` on `blocks_unit`, because
`resolve_move` has no pathfinding: about a tenth of mobile bots' crate-pulled decisions stalled
against a wall without it; the 41 % / 17 % first quoted here mostly counted HUNT_BUSH wall
stalls, corrected later that day) and on the loot being a tile out of the gas; `targeting` lets
a bot shoot a crate while its enemy is out of attack range; `bots/personality.py` drops the crate
pull while a cube pull is active; `configs/train.yaml` `reward.cube_pickup` 0.5 → 1.0. Result at
elite: richest bot 7.4 at 60 s, 7+ in 48 % of matches, killer's cubes 4.3, hero share 16 %, deploy5
win 0.17; at hard: richest bot 7.3, killer's cubes 8.9, win 0.56 (rank 1.67 → 1.00). Granting bots
cubes on a clock (the lead's fallback) was not needed and is not built. Tables in design doc §9
entry 21; probe `scripts/probes/cube_economy_measure.py`. Found on the way: `tests/test_boxes.py`'s
fixture passed `n_boxes` to `cfg` but not to the spec `build_params` reads, so its crate-count
assertions had been passing off `default.yaml`'s own 8; fixed to merge the overrides into both.
Suite: the 33 files that touch bots, crates, configs, rewards, the grid and the env ran green
(900 passed) apart from the stage-cap pin above; the full suite was not re-run end to end that
day because the machine was under load from another program (a 10 000-step autoreset test was
crawling at 0.4 s per step, against about 10 ms unloaded).

**The pre-run audit, the same evening** (design doc §9 entry 22; the lead decided on eight
findings). Built: `brawl_sim/bots/policy.py` and `personality.py`, a loot pull is a unit direction
that REPLACES the mode steering instead of summing with it (`steering.seek`/`flee` return raw
`target - pos` and `combine` normalises once, so the summed 3.0 × 20-tile pull weighed weight ×
distance and froze bots at weighted midpoints: campers parked at crates 27 % of their time, engaged
bots were cube-pulled on 27 % of decisions); both pulls need no enemy target within 8 tiles (a
cube within 2 is grabbed regardless), are off in RETREAT and HOLD_STILL, and a crate is a fire
target only within `min(attack_range, 5)` tiles (the lead: nearby crates, not every crate in
range). `core/hero.py`: the dash grants no i-frames (the lead: an animation during which Mortis
can be hit); `hero.invuln` stays in the spec, always False, `perception/shadow.py` never seeds it,
docs/OBSERVATION.md regenerated. `core/movement.py`: speed scales with the intent's length, so the
EMA-smoothed bot intent decelerates instead of coasting at full speed along its old heading (47 %
of HOLD_STILL ticks moved at hard). `core/combat.py`: the zone can no longer overwrite the
finisher's last-hit on an entity already at 0 HP, so the hero keeps a kill it finishes in the gas.
Settled without a change: crates stay walkable (correct to the game, the lead), the curriculum
keeps its forced advance with `advance_win_rate` 0.25 (the lead's own edit, the 75M pin still
theirs), and the three zone-observation mismatches (pre-shrink `hero_margin_local` to the map edge
vs +10 live, `zone_grid` off-map cells gas from tick 0 vs zeros until seen, one negated side vs
four in the gas) are recorded in the entry only. Measured at elite, 32 episodes: campers in bush
0.60 → 0.86, parked at a crate 0.27 → 0.00, engaged bots cube-pulled 0.27 → 0.03, retreat
decisions moving toward the enemy → 0.00, gas-stolen hero kills 1 → 0, hero combat damage inside
i-frames 0.40 → 0 (1.14 reward per episode had been charged for it); deploy5's own win 0.25 →
0.06, expected, since it dashed through damage that now lands. Found on the way: C4's kept quirk
(the sim ranks an off-screen projectile into a slot and blanks it; the tracker never holds one)
had let `tests/test_deployment_assemble.py`'s parity tests pass by trajectory luck; the fixture now
feeds only on-screen projectiles and the tests skip the projectile group in a frame where the sim
holds an off-screen one, the sim unchanged. Probes in the repo: `scripts/probes/bot_pull_measure.py`
and `kill_credit_measure.py`. Suite on the final code: 2509 passed, 2 skipped, 1 failed (the
stage-cap pin above), 33 min with three CPU probes running beside it.

---

## 0. Rules for every delegate

Paste these into each delegate's prompt; they are repo conventions, not suggestions.

1. **Orientation first.** Run `graphify query "<question>"` before grepping or reading source
   (the repo hook insists on it). `graphify path "<A>" "<B>"` and `graphify explain "<X>"` for
   relationships and single concepts. After editing repo code run `graphify update .`.
2. **Python is not on PATH.** Always `.venv/Scripts/python.exe`. Tests:
   `.venv/Scripts/python.exe -m pytest tests/test_x.py -q`. The whole suite is ~20 min; run only
   the files the chunk names unless the chunk says otherwise.
3. **CPU only.** The one GPU is shared with training runs. Never start a training run, never
   load a detector on CUDA, never call `scripts/train.py`.
4. **No git.** Never commit, stage, branch or suggest doing so. The report lists files changed.
5. **Do not touch** `configs/agent_obs_deploy4.yaml` or earlier deploy specs, anything under
   `runs/`, or `configs/deployment.yaml` (except where C7 says). Runs name their spec by path;
   widening an old spec silently changes what an old checkpoint means.
6. **Frozen dataclass config.** `brawl_sim/config.py` `EnvConfig` is frozen; fields are plain
   Python values (tuples, not lists); `load_config(path, overrides)` keeps the dataclass default
   for an absent yaml key; `validate(cfg, params)` is the only place that raises on values.
7. **No host syncs in per-step code.** Inside anything that runs per tick or per decision never
   write `torch.tensor([...], device=cuda_device)`: build constant tensors once (module-level
   cache keyed by device/dtype/values, the way `obs_select._cached_tensor` does) or use
   `core/geometry.vec2`. Read `brawl_sim/core/geometry.py` lines 13–20 first.
8. **Every leaf of `full_obs` needs a schema row.** `brawl_sim/core/obs_schema.py` `_ROWS` is
   `(name, shape, dtype, units, range, description, privileged, conditional)`;
   `validate_obs` raises on a missing or extra field, and `tests/test_obs_schema.py` set-compares
   `build_obs` against `obs_spec(cfg)`. `_resolve_shape` (line 253) maps shape letters to cfg.
9. **Docs are generated.** Never hand-edit `docs/OBSERVATION.md` or `docs/AGENT_OBS*.md`; rerun
   `.venv/Scripts/python.exe scripts/dump_obs_schema.py [--spec configs/agent_obs_deployN.yaml]`.
10. **Test scaffolding.** `tests/bot_fixtures.py` (`FakeBank`, `grid`, `cfg_and_params`,
    `fresh_state`), `configs/presets/debug_tiny.yaml` (20×20 map `blank`, 2 enemies, view 14×10,
    zone off, 300 steps). On the tiny map the camera is pinned mid-map horizontally (see C2), so
    tests that need an unclamped camera use the 60×60 map with no bush,
    `overrides={"world": {"maps": ["walled"]}}` (`brawl_sim/maps/csv/walled.csv`), and place
    entities by hand after reset, the way `tests/test_observation.py` lines 195–294 do.
11. **Docstrings say why.** Match the codebase: a paragraph on the reason, no essays, no
    marketing. Comments that restate the code are removed in review.
12. **Report format** at the end of the session: files changed (one line each), tests run with
    the pass/fail counts pasted verbatim, measurements as a table, anything left undone and why.
    No next-steps list that includes committing.

## 1. Chunk map

| chunk | what | depends on | can run in parallel with |
|---|---|---|---|
| C1 | `camera:` and `slots:` config blocks, `EnvConfig` fields, validation | — | P1 |
| C2 | `core/camera.py`: clamp, quad test, `hero_view`; calibration parity test | C1 | P1 |
| C3 | wire `hero_view` into `build_obs`, the grid, `env`, `history.push`; leak re-measure | C2 | — |
| C4 | `in_view` of entities / projectiles / boxes / pickups on the camera window | C3 | C5, C6 |
| C5 | `zone.active` latch (`state.zone_seen`, `zone.mark_seen`) | C3 | C4, C6 |
| C6 | `hero.near_edge` in the sim | C3 | C4, C5 |
| C7 | `hero.near_edge` live: tracker offset, assembler, loop, telemetry | C6 | C8 |
| C8 | tracker-style slots: state fields, `core/slots.py`, env wiring, parity test | C3 | C7 |
| C9 | `slots: tracked` in `obs_select`; assembler identity slots | C8 | — |
| C10 | `agent_obs_deploy5.yaml`, docs, test pins, design-doc entry | C4–C9 | — |
| C11 | verification pass: full suite, probes, graph update | C10 | — |
| C12 | retrain and gates (**lead, not a delegate**) | C11 | — |
| P1 | probe: measure the camera clamp onsets from footage | — | C1–C9 |
| Z1 | probe: time the zone schedule from footage (`scripts/probes/zone_probe.py`) | — | C1–C11, P1 |
| Z2 | measured schedule in `default.yaml`, ±10 % overlay, range-aware validation, tests | Z1, C1, C5 | C7–C10 |
| Z3 | design-doc entry 9.20, plan status, docs diff check, full-suite pass | Z2 | C11 (one suite run covers both) |
| Z4 | fine-tune or retrain under the jittered schedule and gates (**lead, not a delegate**) | C12, Z3 | — |

Suggested batches: {C1, P1} → {C2} → {C3} → {C4, C5, C6} → {C7, C8} → {C9} → {C10} → {C11}.
Phase 2 threads through them: Z1 alongside {C1, P1}; Z2 once Z1 has numbers and C5 has landed
(in or after the {C7, C8} batch); Z3 with C11; Z4 after C12. If Z2 lands before C12 starts, C12's
run takes the overlay flag and Z4 folds into it (see Z4).

## 2. Facts every chunk relies on

- **Camera window** (measured, `Terrain_Perception_Build_Plan.md` Phase C/K; recomputed today
  from `brawl_vision.camera.load_camera_model()` and matching to 0.01 tiles): ground quad in
  tiles relative to the hero's nominal screen anchor, corners **TL, TR, BR, BL** (y down):
  `(-14.11, -10.59), (14.77, -10.93), (11.73, 7.45), (-11.55, 7.01)`. The hero's anchor is the
  viewport centre + `(+0.09, +0.80)` tiles (`configs/default.yaml` lines 37–39). The 21×13 grid
  crop is unchanged; only *reveal* uses the quad.
- **Camera clamp:** `cam = clamp(hero, lo=(west, north), hi=(map_w - east, map_h - south))`
  with geometry defaults `west 12.1, east 12.8, north 8.9, south 5.5` until P1 measures them.
  Where `hi < lo` on an axis (map narrower than the two onsets, e.g. the 20-wide test map:
  `20 - 12.8 = 7.2 < 12.1`) the camera sits at `(lo + hi) / 2` on that axis.
- **Edge flag:** `hero.near_edge = max(|hero - cam|) > 2.0` tiles (user's choice, 2026-09-24).
  With the defaults it fires when the hero is within ~10.1 tiles of the west edge, ~10.8 of the
  east, ~6.9 of the north and ~3.5 of the south.
- **Sim path today:** `bots/perception.visibility(state, bank, params, cfg) -> (N,E,E)` is
  bush-only concealment with no range term (`brawl_sim/bots/perception.py` line 40; bots add
  `bots_sight_tiles` in `bot_visibility`, line 59). `core/observation.build_obs(state, bank, vis,
  raw_los, params, cfg)` (line 234) takes the hero row at line 268; `_build_grid` (line 170)
  takes it again at line 199 for the `enemy_revealed` / `enemy_hidden` planes; `env.py` caches
  `self._obs_vis` (lines 213, 256–258, 311) for `history.push(state, action, vis)` (line 23).
- **Live path:** `brawl_deployment/perception/tracker.py` `EntityTracker.update` (line 155):
  `placed = to_tiles(detections, plan, ...)` (line 178) is camera-relative tiles *before*
  odometry is added (line 180). `TrackerResult(hero, enemies, status, segment, n_detections)`
  (line 86). `ObservationAssembler._put_self` (`assemble.py` line 297) writes `full["hero"]`;
  `_put_entities` (line 373) writes tracker slot k at entities index k+1; `_put_projectiles`
  sets `in_view` True for every tracked projectile (line 438). `DeployLoop` builds the tracker
  at `loop.py` line 491 and calls `assemble` at line 801. `ZoneEstimator.active` (
  `perception/zone.py` line 133) already latches on any gas seen. **The tracker updates once
  per 4 Hz decision**, inside `DeployLoop._decide` (line 763), not per perception tick, so its
  `promote_hits=2` / `max_misses=3` are counted in decisions.
- **Probes** live in `scripts/probes/` (copied from the audit's scratchpad on 2026-09-24):
  `obs_leak_measure.py` and `obs_quad_measure.py` roll the deploy4 checkpoint on CPU.
- **Measured baseline** (`scripts/probes/obs_leak_measure.py`, deploy4 checkpoint, elite tier,
  10 107 decisions): revealed rows per decision 3.55, of which 0.73 on screen; "revealed but
  none on screen" 0.482 of decisions. C3's acceptance is against these numbers.

---

## C1 — Config: `camera:` and `slots:` blocks

**Goal.** The camera model's constants and the slot constants exist on `EnvConfig`, load from
`configs/default.yaml`, validate, and nothing else changes.

**Files.** `configs/default.yaml` (after the `view:` line, line 53), `brawl_sim/config.py`
(`EnvConfig` line 57; `_ENV_CONFIG_FIELDS` line 221; `validate` line 842), `tests/test_config.py`.

**Yaml** (comment style: two or three lines on *why*, like the `view:` comment above it):

```yaml
camera:
  quad: [[-14.11, -10.59], [14.77, -10.93], [11.73, 7.45], [-11.55, 7.01]]
  clamp_onset: {west: 12.1, east: 12.8, north: 8.9, south: 5.5}
  edge_flag_tiles: 2.0
slots:
  promote_hits: 2
  max_misses: 3
```

The `camera` comment must say: quad = the ground the screen shows, tiles relative to the hero's
nominal screen anchor, corners TL/TR/BR/BL, re-derived from the shipped homography by
`tests/test_sim_camera.py`; `clamp_onset` = how far from each map edge the game camera stops
following the hero, geometry defaults (quad half-extent minus ~2 tiles of visible off-map
ground) until `scripts/probes/edge_probe.py` measures them; `edge_flag_tiles` = the offset that
sets `hero.near_edge`. The `slots` comment: `EntityTracker`'s defaults, counted in decisions (the tracker updates once
per decision), see C8.

**`EnvConfig` fields** (place after `view_w`; all with defaults equal to the yaml):

```python
camera_quad: tuple[tuple[float, float], ...] = ((-14.11, -10.59), (14.77, -10.93), (11.73, 7.45), (-11.55, 7.01))
camera_clamp_onset: tuple[float, float, float, float] = (12.1, 12.8, 8.9, 5.5)   # west, east, north, south
camera_edge_flag_tiles: float = 2.0
slots_promote_hits: int = 2
slots_max_misses: int = 3
```

**Loader entries** in `_ENV_CONFIG_FIELDS`, with two small transforms next to
`_tuple_transform` (line 216):

- `("camera.quad", "camera_quad", _quad_transform)`: `tuple((float(x), float(y)) for x, y in v)`;
  raise `ValueError` unless exactly 4 corners of 2 numbers.
- `("camera.clamp_onset", "camera_clamp_onset", _onset_transform)`: reads the dict, requires
  exactly the keys `west, east, north, south`, returns the 4-tuple in that order. A dict with a
  side missing raises naming the missing keys, so a hand-written fragment cannot silently zero
  the others. (An override on top of `default.yaml` is deep-merged by `load_config`, line 290,
  so overriding one side there is fine and keeps the yaml's other three.)
- `("camera.edge_flag_tiles", "camera_edge_flag_tiles", float)`,
  `("slots.promote_hits", "slots_promote_hits", int)`, `("slots.max_misses", "slots_max_misses", int)`.

**Validation** (append to `validate`, near the `history_frames` checks at line 919):

- every onset ≥ 0; `edge_flag_tiles > 0`; `slots_promote_hits ≥ 1`; `slots_max_misses ≥ 0`.
- the quad is convex, clockwise and contains the origin: for each edge `a -> b` (corners in
  order, wrapping) the cross product `(bx - ax) * (0 - ay) - (by - ay) * (0 - ax)` must be `> 0`.
  This is the exact predicate C2's `in_camera` uses, so a mis-ordered yaml fails here with a
  message instead of revealing nothing.

**Tests** (`tests/test_config.py`, follow its existing style):

- defaults load from the yaml and equal the dataclass defaults;
- `overrides={"camera": {"edge_flag_tiles": 1.0}}` changes only that field;
- `overrides={"camera": {"clamp_onset": {"west": 5}}}` on `default.yaml` gives
  `(5.0, 12.8, 8.9, 5.5)`; a fragment yaml (`tmp_path`, the style `tests/test_config.py` already
  uses) whose `camera.clamp_onset` names only `west` raises and the message names
  `east, north, south`;
- a 3-corner quad raises; a counter-clockwise quad (reverse the corner order) raises in
  `validate`; `edge_flag_tiles: 0` raises; `slots.promote_hits: 0` raises.

**Run.** `tests/test_config.py tests/test_configs_files.py` (the latter loads every shipped
preset through `validate`).

**Considerations.**

- `DeploymentConfig` (`brawl_deployment/config.py`) gets **no** camera keys. The live loop reads
  the sim config out of the checkpoint's own `train.yaml`, so the threshold on both sides is the
  same object by construction (C7 relies on this).
- `check_removed_keys` (`config.py` line 725) lists yaml keys that no longer exist; nothing to
  add there.
- `train.yaml` files under `runs/` reference `configs/default.yaml` by path, so older runs
  evaluated after this chunk pick the new fields up with their defaults. That is intended.

---

## C2 — `brawl_sim/core/camera.py`: the camera model as functions

**Goal.** Three pure functions, cached constants, one test file, and the outdated "the hero's
observation legitimately covers the whole map" claims removed from docstrings. No caller wiring
yet (that is C3).

**Files.** New `brawl_sim/core/camera.py`; new `tests/test_sim_camera.py`;
`brawl_sim/bots/perception.py` (module docstring lines 1–6 and `bot_visibility`'s docstring
lines 63–66); `brawl_vision/camera.py` (one constant).

**API.**

```python
HERO_ANCHOR_TILES = (0.09, 0.80)   # in brawl_vision/camera.py, next to PIXELS_PER_TILE (line 112)

def camera_centre(hero_pos: Tensor, cfg) -> Tensor          # (N, 2) f32 -> (N, 2)
def in_camera(rel: Tensor, cfg) -> Tensor                    # (..., 2) -> (...) bool
def hero_view(state, vis: Tensor, cfg) -> Tensor             # (N, E) bool
def window_bbox(cfg, device, dtype) -> tuple[Tensor, Tensor] # (2,), (2,): per-axis min / max of the quad
```

- `camera_centre`: `lo = (west, north)`, `hi = (map_w - east, map_h - south)`;
  `cam = torch.minimum(torch.maximum(hero, lo), hi)`; where `hi < lo` per axis substitute
  `(lo + hi) / 2`. Use `torch.where`, never a Python `if` on a tensor.
- `in_camera`: four half-planes. With `quad` (4, 2) in TL, TR, BR, BL order and
  `edge = roll(quad, -1, 0) - quad`, `d = rel.unsqueeze(-2) - quad` gives (..., 4, 2) and
  `cross = edge[:, 0] * d[..., 1] - edge[:, 1] * d[..., 0]`; inside is `(cross >= 0).all(-1)`.
  Boundary inclusive. This is the same convention `scripts/probes/obs_quad_measure.py` used for
  the 23.6 % measurement, so numbers stay comparable.
- `hero_view`: `vis[:, 0, :] & in_camera(state.ent_pos - camera_centre(state.ent_pos[:, 0], cfg).unsqueeze(1), cfg)`.
  Column 0 (the hero itself) is left as `vis` gives it; consumers that must exclude it already do
  (`history.push` sets `seen[:, 0] = False`).
- Constants: one module-level dict keyed by `(str(device), dtype, cfg.camera_quad,
  cfg.camera_clamp_onset, cfg.map_w, cfg.map_h)` holding the quad, `lo`, `hi` and the bbox as
  tensors, built once per key. The per-call path does no tensor construction (rule 7).

**Docstrings to change.** `bots/perception.py` line 3 ("since the camera is a fixed bird's-eye")
and lines 63–66 in `bot_visibility` ("the camera is a fixed bird's-eye view, so the hero's
observation legitimately covers the whole map"). New truth, in one sentence each: `visibility()`
is the concealment rule and stays whole-map because bots and combat read it; the hero's
observation is `core/camera.hero_view` = concealment AND the camera window; bots keep
`sight_tiles`.

**Tests** (`tests/test_sim_camera.py`; use `load_config("configs/default.yaml", overrides=...)`
and `tests/bot_fixtures.py`):

1. *Calibration parity.* Recompute the quad from the shipped model and assert each coordinate is
   within 0.05 tiles of `cfg.camera_quad`:
   ```python
   m = load_camera_model(); w, h = m.viewport
   corners = m.px_to_tile(np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64))
   anchor = m.px_to_tile(np.array([[w / 2, h / 2]], np.float64))[0] + np.array(HERO_ANCHOR_TILES)
   # corners - anchor == cfg.camera_quad to 0.05
   ```
   (Verified 2026-09-24: the four rows come out `-14.11 -10.59 / 14.77 -10.93 / 11.73 7.45 /
   -11.55 7.01` exactly.)
2. *Clamp* on 60×60: hero (30, 30) → cam == hero; (3, 30) → cam.x == 12.1; (58, 30) → 47.2;
   (30, 2) → cam.y == 8.9; (30, 58) → 54.5. On 20×20: cam.x == 9.65 for any hero.x; cam.y clamps
   normally to [8.9, 14.5].
3. *Quad membership* (hand-checked against the edges): inside `(0, 0)`, `(12, 0)`, `(0, -9)`,
   `(0, 7)`, `(-13, -10)`; outside `(16, 0)`, `(0, 9)`, `(14, 6)`, `(-15, 0)`. Also a (N, P, 2)
   batch shape round-trips to (N, P).
4. *hero_view* on a bush-free 60×60 map (`grid(60, 60)` + `FakeBank`), hero at (30, 30):
   enemy (42, 30) True; (46, 30) False; (30, 21) True; (30, 39) False; an enemy in a BUSH tile
   1.5 tiles away True, 3 tiles away False (`bush_reveal_radius` 2.0 comes through `vis`);
   hero at (2, 30) with an enemy at (22, 30) True (the clamped camera sees 20 tiles east);
   a dead hero sees nothing.

**Run.** `tests/test_sim_camera.py tests/test_perception.py tests/test_vision_camera.py -m "not vision"`.

**Considerations.**

- `state.ent_pos` is float32; keep every constant in the same dtype and device as `hero_pos`.
- `test_vision_camera.py` line 443 already computes the same corners for the 21×13 derivation;
  do not duplicate its logic, just call the model.
- The `brawl_vision` import in the test is fine (tests already import both packages); the
  `brawl_sim` package itself must not import `brawl_vision`.

---

## C3 — Wire `hero_view` into the observation, the grid, the env and the history

**Goal.** Everything the policy is told about enemies comes from `hero_view`. After this chunk
"revealed but none on screen" is 0 by construction.

**Files.** `brawl_sim/core/observation.py`, `brawl_sim/core/history.py`, `brawl_sim/env.py`,
`tests/test_observation.py`, `tests/test_history.py`, `scripts/probes/obs_leak_measure.py`.

**Edits.**

1. `build_obs(state, bank, vis, raw_los, params, cfg, hero_view=None)`: when `None`, compute
   `camera.hero_view(state, vis, cfg)` (keeps every existing caller and test working under the
   new semantics). Line 268 `revealed_to_hero = vis[:, _HERO, :]` → `= hero_view`. Line 271
   `hidden_by_bush = in_bush_all & ~revealed_to_hero` stays (an off-screen bush enemy reads as
   hidden, which is what a camera would say). `hero_revealed_to` (line 269) stays on `vis`: it is
   what the bots see, not what the hero sees.
2. `_build_grid(state, bank, hero_view, cfg, origin, out_h, out_w)`: the `vis` parameter becomes
   the (N, E) row; line 199 `revealed_to_hero = vis[:, _HERO, :]` → `= hero_view`. Both call
   sites (lines 500 and 503) pass `hero_view`.
3. `obs["visibility"]["vis"]` (line 497) stays the raw (N, E, E) matrix; its schema row already
   says it is the concealment matrix. Do not add a field.
4. `history.push(state, action, hero_view)`: rename the parameter, `seen = state.ent_alive &
   hero_view` (line 49), update the docstring line 33.
5. `env.py`: keep `self._obs_vis` (scratch probes read it) and add `self._obs_hero_view`
   (line 213). `_build_observation` (line 309): `vis → hv = camera.hero_view(...) → self._obs_vis =
   vis; self._obs_hero_view = hv → build_obs(..., hero_view=hv)`. `step` (line 256): if
   `_obs_hero_view is None`, compute both from a fresh `visibility` pass; `history.push(self.state,
   action, self._obs_hero_view)`.

**Tests.**

- `tests/test_observation.py`, new section "camera-limited reveal", built with
  `_cfg_params_bank(tiny=False, overrides={"world": {"maps": ["walled"]}, "entities": {"n_enemies": 2}})`
  (60×60, no bush) and `_reset`, positions set by hand afterwards as the file's existing tests
  do: an enemy 12 tiles east has `entities.revealed_to_hero` True and is absent from the
  `enemy_revealed` plane (channel index from `obs_select.channel_index(cfg)["enemy_revealed"]`,
  never a literal) because 12 > the crop's 10; 16 tiles east is not revealed and not
  `hidden_by_bush`; 9 north revealed, 9 south not; both `view` and `world` grids agree.
- `tests/test_history.py`: after teleporting an enemy off screen set `env._obs_hero_view = None`
  (so `step` recomputes from the new positions), step once, and assert `hist_enemy_seen[:, 0, j]`
  is False for it and True for an on-screen one. **Fix the existing test**
  `test_enemy_seen_is_alive_and_visible_and_never_the_hero_itself`, which asserts every enemy is
  seen on the tiny map with random spawns: pin the enemies next to the hero first (the tiny map's
  camera sits at x 9.65 and rows beyond ~16 can be off screen when the hero is in the north).
- Review every existing test in both files that asserts `revealed_to_hero` equals the `vis` row
  or "everyone visible"; make the placement explicit rather than weakening the assertion.

**Acceptance (measure, do not skip).** In `scripts/probes/obs_leak_measure.py` replace
`sim._obs_vis[:, 0, 1:]` with `sim._obs_hero_view[:, 1:]` and the rectangle `in_view` with
`camera.in_camera(pos - camera.camera_centre(hero, cfg).unsqueeze(1), cfg)`. Run
`.venv/Scripts/python.exe scripts/probes/obs_leak_measure.py 64` (CPU, about a minute; 64
episodes is the baseline's sample). Expected:
"revealed but none on screen" 0.000; mean revealed rows per decision ≈ 0.7 (was 3.55); share of
revealed rows off-screen 0.000. Paste the block in the report.

**Run.** `tests/test_observation.py tests/test_history.py tests/test_obs_schema.py
tests/test_obs_select.py tests/test_env.py tests/test_deployment_assemble.py`.

**Considerations.**

- `tests/test_deployment_assemble.py` runs the sim and the assembler side by side; it must still
  pass unchanged (the assembler's `revealed_to_hero` comes from `Track.seen_now`).
- `scripts/benchmark.py` and `scripts/smoke_test.py` call `build_obs` or step the env; a quick
  `--help` run of each after the change catches a broken signature.
- Compile mode (`cfg.compile`) traces `_run_tick` only; nothing in this chunk is inside it.

---

## C4 — `in_view` on the camera window

**Goal.** `entities.in_view`, `projectiles.in_view`, `boxes.in_view`, `pickups.in_view` mean "on
screen" (the camera quad around the clamped camera). The projectile fairness mask then stops
admitting off-screen projectiles. The top-K ranking is **not** changed (decision: that quirk is
kept as noise in the same direction as the live detector's misses).

**Files.** `brawl_sim/core/observation.py` (lines 273, 372, 431, 445, 458),
`brawl_sim/core/obs_schema.py` (rows at lines 140, 142, 145, 184, 195, 205),
`tests/test_observation.py`, `tests/test_obs_select.py`.

**Edits.**

- Near line 273 compute once: `cam = camera.camera_centre(hero_pos, cfg)` and
  `rel_of = lambda pos: pos - cam.unsqueeze(1)`. Replace the four `_in_window(...)` calls with
  `camera.in_camera(rel_of(state.xxx_pos), cfg)`. `_view_origin` / `_in_window` stay for the grid
  crop only.
- Schema descriptions: the four `in_view` rows → "On screen: inside the camera's ground window
  around the clamped camera (core/camera.py)". `entities.revealed_to_hero` → "On screen and not
  concealed by a bush (core/camera.hero_view)". `entities.hidden_by_bush` → "In a bush and not
  revealed; off screen counts as not revealed".
- `obs_select._build_entity_group` (line 358) is untouched.

**Tests.** With the deploy4 spec and a 60×60 env: a live projectile 20 tiles east with velocity
toward the hero (so `time_to_closest` is small) produces an all-zero row in `out["projectiles"]`;
the same projectile 5 tiles east produces a non-zero row. `boxes.in_view` / `pickups.in_view` /
`entities.in_view` True at 12 east, False at 16 east. Grep `tests/` for `in_view` and update any
test that pinned the old rectangle semantics.

**Run.** `tests/test_observation.py tests/test_obs_select.py tests/test_obs_schema.py
tests/test_deployment_assemble.py`; then `scripts/dump_obs_schema.py` (docs/OBSERVATION.md).

**Considerations.** The live assembler already sets `projectiles.in_view` True for every
tracked projectile (`assemble.py` line 438), which is the same statement ("the detector found it
on screen"); no deployment change. `_in_window` may be imported by scripts; grep before deleting
anything (do not delete it anyway).

---

## C5 — `zone.active` as a latch

**Goal.** `zone.active` is 0 until gas has been on screen at least once in the episode, then 1
for the rest of it. That is `ZoneEstimator.active` ("has any gas been observed") on the sim
side.

**Files.** `brawl_sim/core/state.py` (`_ZONE_EPISODE_FIELDS` line 112), `brawl_sim/core/zone.py`,
`brawl_sim/env.py` (`_build_observation`), `brawl_sim/core/observation.py` (line 475),
`brawl_sim/core/obs_schema.py` (line 212), tests.

**Edits.**

- State: `("zone_seen", lambda N, E, P, B, U, L, K: (N,), BOOL)` after `zone_step`. It is then
  in `_ALL_FIELD_SPECS`, zeroed on reset by `zero_`, included in `snapshot` and checked by
  `check_invariants` for free.
- `zone.mark_seen(state, cam, cfg) -> None` (MUTATES `zone_seen`; no-op when
  `cfg.zone_enabled` is False):
  ```python
  lo, hi = camera.window_bbox(cfg, device, dtype)           # (2,), (2,)
  bbox_lo, bbox_hi = cam + lo, cam + hi                     # (N, 2)
  shrunk = state.zone_step > 0
  crosses = ((bbox_lo < state.zone_lo) | (bbox_hi > state.zone_hi)).any(-1)
  state.zone_seen |= shrunk & crosses
  ```
  The bounding box, not the quad, by decision: gas is an axis-aligned rect, and the box only
  over-counts the trapezoid's two cut corners.
- `env._build_observation`: after computing `hv`, `zone.mark_seen(self.state,
  camera.camera_centre(self.state.ent_pos[:, 0], self.cfg), self.cfg)` and only then `build_obs`
  (the observation must read this decision's latch).
- `build_obs` line 475: `"active": state.zone_seen`. Schema row: "Latched: gas has been on screen
  at least once this episode (core/zone.mark_seen); always 0 with the zone disabled."

**Tests** (add to `tests/test_zone.py`), on `walled` (60×60; the zone is enabled by default,
only the tiny preset turns it off): before the first shrink `active` is False; force a shrink (`state.time` past
`zone_next_t`, then `zone.step_zone`) with the hero at (30, 30) → still False (the 1-tile gas
ring is off screen: bbox x ∈ [15.9, 44.8]); hero at (10, 30) → True (cam.x = 12.1, bbox_lo.x =
−2.0 < zone_lo.x = 1); hero back at (30, 30) → still True; a reset row → False. With
`zone.enabled: false` the field is always False.

**Run.** `tests/test_zone*.py tests/test_observation.py tests/test_obs_schema.py
tests/test_env.py tests/test_deployment_zone.py`.

**Considerations.** `step_zone` runs every tick inside the compiled `_run_tick`; the latch
runs once per decision in `_build_observation`, outside it, which is enough because the
observation is only built per decision. Live, the estimator latches on any gas in the sticky
`GasMap`, which is detected on the whole rectified frame; nothing changes there.

---

## C6 — `hero.near_edge` in the sim

**Goal.** A bool field that is 1 when the camera has stopped following the hero.

**Files.** `brawl_sim/core/observation.py` (hero dict, line 278), `brawl_sim/core/obs_schema.py`
(after `hero.in_zone`, line 94), `tests/test_observation.py`.

**Edits.** In the hero dict: `"near_edge": (hero_pos - cam).abs().amax(-1) > cfg.camera_edge_flag_tiles`
(bool, like `in_bush` / `in_zone`; the assembler's `_b` helper then serves it live). Schema row:
`("hero.near_edge", ("N",), "bool", "bool", None, "The camera has stopped following the hero: |hero - cam| > camera.edge_flag_tiles on either axis (core/camera.py). Live: the player box's offset from its nominal screen anchor.", False, None)`.

**Tests.** Hero (30, 30) → False; (3, 30) → True; (30, 57) → True (offset 2.5); (11.5, 30) →
False (offset 0.6); (9.5, 30) → True (offset 2.6); (10.5, 30) → False (offset 1.6, under the
2-tile threshold). With `overrides={"camera": {"edge_flag_tiles": 1.0}}` the last one flips.

**Run.** `tests/test_observation.py tests/test_obs_schema.py`; regenerate `docs/OBSERVATION.md`.

**Considerations.** Do **not** add the field to any existing spec yaml; C10 adds it to deploy5.
`_norm_divisor("bool", cfg)` already exists for the other bool rows.

---

## C7 — `hero.near_edge` live: tracker offset, assembler, loop, telemetry

**Goal.** The live loop supplies `hero.near_edge` from the player box's screen offset, with the
threshold read from the checkpoint's own sim config.

**Files.** `brawl_deployment/perception/tracker.py`, `brawl_deployment/perception/assemble.py`,
`brawl_deployment/loop.py` (`TickRow` line 139, tracker construction line 491, `assemble` call
line 801), `tests/test_deployment_tracker.py`, `tests/test_deployment_assemble.py`,
`tests/test_deployment_loop.py`.

**Tracker.**

- `TrackerResult` gains `hero_offset: tuple[float, float] | None = None` as the **last** field
  with a default, so the existing five-positional constructions keep working.
- No constructor change: `RectifyPlan` already carries `viewport` (`brawl_vision/camera.py`
  line 344), and the loop tests' `_Plan` stub has it too.
- Per `update`: the nominal hero tile is the centre pixel of `plan.viewport` pushed through the
  **same** `plan` as the detections (`cv2.perspectiveTransform` with `plan.M`, then
  `plan.rect_to_tile`) plus `brawl_vision.camera.HERO_ANCHOR_TILES`; the offset is the player
  box's `placed` tile (line 178, camera-relative, before odometry) minus that. Using the same plan
  for both makes any registration shift of `origin_tile` cancel. When no player box is seen this
  decision keep the previous offset (the hero track coasts); `reset()` clears it to `None`.
- Keep the odometry gate as is: the loop does not decide on a non-`ok` tick, so the offset is
  never read then.

**Assembler.** `assemble(..., hero_offset=None)`. In `__init__` record whether the loaded spec
names `hero.near_edge` (`any("hero.near_edge" in g.fields for g in spec.groups)`). In
`_put_self`: if the spec wants it, `_require(hero_offset, "hero_offset")` and write
`self._b(max(abs(ox), abs(oy)) > self.cfg.camera_edge_flag_tiles)`; otherwise write `False`
(unread, the same rule as the hero row in `_put_entities`). The threshold is the sim config's
field, never a deployment.yaml key.

**Loop.** Pass `hero_offset=tracked.hero_offset` to `assemble`. `TickRow` gains
`hero_offset_x: float = 0.0`, `hero_offset_y: float = 0.0`, `near_edge: int = -1`, appended
**after** the existing last column so older CSVs keep their column positions
(`TickRow.from_record` defaults missing columns). Set them on decision rows.

**Tests.**

- `test_deployment_tracker.py` with `StubPlan` (identity warp, 48 px/tile; give it
  `viewport = (2002, 1126)` if it lacks one): the nominal tile is the viewport centre through the
  stub's own `rect_to_tile`, plus (0.09, 0.80); a player box placed 1.5 tiles left of it →
  `hero_offset == (-1.5, 0)`; 0.5 left → `(-0.5, 0)`; no player box on the next update → the
  previous value; `reset` → `None`; before any sighting → `None`.
  One test on the real plan (`build_rectify_plan(load_camera_model(), load_hud_mask())`): a
  player box whose 0.30 anchor sits at the viewport centre gives an offset of `(-0.09, -0.80)`
  within 0.05 (it is the anchor, not the centre, that reads 0).
- `test_deployment_assemble.py`: `_suppliers_from` adds `hero_offset = hero_pos -
  camera.camera_centre(hero_pos)` read off the sim; with a temporary spec (deploy4's groups plus
  `hero.near_edge` after `hero.in_zone`, written with the `_write_spec` helper from
  `tests/test_obs_select.py`) the assembled `self` group equals the sim's byte for byte for a
  hero at (3, 30) and at (30, 30). With the deploy4 spec, `assemble` without `hero_offset` still
  works.
- `test_deployment_loop.py`: with the recording `_Assembler`, `calls[-1]["hero_offset"]` is
  the tracker's value; with `_Policy(real_assembler=True)` and a temp spec that names
  `hero.near_edge` (pattern: `test_a_decision_reaches_the_policy_through_the_real_assembler`,
  line 965), a decision goes through without a skip.

**Run.** `tests/test_deployment_tracker.py tests/test_deployment_assemble.py
tests/test_deployment_loop.py tests/test_deployment_policy.py -m "not vision"`.

**Considerations.**

- `to_tiles` is the only projection to use; do not add a second pixel-to-tile path.
- The box-to-tile placement carries sub-tile noise (`project.py` lines 61–67 on how the 0.30
  anchor was measured); the 2-tile threshold is what makes that noise irrelevant. Do not smooth.
- Nothing gates the tracker to the window; the reveal region live is the whole screen already.

---

## C8 — Tracker-style enemy slots (sim state and update rule)

**Goal.** Slot k of the enemy group is assigned the way `EntityTracker` assigns it: a slot on
promotion, held while coasting, freed on retire, lowest free slot on return. In the sim today
slot k is entity k forever.

**Files.** `brawl_sim/core/state.py`, new `brawl_sim/core/slots.py`, `brawl_sim/env.py`
(`_build_observation`), new `tests/test_slots.py`, `tests/test_deployment_tracker.py` (parity).

**Time base.** `EntityTracker.update` runs once per 4 Hz decision (`loop.py` line 763, inside
`_decide`), so its constants are already in decisions and the sim uses the same ones:
`promote_hits 2`, `max_misses 3` (C1). Live trace, which the sim must reproduce exactly: a
`Track` is created with `hits = 1` (line 74) and is pending; seen again next decision → `hits 2`
→ `_promote` gives it the lowest free slot **on that update**, and `assemble` reads the slots
after the update, so it is shown from its 2nd consecutive on-screen decision. A pending track
retires on any miss (`_retire`, "consecutive"). A slotted track last seen at decision d has
`misses` 1, 2, 3 at d+1..d+3 and is retired when `misses > max_misses`, at d+4; while it coasts
its rows are zero (`revealed_to_hero` is `seen_now`). Not modelled: the assembler's HP-commit
rule (a slot is written only once its HP has been read); note it in the module docstring.

**State** (`_SLOT_FIELDS`, added to `_ALL_FIELD_SPECS`; the lambda's `K` is history depth, so
spell the slot count as `E - 1`):

| field | shape | dtype | meaning |
|---|---|---|---|
| `slot_ent` | (N, E−1) | I64 | entity index + 1 held by slot k; 0 = empty |
| `ent_slot` | (N, E) | I64 | slot + 1 of entity e; 0 = none |
| `ent_hits` | (N, E) | I32 | consecutive decisions seen while unslotted |
| `ent_misses` | (N, E) | I32 | consecutive decisions unseen while slotted |

Stored +1 so a zeroed row (reset) means "no slots".

**Update** (`slots.update(state, hero_view, cfg) -> None`, MUTATES the four fields; fully
vectorised, no `.item()`, no loop over N or E):

1. `seen = hero_view & state.ent_alive; seen[:, 0] = False`.
2. Slotted rows: `misses = where(seen, 0, misses + 1)`; `retire = slotted & (misses > max_misses)`.
   Free their slots: `retire_slot = zeros(N, K, int32).scatter_add_(1, (ent_slot - 1).clamp(min=0),
   retire.int()) > 0`; `slot_ent = where(retire_slot, 0, slot_ent)`; `ent_slot = where(retire, 0,
   ent_slot)`. (scatter_add, not scatter, because unslotted rows all map to index 0.)
3. Unslotted rows: `hits = where(seen, hits + 1, 0)` (a pending track retires on any miss, which
   is what makes "consecutive" mean consecutive).
4. Promotion: `need = (ent_slot == 0) & (hits >= promote_hits)`; `free = slot_ent == 0`;
   `need_rank = cumsum(need, 1) - 1`; `free_rank = cumsum(free, 1) - 1`; `take = need &
   (need_rank < free.sum(1, keepdim=True))`; `match = take[:, :, None] & free[:, None, :] &
   (need_rank[:, :, None] == free_rank[:, None, :])` (N, E, K); `new_slot = match.int().argmax(2)`;
   `ent_slot = where(take, new_slot + 1, ent_slot)`; write `slot_ent` with a `scatter_add_` of
   `where(take, e_index + 1, 0)` at `new_slot` (indices are unique by construction). Entities
   beyond the free count keep their hits and stay pending. `hits = where(take, 0, hits)`.
5. Dead entities are simply unseen and coast out, as a box that stops appearing does live.

**Where it runs.** `env._build_observation`, after `hv` and before `build_obs`: live, the
tracker updates on this tick's detections and `assemble` reads the slots after that update, so
the sim promotes on the current sighting too. Not in `step` (that would lag a decision).

**Tests** (`tests/test_slots.py`, hand-built states via `allocate` + fields, calling
`slots.update` directly with a hand-built `hero_view`, default constants unless stated):

- seen once → no slot (pending); twice → slot 0; seen, missed, seen → still no slot (the miss
  reset the count); with `slots.promote_hits: 1` seen once → slot 0;
- two new entities promoted in one decision → slots 0 and 1 in entity order;
- unseen 1, 2, 3 decisions → slot held (misses 1, 2, 3); unseen 4 → freed; with
  `max_misses: 1` freed on the 2nd;
- return after being freed → lowest free slot, which differs from the old one when a newer
  entity took it;
- 10 candidates for 9 slots → the highest-index one pending, promoted when a slot frees;
- a dead entity's slot frees on the same schedule; `zero_` clears everything.
- **Parity with `EntityTracker`** (in `tests/test_deployment_tracker.py`, reusing `det_at` and
  `StubPlan`): a scripted sighting table over ~14 decisions (appear, coast 3, vanish, return,
  two at once, ten for nine slots) fed to the tracker as one `update` per decision (detections
  placed far apart so association is unambiguous) and to `slots.update` once per decision with
  the same table as `hero_view`; the slot → entity tables must be equal at every decision.

**Run.** `tests/test_slots.py tests/test_deployment_tracker.py tests/test_state.py
tests/test_env.py`.

**Considerations.** `check_invariants` (`state.py` line 204) iterates every field; make sure a
new I64 field with value 0 passes whatever range checks it has. `snapshot` picks the new fields
up automatically. No host sync: the whole update is tensor ops on (N, E) and (N, E, K).

---

## C9 — `slots: tracked` in `obs_select`; identity slots in the assembler

**Goal.** A per-entity group can declare `slots: tracked` and its rows are then ordered by
`state.slot_ent`. Deployment supplies the identity permutation, so the assembler's output is
still slot-ordered and nothing in the tracker changes.

**Files.** `brawl_sim/core/observation.py` (new `obs["slots"]`), `brawl_sim/core/obs_schema.py`
(two rows + shape letter), `brawl_sim/core/obs_select.py` (`GroupSpec` line 143,
`load_agent_spec` line 181, `_build_entity_group` line 358), `brawl_deployment/perception/assemble.py`
(`_put_entities`), `tests/test_obs_select.py`, `tests/test_deployment_assemble.py`.

**Edits.**

- `build_obs`: `obs["slots"] = {"entity": state.slot_ent, "valid": state.slot_ent > 0}`. Schema
  rows `("slots.entity", ("N", "S"), "int64", "index", (0, None), "Entity index + 1 held by tracked slot k; 0 = empty (core/slots.py).", False, None)`
  and `slots.valid` (bool); add `"S": cfg.n_enemies` to `_resolve_shape`. Refuse `slots.*` in
  a spec's `fields` in `load_agent_spec` (it is bookkeeping, not an observation).
- `GroupSpec.slots: str | None = None`. `load_agent_spec`: `slots = g.get("slots")`; allowed
  `None` or `"tracked"`; `"tracked"` requires `per_entity`, `entity_prefix == "entities"` and no
  `max_slots`; anything else raises with the group name.
- `_build_entity_group`: after `out` is built (hero dropped, so row `e - 1` is entity `e`):
  ```python
  if g.slots == "tracked":
      ent = full_obs["slots"]["entity"]                       # (N, K), entity + 1, 0 empty
      idx = (ent - 2).clamp(min=0)                            # entity e -> row e - 1
      valid = (ent > 0).to(torch.float32).unsqueeze(-1)
      out = torch.gather(out, 1, idx.unsqueeze(-1).expand(-1, -1, out.shape[-1])) * valid
  ```
  and restructure the fairness block so the mask is gathered by `idx` whenever `idx` is set
  (today it gathers only under `max_slots`). The group's shape is unchanged (K = n_enemies rows),
  so deploy5's `enemies` group has deploy4's width.
- Assembler `_put_entities`: always write `full["slots"] = {"entity": [[k + 2 if track k was
  written else 0 for k in range(K)]], "valid": entity > 0}`: track k sits at entities index
  k + 1 and the field stores entity + 1, so the gather above maps slot k back to row k (identity).

**Tests.**

- `test_obs_select.py`: a temp spec (`_write_spec`) = deploy4's groups with `slots: tracked` on
  `enemies`; an env with 3 enemies; set `env.state.slot_ent` by hand to `[[4, 2, 0]]` (slot 0 ←
  entity 3, slot 1 ← entity 1, slot 2 empty) and the matching `ent_slot`; build `full_obs` via
  `env._build_observation()` — careful: that calls `slots.update`, so either set the slots after
  building and call `build_obs` directly, or drive the update with a hand-built `hero_view`; then
  assert row 0 equals the untracked build's row for entity 3, row 1 entity 1's, row 2 zeros; an
  entity with `revealed_to_hero` False gives a zero row. Loader errors: `slots: tracked` with
  `max_slots` raises; on the `projectiles` prefix raises; `slots: other` raises.
- `test_deployment_assemble.py`: extend the sim-vs-assembler parity so `_suppliers_from` orders
  the enemy list by the sim's `slot_ent` and, with a `slots: tracked` temp spec, the assembled
  `enemies` group equals `build_agent_obs`'s byte for byte.

**Run.** `tests/test_obs_select.py tests/test_obs_schema.py tests/test_deployment_assemble.py
tests/test_configs_files.py tests/test_sb3_features.py`.

**Considerations.** `agent_obs_index_map` and the docs renderer are per-column and do not care
which entity fills a row. `_MAX_GROUPS` is unaffected (`slots` is a group option, not a group).

---

## C10 — `agent_obs_deploy5.yaml`, docs, test pins, design-doc entry

**Goal.** The spec the retrain uses, with everything that must be true of a deploy spec pinned
in tests, and the design doc telling the next reader what changed.

**Files.** New `configs/agent_obs_deploy5.yaml`; `docs/AGENT_OBS_DEPLOY5.md` and
`docs/OBSERVATION.md` (generated); `tests/test_configs_files.py` (`DEPLOY_SPECS` line 263,
`SEES_PICKUPS` line 387, the per-group field lists any parametrised test pins);
`BRAWL_DEPLOYMENT_DESIGN.md` §9; any lineage list that names deploy4 (`grep -rn deploy4 --include=*.md
--include=SKILL.md .`, excluding `runs/`).

**Spec.** Copy deploy4 verbatim, then: `self` gains `hero.near_edge` right after `hero.in_zone`
(27 floats; the two "where am I" flags read together); `enemies` gains `slots: tracked`;
everything else byte-identical. Header in deploy4's style: why a seventh file (specs are named
by path in `runs/*/train.yaml`), what changed and why (one paragraph each for the window, the
edge flag, the slots, the zone latch — the latch changes no column, only its meaning), the
deploy rule for each new field (`near_edge` ← `TrackerResult.hero_offset`; slots ← the tracker
already), and the width change. The `train.py` invocation line:
`python scripts/train.py --set run.agent_obs=configs/agent_obs_deploy5.yaml`.

**Docs.** `.venv/Scripts/python.exe scripts/dump_obs_schema.py --spec configs/agent_obs_deploy5.yaml`
(writes both docs).

**Test pins.** Add deploy5 to `DEPLOY_SPECS` and `SEES_PICKUPS` (True); add
`test_deploy5_differs_from_deploy4_by_exactly_near_edge_and_tracked_slots` (field-set difference
is exactly `{hero.near_edge}`, and the `enemies` group's `slots` is `"tracked"` in deploy5 and
`None` in deploy4). Run the parametrised tests and fix any that hardcode deploy4's `self` field
list. `hero.near_edge` stays out of `agent_obs.yaml` and `agent_obs_lowinfo.yaml` (their
"deploy minus lowinfo" pin applies to the first deploy spec only; check before assuming).

**Design doc.** Append a numbered entry to §9 (the last used number is 9.18, at line 3577; use 9.19): the
camera window and clamp, the edge flag on both sides, the slot rule, the zone latch, with a
pointer to `OBS_PARITY_PLAN.md`. **Patching trap:** this file contains real em-dashes; a typed
`--` will not match. Anchor an insertion on a unique short string, or append at the section end,
and verify with a re-read.

**Run.** `tests/test_configs_files.py tests/test_obs_select.py tests/test_deployment_loop.py
tests/test_deployment_assemble.py tests/test_deployment_grid.py -m "not vision"`.

**Considerations.** `configs/deployment.yaml` `run.dir` still points at deploy4 and must until
a deploy5 run exists (C12). Two places describe the *current* spec and get one sentence each:
`brawl_sim/maps/README.md` "What actually trains" (line 134) and
`.claude/skills/brawl-deployment/SKILL.md` (line 319). Mentions of deploy4 in
`SIM_OVERHAUL_STEPS.md`, `SIM_OVERHAUL_PLAN.md` and `BRAWL_DEPLOYMENT_DESIGN.md` are history and
stay.

---

## C11 — Verification pass

**Goal.** One session that proves the whole of C1–C10 holds together, with numbers.

1. Full suite: `.venv/Scripts/python.exe -m pytest tests -q -m "not vision"` (~20 min; 0
   failures expected as of 2026-09-22). Paste the summary line; list any failure verbatim.
2. `scripts/probes/obs_leak_measure.py 64` and `scripts/probes/obs_quad_measure.py 64`: a
   table of the new numbers next to the baseline in §2 of this file.
3. `scripts/smoke_test.py` (check its CLI for a CPU/short mode) and `scripts/benchmark.py --help`.
4. `graphify update .`.
5. A read-through of every new docstring against rule 11.

The report is a markdown table plus the pasted test summary. No opinions on retraining.

---

## C12 — Retrain and gates (lead, not a delegate)

**Status: FINISHED 2026-09-25; gates 1–4 passed, gate 5 (live) is the lead's** (Status block,
"Done 2026-09-25").

Not for a small model: it needs the GPU and judgement calls.

1. Check nothing is training (`nvidia-smi`; a running `scripts/train.py`).
2. `scripts/train.py --set run.agent_obs=configs/agent_obs_deploy5.yaml` on the deploy4 recipe
   (`runs/mortis_deploy4-20260921-185945/train.yaml`), run name `mortis_deploy5`; add
   `--set run.randomization=configs/randomization.yaml` if Z2 has landed (Z4 then folds into this run).
3. Gates, in order: C3's leak numbers on the new checkpoint; the deploy4 wall-push protocol
   (`TierEvaluator`, elite, 96 episodes, training and holdout maps, wall-push fraction and win
   rate) at or better than deploy4 + mask (note the deploy4 checkpoint is now evaluated under the
   camera-limited sim, so compare both under the same sim); the 2026-09-23 stuck clip replayed
   through the live path picks a legal bin; `configs/deployment.yaml run.dir` → the deploy5 run;
   a live BlueStacks session.
4. Memory and plan status updates.

---

## P1 — Probe: measure the clamp onsets from footage (optional, parallel)

**Status: MEASURED 2026-09-24** (`scripts/probes/edge_probe.py`; results in the Status block and `OBS_PARITY_PLAN.md` §2.3; all four defaults stand).

**Goal.** Replace the four geometry-default onsets with measured ones, or confirm them.

**Script.** `scripts/probes/edge_probe.py`. Open a clip with
`brawl_vision.sources.open_source(spec, cfg)`; build the vision stack the way
`brawl_deployment/loop.py` `VisionStack.build` does (this loads the detectors: **check
`nvidia-smi` first and prefer the CPU provider if the stack allows it**; a training run owns the
GPU). Per frame: detections → the `player` box → `to_tiles([box], plan)` → minus the nominal
anchor (viewport centre through `plan.M` and `plan.rect_to_tile`, plus `HERO_ANCHOR_TILES`) =
offset; also odometry's `position_tiles`. Report every run where `|offset| > 1` tile on an axis
for more than 0.5 s: plateau value and sign, duration, and odometry's camera motion over the
run (near 0 while the hero walks means the clamp is real). The plateau when the hero hugs an
edge *is* that side's onset.

**Clips.** The 2026-09-23 OBS recordings `C:\Users\mateo\Videos\2026-09-23 *.mp4` (six files,
17:32 to 20:42; one holds 13 matches, and sidecar defaults break on multi-match files, so pass
explicit spans; `20-20-47` already has a bounds sidecar);
`brawl_vision/data/training_videos/9-10_new1..8.mp4`; `tests/fixtures/vision/showdown_alternate_map.mp4`,
`showdown_alternate_map2.mp4`, `bluestacks-example-new.mp4`, `bluestacks-example3.mp4`. OBS
captures need `--hud emulator` and skip ~12 s of lobby; date-named clips otherwise get the phone
mask.

**Output.** A table per side (clip, time, plateau, camera motion) appended to
`OBS_PARITY_PLAN.md` §2.3, then the yaml values in C1's block updated with a "measured
<date> on <clips>" comment; a side with no clip reaching it keeps the default and the comment
says so. Expect the BlueStacks clip's mid-map offset to sit near 0 (assumption A1 in the plan).

---

## Phase 2 — the zone schedule timed from the game

**What changes.** Today the sim's zone is three guessed scalars in `configs/default.yaml`
(`zone:` block, lines 170–185): `start_fraction: 0.08` (first shrink at 0.08 × 150 s = 12 s),
`step_seconds: 1.5`, `tiles_per_step: 1`. Phase 2 replaces them with values measured from
recordings and then jitters `start_fraction` and `step_seconds` by ±10 % per episode, so the
policy cannot learn the schedule by heart. No observation row changes: `zone.active` is Phase 1's
latch (C5), `next_shrink_in` stays pinned at zero on the live side (BRAWL_DEPLOYMENT_DESIGN.md
§9 entries 14 and 15, settled), and `docs/OBSERVATION.md` is untouched.

**Why it transfers.** The live `meta.time_frac` is wall-clock since the match gate opened ÷ 150 s
(design doc §9 entry 6, settled) and the sim's episode is `max_episode_steps 3000 × dt 0.05` =
150 s (`configs/default.yaml` lines 57–58), so a first-gas time measured *after the gate* becomes
`start_fraction` by one division, and a step period measured in seconds is `step_seconds` as is.

**Order.** Z1 needs nothing from Phase 1 and runs alongside it. Z2 needs Z1's numbers and must
wait for C1 and C5 to land (shared files: `configs/default.yaml`, `brawl_sim/config.py`,
`tests/test_config.py`, `tests/test_zone.py`). Z3 is the docs-and-suite pass, with or after C11.
Z4 is the lead's run, after C12.

### Facts for Phase 2

- **Sim schedule plumbing.** `zone_step_seconds` and `zone_tiles_per_step` are per-env
  parameters (`brawl_sim/config.py` `PER_ENV_FIELDS`, lines 545–546); `zone_start_time` derives
  per env as `start_fraction × max_episode_steps × dt` (lines 685–687). Every one of them goes
  through `_resolve_value` (line 647): a `{low, high}` dict samples uniformly per env, a scalar
  broadcasts. `resample_params` (line 742; zone lines 758–760) redraws them on every reset, and
  `spawn.reset_envs` calls it (`brawl_sim/core/spawn.py` line 188) *before* `zone.init_zone`
  (line 236), so a fresh draw is what the new episode's zone uses. `core/zone.py` lines 27–65:
  first shrink at `zone_start_time`, then every `zone_step_seconds` the safe rect loses
  `tiles_per_step` per side, clamped so `hi − lo ≥ 2`. Nothing in `core/zone.py` needs to change.
- **Validation is sample-level.** `validate(cfg, params)` (`config.py` line 842; zone checks
  907–913) refuses `zone_start_time ≥ 150 s` and `dt ≥ zone_step_seconds` on the *sampled*
  tensors. A range whose low end is invalid passes construction and fails at some later reset.
  `_resolve_value` (line 650) already refuses `low > high`.
- **The overlay.** `configs/randomization.yaml` ships fully commented out; keys are
  `section.field`, values `{low, high[, mode]}`; `mode: multiplicative` scales a **scalar** base
  (`apply_randomization`, `config.py` lines 324–360, raises if the base is itself a dict). It is
  loaded only when the train yaml sets `run.randomization` (`brawl_sim/training/config.py` line
  70 → `builder.py` line 125 → `BrawlVecEnv(randomization=path)`; `env.py` lines 194–204 build
  `self.spec`, then 207–208 `build_params` + `validate`). `env.py` line 196 also accepts a
  hand-written dict. `TierEvaluator` passes **no** randomization on purpose (`evaluation.py`
  lines 104–110), so evaluation always runs at the base scalars.
- **Tests that read the base as a scalar.** `tests/test_zone.py` lines 77–90 cast
  `spec["zone"]["start_fraction"]` with `float(...)`; `tests/test_config.py` line 208's
  `config_dir` fixture writes its own yaml into `tmp_path` (the shipped files are not what it
  reads); line 689 checks `validate` on sampled values.
- **Live gas path.** `detect_zone(rect, plan, cfg)` (`brawl_vision/terrain/zone.py` line 72)
  returns a `ZoneMask` (line 34): `cells` = gassed tiles on the plan's tile grid, `observed` =
  tiles with enough real world in them to judge (a tile under the HUD or outside the viewport is
  neither gassed nor clear), `gassed_cells` (line 48). `GasMap`
  (`brawl_deployment/perception/grid.py` lines 294–359) is the sticky world-frame map: `update(zone,
  plan, odometry)` deposits `cells` into `gassed` and `observed` into `seen`, returns the number
  of newly gassed cells (`fresh = zone.cells[sub] & ~self.gassed[dst]`), deposits nothing unless
  `odometry.status == "ok"`, and a new odometry `segment` resets both grids. The loop calls it
  every tick from `_perceive` (`loop.py` lines 682–727, `self.grid.observe_zone` line 722) with
  the lattice-registered plan. `ZoneEstimator.active` (`perception/zone.py` line 133) is
  `gassed.any()`.
- **The clock.** `elapsed_s = perf_counter() − _match_t0` (`loop.py` line 807); `_match_t0` is
  set when the gate opens (line 1080), which is `enter_samples: 6` consecutive above-threshold
  ticks at 12 Hz = **0.5 s after the first in-match frame** (`control_calibration.json`
  `match_gate`: anchor `hypercharge`, threshold 0.45, `exit_samples: 4`). So "after the gate" =
  "after the first gameplay frame" + 0.5 s.
- **Reveal vs advance.** The wall the design doc hit (§9 entry 14, lines 3290–3298: new gassed
  cells arrive on 137 of 1543 ticks with a median gap of one frame; re-measured on BlueStacks,
  lines 3313–3336: the deepest visible gas "moves 7 tiles in 3 s at t≈21–24 s", which was the
  camera revealing a band) is that a map that only grows cannot separate "the gas advanced"
  from "the camera panned over gas that was already there". `GasMap.seen` separates them: a cell
  that was `observed` and *not* gassed on an earlier frame and is gassed now advanced in between,
  and the gap between its last clear sighting and its first gassed sighting bounds *when*. Only
  gaps of a frame or two are sharp enough to time a step; everything else is a reveal. This is §10
  item 9 of the design doc (lines 3691–3695, "a front that stays on screen has unambiguous
  steps") done offline on frames that already exist.
- **Clips.** Fixtures in `tests/fixtures/vision/` (each has a `.bounds.json` sidecar):
  `zone_grows_from_east.mp4` (phone HUD; 1351 frames, usable 0–1219, content box 2000×1125; gas
  present by frame 1180 per `tests/test_vision_zone.py` line 166; its tile lattice is exact from
  frame 60 onward), `bluestacks-example-zone.mp4` (emulator HUD; 66 s at 1920×1080; the design doc
  at lines 3313 and 3687 puts the gas spawning at t≈21 s of clip time and closing in until the
  hero dies), `showdown_alternate_map.mp4` (its bushes share the gas hue — a false-positive check,
  not a timing source), `showdown_has_gadget.mp4` (gas at frame 810).
  `brawl_vision/data/training_videos/9-10_new1..8.mp4` (bounds and gameplay sidecars exist).
  `C:\Users\mateo\Videos\2026-09-23 *.mp4`, OBS captures of the emulator: date-named, so
  `labeling.default_hud` gives them the *phone* mask unless told `emulator`; one file holds 13
  matches and the sidecar heuristics keep one span, so pass explicit frame spans; `20-20-47`
  already has a bounds sidecar; the first ~12 s of each are lobby.
- **Frames.** `open_source(path, cfg)` (`brawl_vision/sources.py` line 65) yields `Frame`s with
  real timestamps `frame.t` and file-relative `frame.index`; `ClipReader` (`clips.py` line 180)
  crops to the content box but does not resize, and `RectifyPlan.rectify` refuses anything but
  the calibrated 2002×1126 viewport, so wrap the iterator in `at_viewport(frames, plan.viewport)`
  (`sources.py`, just below `is_live`) exactly as `scripts/vision_watch.py` line 200 does.
  `Odometry(plan, cfg)` (`brawl_vision/terrain/odometry.py` line 81) `.update(rect)` returns an
  `OdometryResult` (line 57: `position_tiles`, `status`, `segment`, `.ok`).
- **Clock origin for a clip.** `scan_gameplay(path, scan_fps)` (`brawl_vision/gameplay.py` line
  293) scores the button rings on a stepped decode and `span_frames(scan)` (line 373) gives the
  file-relative gameplay frames; the first one is the first in-match frame. Call `scan_gameplay`
  directly: `load_gameplay_span` (line 355) *rewrites* the `.gameplay.json` sidecar whenever the
  scan parameters differ from the cached ones, and the training-video sidecars are checked in.
  On 1080p emulator captures the ring scan can find nothing (memory: "NO GAMEPLAY FOUND"); the
  probe then needs `--t0-frame`, found by eye with `scripts/vision_watch.py`.
- **Design-doc numbering.** The last used §9 entry is **18** (line 3577), so C10 writes 9.19 and
  Z3 writes 9.20. Do not renumber anything.

---

## Z1 — Probe: time the zone schedule from footage (parallel with Phase 1)

**Status: MEASURED 2026-09-24** (`scripts/probes/zone_probe.py`; T0 = 19 s, D = 1 tile, rate 0.14 tiles/s per side; the burst `P` below is superseded by the per-line front rate, see the Status block and `OBS_PARITY_PLAN.md` §4).

**Goal.** Three numbers with provenance, each backed by a count of the events behind it:
`T0` = seconds from the match gate to the first gas on screen, `P` = seconds between successive
gas advances, `D` = tiles the front moves per advance. Plus the raw per-clip tables, appended to
`OBS_PARITY_PLAN.md` §4.

**Script.** New `scripts/probes/zone_probe.py` (CPU only; no detector loads — copy the per-frame
pattern from `scripts/vision_watch.py` lines 41–72 and 186–201, *not* `VisionStack.build`, which
loads two ONNX detectors this probe never uses). CLI: `clip` (path), `--hud {phone,emulator}`
(default `labeling.default_hud(clip)`), `--start/--stop` (file-relative frame indices, inclusive),
`--t0-frame N` (overrides the gameplay scan), `--sharp-s 0.3`, `--burst-gap 0.5`, `--out
<path.md>` (optional; otherwise stdout only).

Per frame, in this order:

1. `frame` from `at_viewport(open_source(clip, cfg), plan.viewport)`, skipped outside
   `[start, stop]`; `rect = plan.rectify(frame.image)`; `odo = odometry.update(rect)`;
   `zone = detect_zone(rect, plan, cfg)`.
2. Snapshot `gassed_before = gas.gassed.copy()`, `seg_before = gas._segment`, then
   `n_new = gas.update(zone, plan, odo)` on a `GasMap(cfg.occupancy_grid_h, cfg.occupancy_grid_w)`
   (128×128, `brawl_vision/config.py` lines 127–128). Using the real `GasMap` keeps the probe on
   the loop's own gating and rounding; the bookkeeping below sits outside it.
3. `fresh = gas.gassed & ~gassed_before` over the whole grid (128×128 bools; cheaper to diff than
   to replicate `GasMap.update`'s window arithmetic). If `gas._segment != seg_before` the map was
   reset: also reset the probe's `last_clear_t` grid (float, NaN) and count a reset.
4. For each fresh cell: `gap = frame.t − last_clear_t[cell]` (NaN → never seen clear → a reveal).
   `gap ≤ sharp_s` → a **sharp** advance event at `frame.t`; otherwise a reveal. Record `(t,
   row, col, gap)` for every fresh cell either way.
5. Then update `last_clear_t[cell] = frame.t` for cells that are `observed & ~cells` in this frame
   *and* not yet gassed in `gas.gassed` (deposit them through the same `col0/row0` placement
   `GasMap.update` uses, or simply mark the cells of `zone.observed & ~zone.cells` after mapping
   them with the plan's `origin_tile` + `odo.position_tiles` the way lines 346–347 do; either is
   fine, but the placement must match `GasMap`'s rounding or the two grids disagree by a cell).

Clock origin: `t_gate = t(first gameplay frame) + 0.5` (the gate's `enter_samples` at 12 Hz). The
first gameplay frame is `span_frames(scan_gameplay(clip, scan_fps=10.0))[0]` (±0.05 s); if the
scan returns no span, refuse to guess: print "no gameplay span; pass --t0-frame" and exit
non-zero. Report both `t_clip` (from `frame.t`) and `t_after_gate = t_clip − t_gate`.

**Outputs, per clip.**

- A per-second table: `t_after_gate | frames | frames with odometry ok | gassed cells (cumulative)
  | fresh cells | sharp events | resets`. Odometry-off seconds and resets must be visible here,
  not summarised away.
- **First gas:** the first frame with `gas.gassed.any()` after `update` (so it respects the
  odometry gate the loop has), as `t_clip` and `t_after_gate`.
- **Bursts:** sharp events sorted by time, split wherever the gap to the previous event exceeds
  `burst_gap`. Print every burst (`t_start`, `t_end`, cells, rows spanned, cols spanned). The
  periods between consecutive burst starts, their median and inter-quartile range, and the count
  of periods: that is `P`.
- **Depth:** per burst, the extent along the shorter of the burst's row-span and col-span (a
  front that runs east–west spans many columns and a few rows; the few rows are the depth), and
  `cells / longer span` as a second estimate. Median over bursts is `D`. Both are in *cells* of
  the odometry frame: without the lattice tracker the cell boundaries sit at an arbitrary
  sub-tile phase, so a one-tile advance can print as a 1–2 cell band; `zone_grows_from_east` is
  phase-exact from frame 60 and is the reference for `D`.
- A closing line per clip: `usable for P: yes (n bursts)` when ≥ 3 sharp bursts were found with
  odometry ok throughout, `no` otherwise, with the reason (few sharp events, resets, hero died,
  no gameplay span).

**Clips to run** (in this order; stop adding once `P` has ≥ 6 periods from ≥ 2 clips):
`tests/fixtures/vision/zone_grows_from_east.mp4` (`--hud phone`), `bluestacks-example-zone.mp4`
(`--hud emulator`), `9-10_new1..8.mp4`, then one or two of the 2026-09-23 OBS recordings with
explicit `--start/--stop` per match and `--hud emulator`. `showdown_alternate_map.mp4` runs once as
a false-positive check only: its "first gas" should not land in the lobby or at t = 0.

**Acceptance.**

1. Both fixture clips run end to end on the CPU in minutes and print the tables above.
2. `zone_grows_from_east.mp4`: first gas at a frame index ≤ 1180 and at least one sharp burst.
   `bluestacks-example-zone.mp4`: first gas within 19–23 s of *clip* time (the design doc's
   ≈21 s), and the printed `t_after_gate` differs from `t_clip` by the gate offset.
3. `OBS_PARITY_PLAN.md` §4 gets a "Measured <date>" table: `T0`, `P` (median, IQR, number of
   periods), `D` (median, number of bursts), the clips each came from, and the per-clip usability
   lines. A number that could not be measured (fewer than 3 sharp bursts across all clips for
   `P`, no sharp burst for `D`) is written as "not measurable from these clips", the sim's current
   value is quoted beside it, and Z2 then changes only what was measured.
4. If the burst times are plainly not periodic (a long pause then a run of quick steps, or the
   period drifting monotonically), the report says so with the raw burst list and stops there:
   the sim's single-period model would need a design change, which is the lead's decision.

**Considerations.**

- `frame.t` is the clip's real timestamp (`tests/test_vision_clips.py` line 145 exists because
  it is not `index / fps`); never derive time from the index.
- Tiles under the HUD are never `observed`, so a front arriving from the HUD side (the bottom of
  the screen) yields fewer sharp events than one from the top or sides. Report bursts with their
  direction (which side of the map centroid they sit on) so a one-sided sample is visible.
- A segment reset drops `seen`, so no sharp event is possible within `sharp_s` of a reset; a clip
  with many resets is a poor timing source and the table shows it.
- The odometry origin is wherever tracking started; a two-minute match cannot leave the 128×128
  canvas, but a run of `fresh cells = 0` with gas on screen is what it would look like. Print the
  gassed-cell count of `zone` (per frame) next to `n_new` so that case is distinguishable.
- Computing bounds for a recording that has no sidecar writes a `.bounds.json` beside it
  (`clips.load_bounds`, line 163). For the OBS files that is outside the repo and fine; for
  fixtures the sidecars already exist. Never write a `.gameplay.json` under `tests/fixtures/` or
  `brawl_vision/data/` (call `scan_gameplay`, not `load_gameplay_span`).
- `showdown_alternate_map`'s bushes share the gas hue: gassed counts there are inflated. Do not
  use it for any of the three numbers.

**What not to do.** No YOLO, no lattice tracker, no GPU, no edits to `brawl_vision` or
`brawl_deployment`, no sidecar under a repo data directory, no attempt to supply
`next_shrink_in` live (pinned by decision). The numbers feed the sim config only.

---

## Z2 — Config: measured schedule, ±10 % overlay, range-aware validation

**Status: BUILT 2026-09-25, adapted** (Status block, "Done 2026-09-25"): at the lead's 1.3× pace,
as a run setting in `configs/train.yaml` rather than scalars in `configs/default.yaml`; step 2's
overlay shipped as written; step 3 (`check_zone_ranges`) not built.

**Goal.** `configs/default.yaml` carries the measured schedule with provenance; training jitters
`start_fraction` and `step_seconds` ±10 % per episode; a configuration whose range could sample
an invalid value is refused when the environment is built, not at the thousandth reset.

**Inputs.** Z1's `T0`, `P`, `D` (use only those Z1 marked measured). **Wait for C1 and C5** (shared
files).

**Steps.**

1. `configs/default.yaml` `zone:` block (lines 173–175 today): `start_fraction: <T0 / 150>` with
   the arithmetic in the comment ("measured <date> on <clips>: first gas T0 s after the gate,
   over the 150 s episode"), `step_seconds: <P>`, `tiles_per_step: <D>`, each with its clip
   provenance and event count. **Keep them scalars.** Three reasons, all in the facts block:
   `tests/test_zone.py` line 85 casts the yaml value with `float`; `TierEvaluator` deliberately
   passes no randomization, so evaluation and the deployed loop's own read of the env config see
   the nominal schedule; and `apply_randomization`'s multiplicative mode needs a scalar base.
   This supersedes the plan's §4 step 2 sentence that puts `{low, high}` into `default.yaml`.
2. `configs/randomization.yaml`: add, uncommented, keeping every other line commented (the file
   is shared, and an uncommented bot range would silently change what the deploy5 recipe trains
   against):

   ```yaml
   zone.start_fraction: {low: 0.9, high: 1.1, mode: multiplicative}
   zone.step_seconds:   {low: 0.9, high: 1.1, mode: multiplicative}
   ```

   `tiles_per_step` stays fixed: the game shrinks by whole tiles, and `_F32` would happily apply
   a fractional step.
3. Range-aware validation: `check_zone_ranges(spec: dict, cfg: EnvConfig) -> None` in
   `brawl_sim/config.py`, next to `check_removed_keys` (line 725). Read `zone.step_seconds` via
   `_dget` (line 171): if a dict take `low`, else the scalar; require `> cfg.dt` (the sampled
   rule at line 913, applied to the worst case). Read `zone.start_fraction`: if a dict take
   `high`, else the scalar; require `< 1.0` (so `zone_start_time < episode_seconds`, the line 908
   rule). Raise `ValueError` naming the dotted key and the bound. Call it from
   `BrawlVecEnv.__init__` right after `self.spec` is final (`env.py` line 204) and before
   `build_params` (line 207). Keep `validate`'s sampled checks; they still catch a hand-built
   `SimParams`.
4. Recipe note for C12/Z4: the deploy5 run gains `--set run.randomization=configs/randomization.yaml`.
   If C12 has already started when this lands, that flag belongs to Z4's fine-tune instead. The
   deploy4 recipe has `randomization: null` (`runs/mortis_deploy4-20260921-185945/train.yaml` line
   11); never edit anything under `runs/`.

**Tests.**

- `tests/test_configs_files.py` (pins on the shipped files): `load_randomization(CONFIGS /
  "randomization.yaml")` returns exactly the key set `{"zone.start_fraction", "zone.step_seconds"}`
  (an accidentally uncommented line fails this), both `mode == "multiplicative"` with `low 0.9`
  and `high 1.1`; the shipped `zone:` block holds plain numbers for the three fields; the
  provenance comment contains a date (the `tests/test_vision_zone.py` line 160 "PLACEHOLDER"
  pattern, inverted).
- `tests/test_config.py`: `apply_randomization(spec, load_randomization(shipped))` gives
  `zone.step_seconds == {low: 0.9 P, high: 1.1 P}` (`pytest.approx`) and leaves `tiles_per_step`
  a scalar; `check_zone_ranges` raises for `{"zone": {"step_seconds": {"low": 0.04, "high":
  2.0}}}` at `dt 0.05` and for `start_fraction {low: 0.5, high: 1.0}`, and passes on the shipped
  spec with the shipped overlay. The line 689 test stays as is.
- `tests/test_zone.py`: with the overlaid spec (`apply_randomization(spec, load_randomization(...))`)
  and `n_envs=64`, `params.zone_start_time` lies within `[0.9, 1.1] × T0` for every env with
  `std > 0`, `params.zone_step_seconds` likewise around `P`; after `resample_params` with an
  all-true mask neither tensor is `torch.equal` to before. The line 77 first-shrink test keeps
  passing untouched because `default.yaml` stays scalar. Lines 134–147 untouched.
- An end-to-end construction: `BrawlVecEnv(load_config("configs/default.yaml"), n_envs=4,
  device="cpu", randomization="configs/randomization.yaml")` builds and `validate` passes; the
  same with `randomization={"zone.step_seconds": {"low": 0.02, "high": 0.03}}` (`env.py` line 196
  accepts a dict) raises with `zone.step_seconds` in the message.

**Run.** `.venv/Scripts/python.exe -m pytest tests/test_config.py tests/test_zone.py
tests/test_configs_files.py -q`.

**Acceptance.** Those three files green; the two construction checks above pass and fail as
stated; `grep -n "start_fraction\|step_seconds" brawl_deployment/` finds nothing (the deployed
side reads only `margin_horizon_tiles` from this block, `default.yaml` lines 179–185, so the
schedule change cannot reach the loop by accident); `graphify update .` run.

**Considerations.**

- Redrawing on every reset (`resample_params`, `config.py` lines 758–760; `spawn.reset_envs` line
  188 before `init_zone` line 236) is the intended jitter: each episode of each env gets its own
  schedule.
- The overlay's multiplicative mode scales the base at load time (`apply_randomization` lines
  344–354), so changing the measured scalar in `default.yaml` moves the range with it; the tests
  compute expectations from the loaded scalar, never from a literal.
- `check_removed_keys` (line 725) is untouched; nothing is renamed.
- The smoke test (`scripts/smoke_test.py`) does not pass a randomization; leave it.

**What not to do.** No `{low, high}` in `default.yaml`; no range on `tiles_per_step`; no other
overlay lines uncommented; nothing under `runs/`; no `agent_obs_*` edits (no schema change).

---

## Z3 — Docs, pins and the verification pass

**Status: DONE 2026-09-25** (Status block, "Done 2026-09-25"; the design doc's entry is 20, the
"9.20" below).

**Goal.** The measurement and the config change are recorded where the next reader looks, and
the suite is green with both phases in.

1. **Design doc.** Append entry **9.20** to BRAWL_DEPLOYMENT_DESIGN.md §9 (after C10's 9.19; the
   last entry today is 18 at line 3577): "MEASURED <date> — the zone schedule from footage": the
   three numbers with their event counts and clips, the method in two sentences (`GasMap.seen`
   separates advance from reveal; sharp events only), what changed in the config (scalars +
   the ±10 % overlay), and that `next_shrink_in` stays pinned (entries 14 and 15 unchanged).
   Same patching trap as C10: the file holds real em-dashes, a typed `--` will not match; anchor
   on a unique short string or append at the section end and re-read. If Z1 reached ≥ 3 sharp
   bursts, add one line under §10 item 9 (lines 3691–3695): "superseded — timed offline from
   `GasMap.seen`, see 9.20"; otherwise leave item 9 alone and say so in the report.
2. **Plan.** A `**Status:**` line at the top of `OBS_PARITY_PLAN.md` §4 in the file's header style:
   measured (date, Z1), configured (date, Z2), retrained (Z4 pending).
3. **Docs.** No regeneration: Phase 2 adds no schema row. Prove it: run
   `.venv/Scripts/python.exe scripts/dump_obs_schema.py --spec configs/agent_obs_deploy5.yaml` to
   a scratch path (check its `--out`/stdout options first) and `diff` against the committed
   `docs/AGENT_OBS_DEPLOY5.md`; the diff must be empty. Do not overwrite the docs.
4. **Comments.** Re-read the `default.yaml` and `randomization.yaml` comments against rule 11.
5. **Suite.** `.venv/Scripts/python.exe -m pytest tests -q -m "not vision"` (~20 min, 0 failures
   expected); paste the summary line; any failure verbatim. `graphify update .`.

Report per rule 12. No opinions on the retrain.

---

## Z4 — Fine-tune or retrain under the jittered schedule (lead, not a delegate)

Needs the GPU and judgement.

**Status: the lead's next run; the recipe is in the Status block ("Done 2026-09-25").** Z2 landed
after C12 started, so the fold-in path of step 2 is gone. The new recipe's episode is 185 s: step
4's live check reads `T0 / 185` (about 0.10 at the game's 19 s) against the sim's 0.08, a gap that
is the 1.3× pace, by design.

1. Prerequisites: C12's deploy5 checkpoint has passed its gates; Z2 and Z3 landed; nothing is
   training (`nvidia-smi`; a running `scripts/train.py`).
2. **Cheapest path:** if Z2 lands before C12 starts, C12's recipe simply gains
   `--set run.randomization=configs/randomization.yaml` and Z4 is folded into C12's run and gates.
   **Otherwise fine-tune:** `scripts/train.py --resume runs/mortis_deploy5-<stamp>/best_model.zip
   --set run.randomization=configs/randomization.yaml` on the deploy5 recipe (`--resume`, `train.py`
   line 94; `_resume`, line 274, restores the model, the VecNormalize statistics and
   `curriculum.json`; `--curriculum-stage` line 96 to start at the final stage). The observation
   schema is unchanged, so the normalisation statistics stay valid. Budget about a fifth of the
   deploy5 run; stop early if the C12 gates hold.
3. Gates: C12's gates re-run on the new checkpoint (evaluation is at the nominal schedule by
   design, `evaluation.py` lines 104–110). Phase 2 adds one: gas deaths per episode and time in
   gas, at the nominal schedule and at the two ±10 % extremes, built with a hand-written overlay
   (`BrawlVecEnv(..., randomization={"zone.step_seconds": {"low": 0.9, "high": 0.9, "mode":
   "multiplicative"}, "zone.start_fraction": {...}})`), must not be worse than deploy5's at
   nominal. A policy that only holds at the slow end has learned the schedule, not the gas.
4. Live: one BlueStacks session watching the first-gas moment; `zone.active` (the C5 latch, the
   live `ZoneEstimator.active`) should flip near the `time_frac` the sim now expects
   (`T0 / 150`); read it from the tick telemetry. `next_shrink_in` stays pinned.
5. Memory and plan status updates.
