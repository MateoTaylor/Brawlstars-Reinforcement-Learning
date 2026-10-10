# Brawl sim design

This is the design of `brawl_sim` as it stands on 2026-09-30. It covers the rules in force, how each mechanic works, and what is still open. Numbers are the shipped configs' values (`configs/default.yaml`, `brawlers.yaml`, `train.yaml`, `randomization.yaml`); if this document and those files disagree, the files are right.

Related documents:
- [CONVENTIONS.md](CONVENTIONS.md): how code here is written, and the canonical tick order.
- [TRAINING.md](TRAINING.md): how to train.
- [CHARACTER_DETAILS.md](CHARACTER_DETAILS.md): the real brawlers' statistics.
- [brawl_sim/maps/README.md](brawl_sim/maps/README.md): the map pool and seeds.
- [BRAWL_DEPLOYMENT_DESIGN.md](BRAWL_DEPLOYMENT_DESIGN.md): live play.

This document replaces the sim's plan documents: the build plan, the bot overhaul, the sim overhaul, observation parity and the speed audit. They are kept, unmaintained, in [archive/](archive/) for their measurements and history.

## 1. Status and open items

**Built.**
- Mortis against seven data-driven bot kinds, with his dash, long dash, super, gadget and auto-aimed attack.
- Bot personalities, difficulty tiers and cube farming.
- A camera-window reveal with tracked enemy slots, `hero.near_edge`, a latched `zone.active` and three decisions of local history.
- Gas timed from the real game.
- 38 map CSVs and holdout evaluation.
- Since 2026-10-07, behind keys only `configs/train.yaml` turns on (SIM_ISSUES_PLAN.md): bot pathfinding, the final-4 endgame switch, 8000-HP crates hit once per dash, the game's 99-cube cap and tap-aimed bots; then gas rules measured along the way out, a retreat goal, a strafe that turns round at a wall and a `move_reversal` reward term; and a projectile under 2.0 tiles/s reading as still in the agent's view, as live reports it. `runs/mortis_ppo-20261007-161117` is the first run trained on them.

**In use.**
- `configs/train.yaml` trains `configs/agent_obs_deploy5.yaml` for 600M decisions on 4096 envs and 34 maps.
- `configs/deployment.yaml` serves `runs/mortis_ppo-20261007-161117` (best_model.zip, the 580M eval), since 2026-10-08.

**Open.** Each item waits on its owner.

*Sim behaviour*

| # | Item | Owner |
|---|---|---|
| 2 | **The piercing super leaves the map** for a tick before its range runs out. With `debug_checks` off this is harmless, because terrain lookups clamp. With `debug_checks` on, `state.check_invariants` raises. Either kill the bolt at the border or relax the invariant. | user |
| 3 | **The super ignores cubes.** `super_damage` and `super_heal` are flat, while attacks and the gadget get +10 % per cube. No decision is on record either way. | user |
| 4 | **Same-tick pickup revive.** Pickups (phase 12) run before deaths (phase 13) and check `ent_alive`, not HP > 0. A unit at 0 HP can therefore take a cube, gain 400 HP and survive. | user |
| 5 | **Same-tick double death.** `compute_rank` ties same-tick deaths. A hero who dies on the same tick as the last bot ranks first and is paid `win_bonus` and `death_penalty` together. | user |
| 6 | **Off-screen gadget aim and in-reach pay.** `hero.gadget_target` and the `attack_in_reach` flag read concealment-only visibility. So the spinner can bear toward an unconcealed enemy off camera (still at most 2 tiles), and the reward can pay an attack whose only in-reach enemy is off screen. The second is rare, because the 3.77-tile reach sits inside the window. | user |
| 7 | **`ent_react_t` is never set.** The bots' `react_t <= 0` fire gate and the privileged `react_t` field are therefore inert, and `reaction_delay` only smooths movement. Wire it or remove it. | user |
| 8 | **Projectile slots.** There are 192, against a demand bound of 162 and much lower observed peaks. Overflow drops shots silently, so add a dropped-shot counter before lowering the count. | user |

*Parity with the live game*

| # | Item | Owner |
|---|---|---|
| 9 | **Clamp onsets are unmeasured.** The defaults (W 12.1, E 12.8, N 8.9, S 5.5) look 2–3 tiles too large. Localization's fixes put the game's camera 1.9–2.9 tiles past the sim's limits, which means W ≤ 10.2, E ≤ 10.3, N ≤ 6.0 and S ≤ 2.6. The edge probe gives only lower bounds (W ≥ 10.1, E ≥ 7.2), and on S it disagrees with localization by about a tile. A refit would move the reveal near edges, `near_edge`, `localize.py`'s landing windows and every run trained after it. | user |
| 12 | **Grid sliver near a W/E clamp.** The lower rows of the hero-centred crop hold about 0.5 tiles (W) or 0.9 tiles (E) of map that the slanted window does not show. The crate, pickup and projectile planes draw it. Accept it, or window those planes. | user |
| 13 | **Action latency.** The sim trains at 0 ticks of latency. Live, the attack lands about 167 ms after the decision and movement a tick later. The proposal is a per-column latency in `act_buf`. | user |
| 14 | **Observable lag.** Live, ammo, HP and super read about 1 s late; the sim models none of it. Deferred on 2026-09-24. | user |
| 15 | **Live checks not yet run:**<br>• `hero_offset` and `near_edge` on a real edge approach, and `zone.active` flipping near `time_frac` ≈ 0.10.<br>• The cadence audit on live telemetry (`scripts/audit_attack_cadence.py --telemetry`).<br>• The gas gate for the 1.3× run: gas deaths and time in gas at nominal and at both ±10 % ends, no worse than deploy5 at nominal. | user |
| 16 | **A CV reader for the gadget button**, then adding the gadget to `resync`. The trigger is telemetry showing a modelled throw with no damage spike within 0.5 s. | user |
| 17 | **Replay the live gadget-anchor trace in a test.** Copy the trace under `tests/fixtures/` and replay it through the match gate; only such a test guards against regressions. | unowned (code) |
| 18 | **The dry run against the gadget spec** (`control.backend: null`) is probably superseded by the 2026-09-28 localization dry runs. Close it or re-run it. | user |
| 26 | **The game's super tap-aim is unmeasured.** Since 2026-10-06 the live idle super is a bare tap on the super button, as the sim aims it (tap-to-fire at the nearest enemy in the bolt's 11.1-tile reach, §4; BRAWL_DEPLOYMENT_DESIGN.md §4.4). The game's own tap rules are still unmeasured: its reach, and whether it picks enemies in bushes or off screen. | user |

*Training and experiments*

| # | Item | Owner |
|---|---|---|
| 19 | **Map memorisation.** Consider procedural or augmented maps; the generated families of 2026-09-27 are partial progress. | user |
| 20 | **History.** Two experiments: zero the `hist` group with the same seed and compare held-out win rate; and try another radius or depth than 4 tiles × 3 decisions. The second is config only, but needs a train from scratch. | user |
| 21 | **Bots with gadgets or supers.** Each needs a `brawlers.yaml` block plus a fire decision, since the mechanics are kind-agnostic. | user |
| 22 | **Eval frequency.** Both evaluators run back to back every 20M decisions. | user |
| 23 | **`scripts/benchmark.py`** still draws the attack from {0, 1}. Widen it to `range(cfg.action_nvec[1])` before using it again. | unowned (code) |
| 24 | **Deferred, not owed** (2026-09-21): a fire latch that fires on the first sub-tick the gate opens, its deployment mirror, and a deploy-side cadence baseline. Build the latch only if a measured utilization shows that the decision-boundary cap binds (§5). | user |
| 25 | **The ten generated maps await a look.** Their picks are one reader's judgment of the ASCII renders; the acceptance step is a look at the renders next to the screenshots. The maps README says how to swap one. | user |

## 2. Decisions in force

These are settled. Do not re-open one without new evidence. "User" marks the user's own call; a date alone marks a design decision taken with the user.

| Rule | Why | Settled |
|---|---|---|
| **Engine** | | |
| One `step()` is one decision: `action_repeat` 5 ticks of `dt` 0.05 s (a 20 Hz world, a 4 Hz agent). | The agent commits for a beat, and a simulated second costs fewer transitions. | original design |
| Any `SimParams` field may be a scalar or a `{low, high}` range (optionally multiplicative), resampled per env at every reset. | Variety without code, so the agent cannot overfit one tuning. | original design |
| The sim computes no reward (`ZeroReward`); training installs its own `reward_fn`. | Shaping iterates without touching the sim. | original design |
| The native obs is full-information. Fairness lives only in `obs_select` (`fair: true`), which refuses privileged fields. | Cheating vs fair is one flag, and the refusals keep a misconfigured spec from training a cheater. | original design |
| Bush is the only concealment; walls never block sight. | The game's camera is a fixed bird's-eye view. | original design |
| The native zero-copy `BrawlVecEnv` API stays; SB3 wraps it and never replaces it. | It is the way out to a GPU-native learner. | original design |
| `torch.compile` sits behind `engine.compile`, off; the viewer is matplotlib. | Compile cannot run on this machine (§12); pygame has no cp314 wheel. | original design |
| Speed work stays in repo code: no CUDA graphs, no GPU rollout buffer, no SB3 internals, no edits to installed packages. | | user, 2026-09-30 |
| No synthetic speed benchmarks. Read speed off a real run's `time/fps`, and hunt unbounded memory or storage growth instead. | Past synthetic numbers ignored 4000+ envs and training time. | user, 2026-08; reaffirmed 2026-09-30 |
| **Game fidelity** | | |
| Real game statistics, never rebalanced. | An uneven roster is something to learn, not a bug. Tiers (§11) are a separate difficulty layer. | user, 2026-08-19 |
| Power cubes give +400 max HP flat and +10 % damage each. `default.yaml` caps them at 16; `train.yaml` uses the game's own cap, 99 (Brawl Stars, 2019), which no match reaches. | This is the real mechanic. A fractional HP bonus made the tankiest brawlers tankier. A hero at 16 circled cubes he could not take, with nothing in the obs to say why. | user, 2026-08; cap 2026-10-06 (SIM_ISSUES_PLAN.md §3) |
| Regen follows the official rule: 13 % of max HP per second after 3 s without attacking or being hit. | Disengaging to heal is intended; the gas punishes it. | 2026-08-18 |
| Gas damage is a fraction of each unit's own max HP: 0.20/s, +0.04 per shrink. | Nobody should survive ~5 s in the gas whatever their HP; a flat rate let cube-stacked tanks ignore it. | 2026-08-18 |
| No attack animations. `attack_cooldown` is a window with no attack and no reload; the floor is 0.25 s (one decision), Mortis has 0.35 and Buzz 1.00. | This removed an animation subsystem. | 2026-08-18 |
| Mortis's reload pause during his cooldown is his attack animation. Never measure it or propose removing it (`reload_seconds` refits are fine). | | user, 2026-09-11 |
| One projectile buffer with three classes (PROJECTILE, ARTILLERY, HAZARD). There is no second buffer and no new obs group. | | 2026-08-18 |
| Brock's sphere ticks at 2 s and 4 s. It stacks, never hurts its owner, and ignores walls. | It is burning ground, not a shot. | 2026-08-18 |
| Buzz keeps his first facing for the whole sweep. Each sub-swing has a 0.65 rad arc, so at most 3 of 5 connect on one target, and a sweep costs 1 ammo. | | 2026-08-18; ammo user-confirmed |
| Supers are hero-only, but the machinery is kind-agnostic (`super_charge_hits: 0` means none). | A bot super would be data plus a fire decision. | 2026-08-18 |
| Units never collide; `unit_radius` is a hitbox and sets spawn spacing only. | Soft separation behaved like bumper cars. | original design, revised |
| Raise the projectile slot budget only when a measurement justifies it. The agent's input width never depends on it. | | 2026-08-18 |
| A change to the obs or action width means a new spec file and a train from scratch; no checkpoint is reused across one. Pre-gadget `(17, 3)` checkpoints are refused. | Widening stays free. The user deleted the old runs. | 2026-08-18; user, 2026-09-21 |
| **Hero** | | |
| Attack, super, gadget and auto-aim are values of one attack column: at most one per decision, and the action stays `(N, 2)`. | Choosing one rules out the rest, so no priority order is needed. | user, 2026-09-21 |
| Attack value 4 is an auto-aimed attack, behind `action.auto_aim` (off in `default.yaml`, on in `train.yaml`). | The game's bare tap auto-aims. | user, 2026-09-26 |
| An idle super (move bin 0) aims like the game's tap-to-fire: at the nearest alive enemy within the bolt's reach (11.1 tiles), even off screen, skipping crates; along facing with none. A moving super follows the move bin. The bolt gets no age limit. | Move bin 0's (0, 0) `move_dir` launched a bolt that never moved or expired, a mine that hit each player once for 1800 and healed Mortis. | user, 2026-09-30 |
| The dash grants no i-frames. | It is an attack animation, and Mortis can be hit during it. At elite, i-frames voided 40 % of the combat damage aimed at him. | user, 2026-09-25 |
| A wall stops a dash's momentum and never redirects it (no slide). | | user, 2026-09-25 |
| The gadget has an 18 s cooldown and starts charged. A spinner flies up to 2 tiles toward the nearest enemy the thrower can see (along facing if none), lands after 0.2 s and deals 2000 in a 1-tile radius. | This is the user's spec of the real gadget. | user, 2026-09 |
| No fire latch and no deploy-side cadence baseline. | The policy itself holds fire; the user took reward shaping (`attack_in_reach`) instead. | user, 2026-09-21 |
| `scripts/play_manual.py` gets no super key; `g` throws the gadget. | It is a movement and dash feel-check, and tests cover the rest. | user, 2026-09-21 |
| **Bots** | | |
| Elite bots have at most 1.5× HP and damage; tiers differ by accuracy, aggression and hero focus. | A harder bot should play better, not soak more. | user brief; retuned 2026-09-18 |
| From hard up, bots favour the hero while they can see him (`hero_focus` 0.5 in `brawlers.yaml`, 0.85 at elite). | | user brief, 2026-09 |
| A KITE bot holds no farther away than its own fire reach. | Uncapped, an easy Brock parked out of his own range. | user, 2026-09-21 |
| Bots farm cubes as live lobbies do; there is no clock-based cube grant. | The agent lost live to bots holding 7+ cubes. | user, 2026-09-25 |
| Loot pulls replace the personality's steering. They need no enemy target within 8 tiles (a cube within 2 tiles is exempt) and are off in RETREAT and HOLD_STILL. | Summed pulls froze bots at weighted midpoints. | 2026-09-25 |
| Crates stay walkable. A bot shoots one only within `min(attack_range, 5)` tiles, and a CAMPER never does. | This matches the game: nearby crates, not every crate in range. | user, 2026-09-25 |
| A dash hits each crate once, as it hits each unit (`boxes.dash_hits_once`), and crates have 8000 HP (`boxes.hp`), both in `train.yaml` only. | "Dashes should not be damaging any target more than once." One dash broke a 3000-HP crate; 8000 takes Mortis 4 dashes at 0–3 cubes. | user, 2026-10-06 |
| Speed scales with the length of the intent, and the gas never takes a kill's last hit. | Coasting bots moved on 47 % of HOLD_STILL ticks; the hero lost kills he finished in the gas. | 2026-09-25 |
| Under `bots.nav` (on in `train.yaml`) bots walk shortest walkable paths to their enemy, bush or waypoint and out of the gas (§7), and a KITE with no line of sight closes. Without it they steer straight. | "Bots should get as real pathfinding as possible without compromising training time seriously": pinned on walls in watch.py. With the other bot flags its bot code costs about 3 % of a CUDA env step at 4096 envs (2026-10-07); with train.yaml's full config, the later gas and retreat rules included, an interleaved A/B on an idle machine reads +9.9 % (311 → 342 ms), the rest coming with the harder game, which ends 16 % more episodes per step. | user, 2026-10-06 (was: steer only) |
| Once at most 4 players are alive, the hero included, CAMPERs and TRAPPERs play as HUNTERs (`bots.endgame_players`, 4 in `train.yaml`). | "Campers should switch to hunting in final 4." | user, 2026-10-06 |
| Bots aim at where the target is (`bots.tap_aim`), and read its velocity through a 0.25 s low-pass (`bots.lead_velocity_tau`), both in `train.yaml`. Tier aim noise stays. | The game's bots attack with a tap, whose auto-aim never leads. "Smoothed bot velocity should be solid." | user, 2026-10-06 |
| Under `bots.nav` the gas rules count a bot's room along its way out: its clearance, less how much nearer the gas the centre field's path from its tile comes (`nav.centre_path_dip`). The CAMPER and TRAPPER flee, the gas push's ramp and the bush and waypoint margins all read it. | A pocket's exit closes before the bot's own tile does, and a CAMPER that waited for its own tile was sealed in and died in the gas. "1 tick per bot and bush seems expensive, but I think it's worthwhile." | user, 2026-10-07 |
| Under `bots.nav` a bot backing off, in RETREAT or a HOLD_RANGE too close to its enemy, paths to `policy.retreat_goal`: 6 tiles straight away from the enemy, kept 1 tile inside the safe rect and on the map. | The straight flee pinned on any wall behind it: RETREAT stalled on 12–24 % of its ticks, and on at most 0.4 % after. "Yes, add a destination." | user, 2026-10-07 |
| Under `bots.nav` a strafing bot turns its strafe round when the point 1 tile ahead along it blocks a unit and the point 1 tile behind does not (`state.ent_strafe_sign`, latched until the next wall). | A KITE inside its range band, where only the strafe moves it, pushed into a wall until its enemy moved: HOLD_RANGE stalled on 6.5 % of its ticks, 0.6 % after (GPU smoke, 2026-10-07). "Build this as long as it's relatively cheap": two terrain lookups per bot per tick. | user, 2026-10-07 |
| **Maps** | | |
| The sim has no fences: the loader refuses `f`, but `Tile.FENCE` stays in the enum. | `brawl_vision`'s classifier and labels still index it. | user brief, 2026-09 |
| The generated maps are approved, and the holdouts are `split_river` and `hollow_ring`. | The pair is one standard map and one water-border map. | user, 2026-09-21 (maps added 2026-09-27) |
| **Observation** | | |
| The hero's reveal is the camera window (the ground quad), not the 13×21 rectangle. | The detectors see only the screen; the rectangle drops 23.6 % of on-screen sightings. | user, 2026-09-24 |
| Clamp onsets default to the quad's half-extent minus about 2 tiles of off-map ground. | The game shows ground past the edge, and footage has not measured the onsets (§1 #9). | user, 2026-09-24 |
| `hero.near_edge` means a camera offset over 2 tiles. There is no sim-side dead-bin mask; `policy.dead_bin_mask` is live-only. | Readable on all four sides, which "within N tiles of an edge" is not at the south edge. 2 rather than 1 "to be extra safe". | user, 2026-09-24 |
| Off-screen projectiles keep their ranked slot with the row blanked, and there is no dropout augmentation. | The empty rows are noise in the direction of the live detector's misses. | user, 2026-09-24 |
| Tied projectiles go nearest the hero first. | `topk`'s tie order came from the sim's slot layout, which deployment cannot reproduce. | user, 2026-09-30 |
| Under `observation.projectile_static_speed` (2.0 in `train.yaml`, 0 = off in `default.yaml`) a projectile slower than 2.0 tiles/s reads as still in the agent's view, `vel` 0 and `time_to_closest` 0, before `max_slots` ranks it. A test pins the key to the live tracker's `STATIC_TILES_S`; `full_obs` keeps the true values. | It is how the live tracker reports one. Grom's shell lands 1.25 s after the throw, so a short lob crawled, and its straight-line time, unscaled, read up to 34 s in the agent's view (CPU probe) and 1226 s at a GPU smoke's peak. On screen the time now stays under the farthest visible point over 2 tiles/s, 9.2 s mid-map and 15.8 s in a corner, as it does live. "#32: go with option C". | user, 2026-10-07 |
| Enemy slots follow the live tracker's promote, coast and reuse rule, and the live tracker takes `slots.*` from the run's config. | Permanent slots let the policy learn "slot 3 is the one I hurt". A tracker on its own defaults would desync silently under a `slots.*` override. | user, 2026-09-24 and 2026-09-30 |
| `zone.active` latches "gas has been on screen", tested on the window's bounding box; there is no per-side latch. Live, the latch outlives odometry segments and clears only at a match start. | It is what `ZoneEstimator.active` supplies. The box over-counts only two corners. A new segment empties the live gas map, so the map alone would un-latch. | user, 2026-09-24 and 2026-09-30 |
| HUD-hole terrain, the bush constants and a live tracker without a window stay as they are. | Live reads are unmasked, the occupancy map is world-frame memory, and live the screen is the window. | user, 2026-09-24 |
| Fix the player-box bias live-only (`PLAYER_BOX_FROM_RING_TILES`). Never change `HERO_ANCHOR_TILES`. | `camera.quad` derives from it; changing it moves the sim's reveal. | user, 2026-09-25 |
| Three zone-observation mismatches (§9) are recorded, not fixed. | | user, 2026-09-25 |
| History covers the last 3 decisions within 4 tiles. | The user's brief; both numbers are config (§1 #20). | user brief, 2026-09 |
| **Match flow and training** | | |
| Gas is timed from the game and jittered ±10 % per env in training. | Learn the gas, not one clock. | user, 2026-09-24 |
| Gas runs at 1.3× the game's pace in a 185 s episode. This is set in `train.yaml`, never `default.yaml`. | A bit shorter than a real match. Runs name `default.yaml` by path, so it would change under them silently. | user, 2026-09-25 |
| The curriculum keeps its forced advance (`max_timesteps_at_stage`). | A run always reaches elite, even when no gate clears. | user, 2026-09-25 |
| `reward.attack_in_reach` 0.05. | "A small reward for attacking when in range". | user, 2026-09-21 |
| `reward.gadget_hit` 0.3, paid per landed gadget that hurts a player; crates alone never count. | "A bonus rwd for the agent if it uses the gadget on another player (not a crate)". | user, 2026-09-30 |
| `reward.move_reversal` −0.01, charged per decision whose move bin swung 135° or more from the previous decision's, both non-idle. | The policy reversed on about a quarter of its decisions, as often with no enemy in view as with one near. "Add the 0.01 reversal penalty." | user, 2026-10-07 |
| `n_steps` stays 128 and `gae_lambda` 0.98. Tune the horizon through gamma, critic quality and LR floors, never through shorter rollouts or a lower lambda. | The user wants planning from the start of a match to its end. | user, 2026-09-26 |

**Accepted divergences from the game:**
- No Star Powers or Hypercharges; only Mortis has a super and a gadget.
- 16 move bins.
- Units pass through each other, and there is no knockback, stun or slow.
- Only attacking reveals a bush-hider; taking damage does not.
- The smallest non-zero latency is one 50 ms tick.
- No cube redistribution and no timed cube spawns.
- Bots are heuristic, so their parameters are randomized.
- Crates have a flat HP (3000 in `default.yaml`, 8000 in `train.yaml`), where the real game's 4500–8500 scales with lobby power.
- A bot that dies mid-dash still finishes it: the rest of the dash (at most 0.3 s) can hit a crate, though never a unit. "#31 isn't an issue, no fix needed" (user, 2026-10-07).

## 3. Architecture

**Packages.**
- `brawl_sim/{constants,config,env}.py`.
- `core/`: state, geometry, terrain, stats, hero, movement, melee_sweep, projectiles, combat, boxes, zone, spawn, camera, slots, history, observation, obs_schema, obs_select, events, reward.
- `bots/`: perception, steering, combat_rules, personality, policy.
- `maps/`: loader, generate, nav (bot pathfinding, §7), `csv/`.
- `wrappers/`: sb3_vecenv, gym_single, sb3_features, episode_stats.
- `render/`: ascii, viewer.
- `training/`.

**Import boundary.** `core/`, `wrappers/` and `training/` never import `bots/`; `env.py` is the only module that imports both `core/` and `bots/`. `core/camera.py` lives in `core` because observation, history, zone and env all need it. `training/` holds everything the sim deliberately leaves out: reward, policy architecture, runs and hyperparameters.

**Config views.**
- `EnvConfig`: static Python values (shapes, toggles, episode length, map list), never randomized.
- `SimParams`: per-env tensors from `brawlers.yaml` and `default.yaml`, resampled on reset. Its `SCALAR_FIELDS` (`cone_ray_tiles`, `attack_ray_tiles`, `dash_ray_tiles`, `shot_step_tiles`, `box_reach_tiles`) are Python floats derived once from the spec's upper bounds. They size tensors and drive control flow, so reading them off the device would sync inside autoreset.
- `AgentObsSpec`: what the agent sees (§9).

**Overlays and overrides.**
- `configs/randomization.yaml` is applied once at construction, in training only.
- `configs/presets/*.yaml` are override fragments.
- A `SimParams`-only override is invisible to `EnvConfig`, so `training/builder.build_env` merges `run.env_overrides` into both the `EnvConfig` and the `spec=` dict.
- **Rename, never reinterpret:** `config._REMOVED_KEYS` refuses keys whose unit changed (`zone.dps`, `regen.per_second`, `cubes.hp_bonus_per_cube`), because a stale key silently resolves to 0 and disables a mechanic.

**Per-kind data, generic code.** A kind that lacks a mechanic sets its field to 0 and pays only a masked `torch.where` (`hitscan_count <= 1`, `split_count 0`, `super_charge_hits 0`, `gadget_cooldown 0`). Prefer widening an existing structure to adding a buffer.

**Compute, then commit.** Physics steps return damage and heal totals, and `combat.apply_damage` and `apply_heal` own the rule that a dead entity is never touched. For example, `step_projectiles` returns `(dmg_ent, dmg_by, dmg_box, heal_ent, charge_hit)` and never writes `ent_hp`.

**SimState** is one preallocated struct-of-arrays, declared in `core/state.py` as `(name, shape over N,E,P,B,U,L,K, dtype)` groups. It is allocated once and reset by `torch.where`; `check_invariants` runs only under `debug_checks`.

**Vectorization rules beyond CONVENTIONS.md.**
- Never materialize a per-env (N,H,W) map slice; gather `MapBank`'s (M,H,W) tensors by `map_id`.
- The only per-env grids are obs grids. They are scattered into a zeroed buffer from a WALL-padded bank, so crops need no bounds checks.
- Constant tables are cached per device.
- March and ray lengths are fixed Python budgets. `terrain.march(..., max_tiles=)` takes `ceil(max_tiles / los_step_tiles)` samples plus the endpoint. A budget shorter than an element's own distance changes that element's answer, so it is legal only where the caller discards every such element.

**`step()`** runs these stages in order:
1. `history.push` records the pre-step state, this action, and the reveal it answered.
2. `_run_decision` runs `_run_tick` `action_repeat` times. Ticks 2..K hold the move bin and zero the whole attack column, so a decision is at most one attack attempt, which matches the mask the policy saw. Event deltas are summed and outcomes latch (`events.advance_decision_tally`). The phases of a tick are CONVENTIONS.md's tick order. Attack before movement is what makes the dash replace the walk, and deaths after every damage source are what make simultaneous kills work.
3. `_observe` computes done and info, and calls `reward_fn` once.
4. Finished envs' final obs and info are cloned, then `_autoreset` resets those envs in place and rebuilds their obs.

**Obs build**, once per decision (twice for an env that resets): visibility, then `camera.hero_view`, then `slots.update`, then `zone.mark_seen`, then `raw_los` (if read), then `build_obs`. `step` hands that `hero_view` to the next `history.push`.

**Step contract.**
- For a finished env, `obs` is the new episode's first observation, and the old episode's last one is in `info["final_observation"]` / `info["final_info"]`.
- `reward`, `terminated`, `truncated` and `info` describe what just happened. With `autoreset=False` (gym_single), the terminal obs comes back and the caller resets.
- Returned obs and info are zero-copy views into SimState, valid until the next `step()`; `observation.clone_obs` keeps one.
- `tick_hook` is a debug callback after every sub-tick, used by `watch.py` and `record_rollout.py` for 20 Hz frames. Training never installs one.
- `step(action, override)` can drive any entity's movement and ordinary attack; −1 in column 0 means "not overridden".
- Latency is set in seconds: 0.001 s rounds to 0 ticks, and the ring buffer passes through.
- One device `torch.Generator` makes runs reproducible for a fixed (seed, n_envs, device, actions).

**Wrappers.**
- `BrawlSB3VecEnv` takes over `reward_fn`, provides `action_masks()`, and sets `terminal_observation`, `TimeLimit.truncated`, `episode`, `outcome{rank, won}` and `episode_stats`. `info_mode` (minimal, episode or full) sets how much it copies.
- `BrawlGymEnv` requires n_envs = 1, and raises on a step after done.
- Host transfers happen only in these two wrappers (pinned buffers, one sync per step, one transfer per obs group). Training syncs once per rollout.

**Tools.**
- `scripts/watch.py`, `record_rollout.py`, `play_manual.py`, `dump_obs_schema.py`, `gen_maps.py`, `audit_attack_cadence.py`, `sb3_smoke.py`.
- `smoke_test.py` runs every config and preset under `debug_checks`.
- `benchmark.py` is in no workflow.
- `tests/test_integration.py` guards determinism: the same seed and actions give bit-identical runs, and perturbing env 0 leaves every other env bit-identical. It also guards the pre-reset `final_observation`, range resampling, the per-field privileged refusals, and a sync-free native step on CUDA.

## 4. The hero (Mortis)

**Stats** (`hero_mortis`):

| Stat | Value |
|---|---|
| HP | 8000 |
| Damage | 2000 |
| Speed | 2.73 tiles/s |
| Ammo | 3 |
| Reload | 2.50 s (measured off Nulls Brawl, `brawl_deployment/data/kit_timing.json`) |
| Attack cooldown | 0.35 s |

**Action.** The action is `(N, 2)` int64:
- Column 0 is the move bin: 0 = idle, 1..16 = directions.
- Column 1 is the attack value: 0 none, 1 attack, 2 super, 3 gadget, and 4 auto-aimed attack (only under `action.auto_aim`, which is on in `train.yaml` only).

`cfg.action_nvec` is (17, 4), or (17, 5) with auto-aim. The mask is `{"move": (N,17) all legal, "attack": (N,4|5)}`, with columns `[none, attack, super, gadget(, attack)]`. `hero.gadget_ready` is the one definition shared by the mask and the attack phase.

**Legality.**
- Attack and auto-aim need: alive, `attack_cd` ≤ 0, `dash_t` ≤ 0, and ammo ≥ 1.
- The super needs the same, with "charged" in place of ammo.
- The gadget needs only alive and `gadget_cd` ≤ 0. It is legal mid-dash, during the cooldown and on an empty clip.
- Illegal values are silent no-ops: decode gates them, and the attack phase gates them again.

**Dash (the attack).**
- It goes 2.67 tiles in 0.30 s with a 0.70 hit radius, along the decision's move bin, or along facing when idle (`action.dash_on_idle: facing`).
- It replaces that decision's walk and cannot be steered.
- `attack_cooldown` 0.35 must exceed `dash_duration` 0.30, or the dash would mask the cooldown.
- Each enemy is hit once per dash (`dash_hits`). Crates are hit on every tick, or once per dash under `boxes.dash_hits_once` (`ent_dash_box_hits`, on in `train.yaml`).
- `terrain.body_travel` stops the body at its first wall contact, and the last tick advances only the remaining time.

**Long dash.** After 4.5 s without attacking, the next dash goes twice as far in the same unscaled 0.30 s (5.34 tiles at 17.8 tiles/s). Only an attack or a super resets the stopwatch (`attack_idle_t`); damage and the gadget do not.

**Super.**
- **Charging:** +1 per player hit per tick, from dash, melee or projectile hits, up to 5. Crate hits and the gadget never charge it; super hits count toward the next charge.
- **Firing:** zeroes the charge, takes `attack_cooldown`, and launches one piercing bolt (`projectiles.spawn_supers`).
- **Aim:** along the move bin. On an idle bin it aims like the game's tap-to-fire (`hero.super_aim_target`): at the nearest alive enemy within the bolt's reach, `super_range` + `super_radius` + `unit_radius` = 11.1 tiles, even off screen, and along facing with none. It skips crates, which the bolt passes through, and the long dash does not stretch it. `env._attack_phase` builds the aim as its own `super_dir`, as the dash has `dash_dir`, so no firing row hands `spawn_supers` a zero aim.
- **The bolt:** it flies 10 tiles through walls and past its first victim, hitting each player once (`prj_hits`) for a flat 1800 and healing Mortis 1800 per player hit. It ignores crates.

**Gadget.**
- **Throwing:** one ARTILLERY `GADGET_SPINNER` (`hero.gadget_target`, then `projectiles.spawn_gadget`) flies `min(2.0, distance)` toward the nearest enemy in the thrower's concealment visibility, or along facing when none is visible or the target stands on him.
- **Travel:** it is pre-clipped against walls with `step_projectiles`' march, because artillery is not stopped in flight. A 1e-4 velocity surplus makes it pass its landing point on a deterministic tick.
- **Blast:** after 0.2 s it deals 2000 (+10 % per cube) within radius 1.0 to enemies and crates. It never hurts its owner and never charges the super.
- **Side effects:** a throw sets `gadget_cd` 18.0 and nothing else. It spends no ammo, starts no cooldown, leaves the long-dash stopwatch alone, and is not an attack for `attack_in_reach`; a landing that hurts a player pays `gadget_hit` instead (§10). It does reveal Mortis and break out-of-combat regen.
- **Timing:** the earliest second throw is decision 73, 18.25 s after the first. `tests/test_gadget.py` pins this, and the deployment shadow must match it.

**Auto-aim (value 4).**
- It dashes at the nearest alive enemy or unbroken crate within the current dash reach (long dash included, plus `dash_radius` and the target's body), even off screen, like the game's tap.
- With nothing in reach, it follows the move bin or facing.
- It is legal exactly when value 1 is, and it counts as `fire` everywhere downstream. It bends the dash, never the walk.

**Reveal and regen.** Every offensive action (attack, super, gadget) sets `reveal_t` (`perception.reveal_after_attack` 1.0 s) and resets `out_of_combat_t`. Without that, a super could be fired from a bush for free, or a hero could out-heal a fight.

**No i-frames.** Nothing seeds `ent_invuln_t`. The i-frame masks in `core/combat.py` and `zone.iframes_block_zone` are dormant, and `hero.invuln` keeps its obs slot, always reading False.

## 5. Combat

**Damage and credit.**
- `apply_damage` runs once per source (dash, melee, projectiles, gas).
- `dominant_attacker` credits a victim's tick to whoever dealt it the most.
- `last_hit_by` is written only when damage lands on a unit that is alive with HP > 0, so the gas (phase 10) cannot steal a finisher's credit. A death's cause is ZONE only when `last_hit_by` < 0.

**Decision-boundary cap.** The mask is read only at decision boundaries. After a dash, the next legal attack is 0.50 s later (10 ticks), not 0.35 s: 70 % of the weapon's rate. The cadence audit's `phasing_loss` measures this; the deferred latch (§1 #24) would remove it.

**Reload** accrues only while `attack_cd` ≤ 0, so sustained fire costs `attack_cooldown + reload_seconds` per shot.

**Melee.**
- **Hit test:** a cone plus wall LOS (budget `cone_ray_tiles`). All melee resolves in one `melee_hitscan` pass, with a per-entity cone direction; lifesteal applies after damage.
- **Edgar's lifesteal:** 35 % of damage dealt. It counts entities only (crates are excluded), and overkill counts.
- **Swept melee (Buzz, `core/melee_sweep`):** `hitscan_count` > 1 spreads that many cones across `attack_cooldown`, sweeping clockwise on screen. Sub-swing k is centred at `facing - sweep/2 + k*sweep/(count-1)`. The clock is `ent_attack_cd`. Movement freezes facing for swept kinds only while `attack_cd` > 0. A whole sweep costs 1 ammo, not 1 per sub-swing (user-confirmed).
- **Stateless schedules:** sub-swings and hazard ticks both fire on floor crossings of an age or cooldown, so nothing needs keeping in sync or resetting, and no tick is dropped or doubled when `dt` doesn't divide the interval.

**Projectile classes** (`prj_class`).
- **PROJECTILE** travels and dies on its first wall, unit or crate. The piercing super is the exception.
- **ARTILLERY** does no damage in flight and resolves at its landing point: a blast only when `prj_aoe` > 0, with no LOS test.
- **HAZARD** does not move and never dies on contact. It damages everyone within `prj_aoe` on a schedule.

Only hazards and the gadget spinner spare their owner; artillery can hurt its thrower. Timed lobs (`proj_flight_seconds`) land after a fixed time at any distance, with aim points clamped into the map.

**`step_projectiles` order:**
1. Integrate.
2. Wall march within `shot_step_tiles` plus an endpoint sample, so slow shots cannot tunnel. Piercing shots pass walls.
3. Units: the earliest hit wins; a pierce hits each victim once.
4. Crates, through the nearby-box grid (§12). A pierce ignores crates.
5. Artillery detonates on its stored target or at range end, and hazards tick.
6. Range expiry.
7. Split shards from detonating shells.
8. A lingering hazard wherever a Brock rocket died, whether on a wall, unit or crate, or at range end.

**Brock's sphere** keeps the rocket's `Proj` kind, so the obs and the renderer credit the weapon. It deals `on_hit_area_damage_fraction` (0.30) × the rocket's damage, so it scales with cubes and tier.

**Splits (Grom, Spike).** A detonating shell hands `split_count` shards (at most `MAX_SPLITS` 6) to free slots. They are ordinary projectiles that fly an even ring for `split_distance`, each carrying `split_damage_fraction` of the shell. Stats come from the owner's kind, so a dead shooter's shell still splits. `aoe_radius: 0` means no landing blast (Spike); the `prj_aoe > 0` gate matters because a noiseless lob lands exactly on a stationary target.

**Projectile slots.** Allocation is a `cumsum` plus `searchsorted`, and a full buffer drops the shot silently. `validate()` therefore requires `limits.max_projectiles` ≥ `config.peak_projectile_demand`: the hero plus `n_enemies` × the worst kind's volleys × pellets × splits over a projectile's residency. That is 162 today (Spike-bound), against 192 slots. The bound ignores reloads during a long residency, so it is not strictly tight.

**Cubes and healing.**
- A pickup heals by the max-HP increase. `enemy_hp_mult` and `enemy_damage_mult` scale the totals.
- Regen (0.13 × max HP per second, after 3.0 s) runs in tick phase 2, before this tick's attack resets its stopwatch, so it never undoes a fatal hit landed later in the tick.
- `hp_healed` = regen + super lifesteal + melee lifesteal, counted as HP actually restored. The cube max-HP bump is excluded.

## 6. Terrain, movement and maps

**Tiles.** FLOOR, WALL, BUSH, WATER, FENCE, SPAWN, BOX.

| Blocks | Tiles |
|---|---|
| Units | WALL, WATER (FENCE too, but the sim has none) |
| Shots | WALL only |
| Sight | nothing |

`line_of_sight` checks walls only, for four users: the melee hit test, the bots' `fire_needs_los` gate, the loot-crate shot, and the `raw_los` observation.

**Terrain functions.**
- `circle_blocked` uses 8 probes.
- `resolve_move` is an axis-separated slide.
- `body_travel` tests `circle_blocked` every `los_step_tiles` within `dash_ray_tiles` and bisects 4 times inside the failing step. It backs off 1e-3 from the contact. A body that starts overlapping a wall stays put.
- A rare corner clip between samples lands clear and is accepted.

**Walking** moves alive, non-dashing units by `resolve_move`, with throttle = |intent| clamped to 1, so a bot's decaying intent slows it instead of letting it coast. Units never collide; crates are walkable and block no unit.

**Map validation at load.**
- Maps are 60×60 with an all-WALL border.
- Each map has 8–32 SPAWN and 8–64 BOX markers, and one connected region units can pass through.
- Every pool map has at least 10 spawns, which the 10 entities need; with fewer, players share spawns.
- CSV characters: `.` floor, `#` wall, `b` bush, `~` water, `S` spawn, `X` crate. `f` is refused.

**`MapBank`** holds (M,H,W) tile tensors, lookup tables, and WALL-padded copies for crops. Spawns are sorted by angle from the map centre. Bush waypoints are one per `hunt_cell_tiles` (10) cell, at most 63 per map, so the visited set fits an int64 bitmask.

**Map pool.**
- `brawl_sim/maps/csv/` holds 38 CSVs. `default.yaml` lists 36; `walled` (no bush) and `blank` are test fixtures.
- The 36 are 6 hand-authored maps, 10 generated in four families, 5 transcribed from screenshots, and 15 generated from five screenshot families.
- `train.yaml` trains on 34, which is the 36 minus the holdouts. Its `run.env_overrides.world.maps` replaces the base list, so a map added to `default.yaml` trains only once `train.yaml` lists it too.
- Every name must be in `config._KNOWN_MAP_NAMES`.
- The obs carries no map id, so a checkpoint can resume or evaluate on a different pool.

**Generator** (`maps/generate.py`, CLI `scripts/gen_maps.py`). It is deterministic in (seed, family, symmetry), and `tests/test_maps.py` regenerates every shipped generated map byte for byte. Nine `FAMILIES` set wall, bush and water bands and a crate count. The first ten generated maps are 8 point-symmetric and 2 mirrored, so "the far corner is mine rotated" can't be learned; the fifteen screenshot-family maps all mirror.

Rules whose reason isn't obvious:
- Solid stamps keep a 2-cell floor margin, so no 1-wide corridor forms.
- Repair runs before spawns and crates: it gaps walls longer than 8, fills 1-wide dead ends and connects components.
- **Pocket check:** every floor cell must reach at least 24 of an open field's 85 cells within BFS radius 6 (0.28, scaled by in-bounds cells). 0.28 still rejects a 2×3 notch or a dead-end lane, but not a 2-wide lane.
- Water-border moats sit 4 tiles in, behind a 3-wide rim; flush against the border, their gaps led nowhere.
- Bush must cover at least 80 % of the hunt cells.

Seeds and per-map notes are in `brawl_sim/maps/README.md`.

## 7. Bots

**Roster.** There are 7 bot kinds plus the hero (`N_KINDS` = 8). Kinds are named by role but grounded on real brawlers. A kind is pure data: one `brawlers.yaml` block read by `bots/combat_rules.py`.

| Kind | Brawler | Attack |
|---|---|---|
| sniper | Brock | LEAD-aimed rocket; where it dies, a burning sphere (0.75 tiles) ticks at 2 s and 4 s |
| artillery | Grom | LOB shell over walls (1.25 s flight), 0.6-tile landing blast, 4 shards; 7.33-tile range is an assumption |
| melee | Buzz | 5-cone sweep over his 1.0 s cooldown |
| rifle | Shelly | 5-pellet fan (0.6 rad), fires within 0.9 of range |
| edgar | Edgar | two-hit melee combo, 35 % lifesteal |
| spike | Spike | LOB-aimed shell that needs LOS and bursts into 6 full-damage spikes, no landing blast |
| bull | Bull | 5-pellet fan (0.5 rad), 7-tile range, fires within 0.9 of range |

Stat conversions from the game's data: Power 11 = 2 × Power 1; `Speed`/300 = tiles/s; `CastingRange`/3 = tiles; `RechargeTime`/1000 = s. If `brawlers.yaml` and `CHARACTER_DETAILS.md` disagree, CHARACTER_DETAILS.md wins. No bot has a super or a gadget (`super_charge_hits` and `gadget_cooldown` are 0, and `BotIntent.super_fire` is always False).

**Personalities** are drawn per entity per reset, independently of kind. Weights: rush .28, camper .12, hunter .24, trapper .16, kite .20.

| Personality | Behaviour |
|---|---|
| RUSH | Closes on anything visible. |
| CAMPER | Holds a bush and fires only once something can see it (the veto lifts at aggression ≥ 1.25). Leaves only when the gas is within `camper_zone_flee_tiles`. |
| HUNTER | Sweeps bush waypoints, one per hunt cell. A leg is abandoned after `hunt_timeout_seconds`, and the mask resets once every waypoint is searched. |
| TRAPPER | Holds a bush and fires freely, moving to an unsat bush when idle. |
| KITE | Holds range in the open, capped at fire reach. Under `bots.nav` it closes while it has no line of sight, backs off along a path, and turns its strafe round at a wall (below). |

- A bush personality with no safe bush behaves as RUSH.
- **Endgame** (`bots.endgame_players`, 0 = off): once at most that many players are alive, the hero included, a CAMPER or TRAPPER plays as a HUNTER, in its mode, its targeting and the fire veto (`policy.effective_person`).
- Only HUNTER and KITE retreat, below `0.35 / aggression` HP (clamped to 0.05–0.90). Under `bots.nav` a retreat paths to a goal (below).
- Idle bots wander: each holds a random heading for `wander_seconds` (±50 %), re-rolled when a 2.5-tile probe hits a wall or the gas.
- `min_aggressive: 1` forces at least one RUSH or HUNTER per lobby at a random slot, so no lobby can be beaten by standing still and slot order stays unreadable.

**`all_bot_intents`**, every sub-tick:
1. Sight-limited visibility (`bot_visibility`): bush concealment within `bots.sight_tiles` 14. Bots are not camera-limited. Uncapped, 84 % of bot-ticks held a target at a mean 22.7 tiles.
2. `select_target`: targets are sticky. The hero's distance is scaled by `1 - hero_focus`, and if he wins, the bot switches even from a valid sticky target.
3. `target_los`: target-only (N,E) wall LOS.
4. `targeting`: a crate becomes the target when no enemy target is within attack range and the crate is within `min(attack_range, 5)` tiles, never for a CAMPER. The desired range is `min(desired_range_fraction × range × clamp(1/aggression, 0.6, 1.4), fire reach)`.
5. `combat_rules.combat`: `fire = fire_gate & los_ok & range_ok & cone_ok & ~holds_fire`.
6. `personality.movement`, with loot pulls and gas avoidance (below).
7. The CAMPER fire veto.
8. `decision_period_ticks` gates fire only, staggered by slot (`(step + e) % period == 0`). `reaction_delay` is an EMA on movement only (`ent_move_smooth`).
9. The hero row and dead bots are zeroed.

**Fire rule details.**
- `fire_needs_los` applies to straight shots. Lobbed shells arc over walls, and melee runs its own LOS.
- `fire_range_fraction` (0 read as 1.0) is 0.9 for Shelly and Bull, because the fan is unreliable at its rim.
- **Lateral hold:** a bot holds fire on a target crossing faster than `fire_lateral_speed_limit` beyond `fire_lateral_hold_range_fraction` of range.
- **Cone gate:** applies only when `attack_arc_rad` > 0. Its half-angle is `(hitscan_sweep_rad + attack_arc_rad) / 2`, the whole sweep's reach. It reads pre-movement facing, so turning costs a tick or two.
- **Aim models:**
  - DIRECT: straight at the target (Buzz, Edgar).
  - LEAD: intercept × `lead_target_fraction`, plus angular noise (Brock, Shelly, Bull).
  - LOB: the closed-form timed lead `pos + vel × lead × flight`, plus positional noise in tiles (Grom, Spike).

  Tier `aim_noise` scales both kinds of noise. Under `bots.tap_aim` the lead is 0, so LEAD and LOB aim at the target's current position, as the game's tap auto-aim does; the noise stays.
- **The velocity a bot reads** (the lead, and the lateral hold) is `ent_vel`, or under `bots.lead_velocity_tau` > 0 `ent_vel_seen`: a first-order low-pass of it with that time constant, updated after the dash phase.

**Aggression** (`stats.aggression_of` reads 0 as 1.0) is read in three places: the retreat threshold, the KITE hold scale, and the CAMPER veto.

**Steering.** Primitives return direction vectors, which the weights balance; `combine()` normalizes once at the end. Gas avoidance is a predictive inward push that starts `zone_avoid_tiles` (4) from the safe edge, before any damage. No loot pull silences it.

**Pathfinding** (`bots.nav`, `maps/nav.py`). Built once per map bank on CPU: about 9 s and 52 MB for 36 maps, cached per map for the process.
- One anchor per `nav_anchor_tiles` (3) square cell, at its walkable tile nearest the cell centre, and one flow field per anchor: the first step of a shortest 8-connected path from every tile. A diagonal step needs its whole 2×2 block walkable, and ties go to the step nearest the straight bearing, so open ground is crossed on the straight line. One more field leads to the map centre, where `core/zone.py` closes the gas.
- A clear straight walk is the BODY's: `nav.walk_blocked` tests the two lines `unit_radius` to either side of the centre line, which covers every tile the body crosses while the radius is under half a tile (`config.validate`). Each line is tested exactly (`nav.segment_blocked`: the tile past every grid line it crosses), so a tail of a clear walk is clear; a march, sampled every half tile, read one line clear from one spot and blocked from a spot a step further along it. A centre line passes wall corners the body cannot, and a push along a wall face gives `resolve_move` no axis to slide on. Measured 2026-10-07 under the centre-line test: a bot pulled to a crate past a water corner ground there for 26 s, and a CAMPER closing on an enemy past a wall corner stood in the gas until it died.
- A goal reads the field of its anchor: the path-nearest anchor among its own and the 8 neighbouring cells from whose centre the body walks straight to the goal tile's centre. A walkable tile with none (an orphan: 56 on the 36 maps, at most 6 on one, bush_halo) is made an anchor itself, with its own field, so the table's slots grow from 401 to 405; a wall tile takes the nearest anchor by straight distance. So every walkable goal tile has a clear body walk from its anchor. Measured 2026-10-07: choosing by the centre line left 3,105 of the 102,935 walkable tiles with that walk blocked, and by the body's sampled walk 24, which fell back to a clear centre line.
- `policy.path_toward` aims, the first that applies: (1) at the goal, within 8 tiles with a clear straight walk; (2) at the goal tile's centre, on the same terms; (3) along the field; (4) on the field's last tile, its anchor's (`nav.at_anchor`), at that tile's centre; (5) at the goal, where no field leads (another component, or a goal in a wall). Each is scaled to the straight distance, so the steering weights are unchanged. The anchor rule makes 4 hand over to 2 and 2 to 1, and a bot on 1 or 2 stays on it as it walks the line, since a tail of a clear walk is clear. Measured 2026-10-07 with the goal walk alone and the sampled test: a body following `path_toward` from 9 or more tiles out never reached 59 of the 102,935 walkable goal tiles, nor 3 of the 22,424 bush tiles. Each bot cycled a tile or two short of its anchor, the field leading it back and the straight walk out again, every tick, for good; an Edgar chasing a hero who stood at three such spots never came within his 2-tile range. With the chain: 0 and 0, and `tests/test_nav.py` walks a body to every walkable tile of two maps (it fails on the old code). One goal per bot per tick, chosen by mode (enemy, bush or hunt waypoint): one walk test of four lines and four gathers. On the training device (CUDA, train.yaml's 4096 envs, interleaved blocks) the chain costs 11.8 ms per env step, +2.8 % (423.6 → 435.4 ms), and episodes end at the same rate.
- The field is followed by the body (`nav.step_dir` at `unit_radius`). A body overhanging its tile toward a blocked tile it could clip on the next step first moves to its tile's centre line across that edge; that move only re-enters tiles it already touches. Aiming straight at the next tile's centre, `resolve_move` (an axis's whole step or none) stopped the bot one step short of the corner. Measured 2026-10-07: a bot closing past a water corner stood there 20.6 s, its strafe cancelling the small correction.
- The gas terms re-aim along the centre field, keeping their length, while the safe rect is centred on the map. In a pocket, where the field's path leaves the rect (`nav.centre_path_inside`, from a per-tile bounding box of that path), the inward push is off and the bot's own steering moves it. Measured 2026-10-07, a CAMPER in a pocket at the rect's edge died both ways the push was tried. Following the field, the push led it into the gas, where the escape (scaled by depth) lost to its pull back to its bush, and it crossed the edge until the gas killed it. Aimed straight at the rect's centre, the push (about 40 against a seek's 12) held it against the pocket's inner wall for 7 s, until the gas arrived. The loot pulls keep their straight line, gated by the body's walk.
- The gas rules measure a bot's room along its way out (user, 2026-10-07): its clearance less `nav.centre_path_dip`, how much nearer the rect's edge the centre field's path from its tile comes than the tile's own centre, read off the same per-tile box. The CAMPER and TRAPPER flee (`policy.zone_clearance`), the gas push's ramp, and `bush_scan`'s and `hunt_waypoint`'s margins subtract it, so a bot leaves a pocket while its exit is still open, and never picks a bush or waypoint whose exit is closing. The dip is exactly 0 wherever the path comes no nearer the edge than its first tile (all open ground, since a walk to the centre only moves inward) and where the rect is off the map's centre. The probe's one pocket-camper gas death (seed 0, training mode) is gone. Each tick it gathers the box twice per bot (the flee's clearance, the push's ramp), once per tile of each bot's 49-tile bush scan, and once per map waypoint per env.
- A bot backing off, in RETREAT or in a HOLD_RANGE too close to its enemy, paths to `policy.retreat_goal` (user, 2026-10-07): 6 tiles straight away from the enemy, held 1 tile inside an active safe rect (a rect narrower than 2 tiles keeps its centre line) and on the map. It is the bot's one `path_toward` goal for the tick, and the flee keeps its straight length, so the weights are unchanged. With the straight walk clear it is the old flee's direction. In scripted scenes a straight wall behind the bot did not pin it (the strafe slid it along); a concave pocket did. Probe, 2026-10-07: RETREAT stalled on 0–0.4 % of its ticks, against 12–24 % before, and HOLD_RANGE's backing off on 0 %.
- A strafing bot (one with an enemy, in a mode whose strafe weight is non-zero: CLOSE, HOLD_RANGE, RETREAT) turns its strafe round when the point 1 tile ahead along the strafe blocks a unit, off the map included, and the point 1 tile behind does not (`personality.advance_strafe`; user, 2026-10-07). The sense is latched in `state.ent_strafe_sign` until the next wall; 0, its reset value, reads as the slot's `policy.strafe_sign`, so on open ground a bot strafes as it did without the turn. The open-behind test stops a bot boxed in on both sides from flipping every tick. It costs two terrain lookups per bot per tick. GPU smoke, 2026-10-07, 1536 envs × 1600 decisions, the same run with the turn off: stalled share of the mode's ticks HOLD_RANGE 0.57 % against 6.49 %, CLOSE 0.14 % against 1.32 %, RETREAT 0.07 % against 0.37 %, every bot 1.00 % against 1.81 %; WANDER (3–3.5 %, no enemy) is now most of what is left. Time per decision did not change beyond noise.

**Loot pulls** (`bots/policy.py`).
- A crate within 20 tiles pulls at weight 3.0, and a cube within 12 tiles at weight 4.0. Each is a unit direction that replaces the personality's steering while active; only the gas terms compete.
- Both need no enemy target within 8 tiles (a cube within 2 tiles is exempt, since kill drops land mid-fight), a clear straight walk (`_walk_clear`: the centre line, or the body's under `bots.nav`), and a spot at least 1 tile inside the safe area.
- A cube pull cancels the crate pull, both are off in RETREAT, and CAMPERs never get the crate pull.
- These radii and weights make live-like farming: the richest elite bot holds 7+ cubes at 60 s in more than half the matches. The measurements are in BRAWL_DEPLOYMENT_DESIGN.md §9 entries 21–22, and the probes are `scripts/probes/cube_economy_measure.py` and `bot_pull_measure.py`. Re-measured 2026-10-07 with configs/train.yaml's new keys on (SIM_ISSUES_PLAN.md: 8000-HP crates hit once per dash, pathfinding, tap-aimed bots), on the 2026-09-30 checkpoint, elite tier, 64 matches. The target still holds, and earlier: a bot held 7+ at 60 s in 5 of the 6 matches still running (12 of 23 before), and at 45 s in 10 of 11. Paths get bots to their crates: pulled decisions stalled more than 1.5 tiles out fell from 32 % to 6 %. The sample is small because that checkpoint now dies by a median 18 s.

**Toggles:** `bots.break_boxes` (the crate pull), `attack_boxes` (the crate target), `collect_cubes`, `avoid_zone`, `personalities`, `nav`, `tap_aim`, and the numbers `endgame_players` and `lead_velocity_tau` (0 is off). Every lobby is free-for-all.

## 8. Match flow

**Reset** (`spawn.reset_envs`) runs in this order:
1. Zero the state.
2. Resample params, then call `params_hook` (the curriculum). The hook must run straight after resampling, because max HP is derived from `base_hp` next.
3. Map, kinds, personalities, positions.
4. HP, ammo, facing (toward the map centre), alive.
5. Targets −1 and death steps −1, then the hunt timer.
6. Crates, gas, `n_alive`.

Every reset leaves the gadget charged and the history empty, so an episode's first 3 decisions have partial history. Training draws a map uniformly per episode.

**Spawns.** One random rotation `r` over the angle-sorted spawn list; entity k takes spawn `(r + floor(k·n/E)) mod n`, so players spread evenly.

**Crates.**
- `n_boxes` 48 is clamped to the map's marked spots, so every spot is filled: 16–44 per map, 24.2 on average.
- Filling every spot raised the hero's cubes and win rate markedly.
- A broken crate's cube lands 0.3–1.8 tiles away (measured from footage). It gets 4 tries before falling back onto the crate.

**Pickups.** The lowest entity index wins ties, and an entity already at the cube cap (`cubes.max_cubes`: 16, or 99 in `train.yaml`) leaves the cube. Every death drops cubes, gas deaths included.

**Gas.**
- The safe area is a rect that shrinks about its own centre to a 2×2 minimum, at most one shrink per tick.
- The first shrink comes at `start_fraction` × episode seconds, then one every `step_seconds`.
- Damage is 0.20 × max HP per second (5 s from full) plus 0.04 per shrink.

| Setting | First gas | Per tile | Episode |
|---|---|---|---|
| `default.yaml` | 12 s (0.08 × 150 s) | 1.5 s | 150 s (3000 ticks) |
| `train.yaml` | 14.8 s | 5.0 s | 185 s (3700 ticks, 740 decisions) |
| The game | 19 s after the loop's gate | 6–7 s (1 tile per side) | — |

- **Source of the game timings:** 8 matches measured 2026-09-24 with `scripts/probes/zone_probe.py` on the loop's `GasMap`. The gas moves per line, so trust the E/W fronts over N/S.
- **The 1.3× schedule:** `train.yaml` runs 1.3× the game's pace.
- **Jitter:** `configs/randomization.yaml` scales `start_fraction` and `step_seconds` by an independent 0.9–1.1 per env at every reset in training; eval runs nominal.
- **Where the episode length lives:** in `train.yaml` only, because deployment rebuilds `time_frac` from the run's own `max_episode_steps × dt`.

**Termination.**
- `terminated` = hero dead, or `n_alive` == 1 with the hero alive.
- `truncated` = `step_count` ≥ `max_episode_steps`. A timeout with the hero alive but not alone is neither a win nor a death.

**Rank** is derived, never stored: 1 + the entities that outlived you. Alive entities tie at `n_alive`, and same-tick deaths tie. The obs rank is 1-indexed; `info["hero_rank"]` is 0-indexed (0 = won).

**Latching.** Outcomes latch at the sub-tick the episode ended. The world keeps ticking, so `final_observation` can be up to `action_repeat` − 1 ticks stale; this is accepted.

**Per-decision info:** `damage_matrix`, `damage_dealt_tick`, `damage_taken_tick` (combat only), `hp_healed_tick`, `kills_tick`, `deaths_tick`, `death_cause_tick`, `cubes_gained_tick`, `boxes_broken_tick`, `shots_fired_tick` (a proxy), `dash_hits_tick`, `attack_in_reach_tick`, `gadget_hit_tick`, `move_reversal_tick` (0 or 1 per decision), `hero_rank`, `hero_alive`, `alive_ticks`, `in_zone_ticks`, `n_ticks`.

## 9. Observation

**Native obs.**
- A nested dict. It is index-stable (the hero is entity 0, and slot order is fixed per episode) and full-information, with visibility as flags.
- **Groups:** hero, hist, entities (plus privileged), projectiles, boxes, pickups, zone, slots, visibility, view, world (a toggle), action_mask, meta.
- `core/obs_schema.OBS_SCHEMA` declares every field. `docs/OBSERVATION.md` and `docs/AGENT_OBS*.md` are generated by `scripts/dump_obs_schema.py` (`--spec` renders any spec); never hand-edit them.

**Grid** (uint8, accumulated in int32 and clamped to 255).
- **Channels:** 12 base channels (blocks_unit, blocks_projectile, is_bush, is_water, in_zone, enemy_any, enemy_revealed, enemy_hidden, hero, box, pickup, projectile), then `history_frames` (3) `enemy_hist` planes.
- **The view:** 13×21 around the hero, the largest axis-aligned rectangle inside the real camera's ground trapezoid. A bounding box would hand the policy corners it never sees live.
- **The world grid** uses the same rasterizer.

**Camera** (`core/camera.py`).
- **The quad:** `camera.quad` is the ground on screen, in tiles from the hero's nominal anchor. The anchor is the green ring, at `HERO_ANCHOR_TILES` (0.09, 0.80) from the viewport centre, with y pointing down. The corners are TL (−14.11, −10.59), TR (14.77, −10.93), BR (11.73, 7.45), BL (−11.55, 7.01): the viewport corners through the phone homography, which BlueStacks shares. `test_sim_camera.py` re-derives them.
- **The clamp:** the camera centre is `clamp(hero, (W, N), (map_w − E, map_h − S))`, with onsets W 12.1, E 12.8, N 8.9, S 5.5. These are geometry defaults (§1 #9): `scripts/probes/edge_probe.py` (19 clips, 2026-09-24) found that nearly every clamp run started at spawn, so footage only bounds them from below. An axis narrower than its onsets puts the camera midway between them; only 20-wide test maps do this.
- **The reveal:** `in_camera` tests four half-planes (inclusive), and `hero_view = vis[:, 0] & in_camera`. It drives `revealed_to_hero`, `hidden_by_bush`, the `enemy_revealed` and `enemy_hidden` planes, and history. Every `in_view` (entities, projectiles, crates, pickups) is the same window. `hero_revealed_to` stays on concealment, and `n_enemies_alive` stays exact (it is the HUD counter).
- **The grid:** the 13×21 crop stays hero-centred; only its enemy planes are windowed.

**`obs_select`** (`AgentObsSpec`) is the only fairness point.
- **Gating:** entities drop the hero row and are gated by `revealed_to_hero`. Projectiles are gated by `in_view`. Boxes and pickups pass ungated.
- **Projectile ranking:** projectiles are ranked by `time_to_closest` over all live projectiles, on screen or not, with ties nearest-first (`_nearest_first`: sort by distance, then stably by `time_to_closest`). An off-screen projectile keeps its slot, blanked. Ties are common, since anything moving away reads 0. Under `observation.projectile_static_speed` (train.yaml), anything slower than 2.0 tiles/s reads 0 too, with `vel` 0, before the ranking, as the live tracker reports it.
- **Refusals:** under `fair`, `enemy_any` and `enemy_hidden` are refused. So are `entities.privileged.*` and raw `slots.*`, always.
- **Groups:** at most 6 (`_MAX_GROUPS`), one host copy each. `BrawlFeaturesExtractor` takes at most one uint8 group, so a new field joins an existing group.
- **Normalizers:** tiles ÷ the larger map dimension, speed ÷ 20, HP ÷ 20000, counts ÷ 20.

**Tracked enemy slots** mirror the live `EntityTracker`, counted in decisions.
- An enemy takes the lowest free slot on its second consecutive sighting (`slots.promote_hits` 2).
- It keeps the slot through 3 unseen decisions (`slots.max_misses` 3), with its row zeroed by the fairness mask, and frees it on the fourth.
- Retirement runs before promotion, and `slots.update` runs before `build_obs`, so a sighting counts on its own decision.
- Not modelled: the tracker's HP-commit delay, its promotion tie order (moot with 9 slots for 9 enemies), and its promotion-tick `rel_vel`.
- Live, `DeployLoop` builds its `EntityTracker` from the run's `slots.*`, so an override reaches both sides. The tracker's `max_misses` also bounds its hero track's coast, which has no sim counterpart.

**History.**
- **`hist` group:** K = 3 decisions, newest first. Each slot has `valid`, `move_onehot` (17), `attack_onehot` (4, or 5 under auto-aim), `hp`, `ammo_frac` and `displacement` (position then minus position now): 78 floats, or 81. Every field is 0 when `valid` is False, since an empty slot's action (0, 0) would otherwise read as "idle, no attack".
- **`enemy_hist` planes:** plane k draws the enemies seen k decisions ago, at their world positions in the current window. Only those within `history_radius_tiles` (4) of the hero's tile now are drawn, by Chebyshev distance on tile indices: exactly the 9×9 block deployment can scatter.
- **What counts as seen:** `alive & hero_view`, so a sighting the screen never showed is never remembered.
- **Not stacked:** past terrain (it is static, and `displacement` stands in), and no per-frame enemy vector (the planes already give tile-resolution trajectories).

**`zone.active`** is `state.zone_seen`.
- It latches once `zone_step` > 0 and the window's bounding box around the camera centre crosses the safe rect. It is checked once per decision and cleared only by reset.
- Live, it is `ZoneEstimator.active`, which latches the same way: set by the first gas the loop's gas map holds, kept through odometry segment changes (which empty that map), cleared only at a match start.
- `zone.hero_margin_local` stays exact: three of its four horizons are inside the window, and the jitter removes the timing leak.

**Recorded zone mismatches** (accepted, §2):
- Before the first shrink, `hero_margin_local` is the map-edge distance in the sim but +10 live.
- `zone_grid` paints off-map cells as gas from tick 0, while the live plane stays zero until gas is seen.
- In the gas, the sim negates the crossed side and the live loop negates all four.

**`hero.near_edge`** is `|hero − camera centre|.amax > camera.edge_flag_tiles` (2.0). It fires within 10.1, 10.8, 6.9 and 3.5 tiles of the W, E, N and S edges.

Its live counterpart is `TrackerResult.hero_offset`: the player box's tile minus its nominal point, where the nominal point is the viewport centre + `HERO_ANCHOR_TILES` + `PLAYER_BOX_FROM_RING_TILES` (−0.08, −0.78). That last offset corrects the box being read at `detector.anchor_frac` rather than the ring; it was measured on five BlueStacks recordings.

**Deploy parity rule.** A field enters a deployable spec only if `brawl_deployment` can supply it, either from `brawl_vision` or from the shadow's proprioception. `agent_obs_lowinfo.yaml` is stricter: a field must be recoverable by CV.

**The training spec, `agent_obs_deploy5.yaml`,** has six groups and 266 extractor floats (263 without auto-aim), all pinned in `test_configs_files.py`:

| Group | Contents |
|---|---|
| self | 27 floats; the gadget pair (`gadget_ready`, `gadget_charge_frac`) at columns 23–24, then `near_edge` after `in_zone` |
| enemies | 9×9, in tracked-slot order |
| projectiles | 12×6 |
| zone | 5 |
| history | `hist`, 81 floats under the run's auto-aim |
| grid | 13 channels, 13×21, uint8 |

`projectiles.class_onehot` exists in the schema but deploy5 omits it. `train.yaml` turns off the world grid and `raw_los` (the (N,E,E) LOS used only by `entities.los_from_hero` and `visibility.los`); turn either back on only if a spec reads it. The camera and slot dataclass defaults equal the shipped yaml, so older frozen configs get the same camera.

## 10. Reward

**Wiring.** The sim computes none. Training builds `ShapedReward`, and the SB3 adapter installs it as the env's `reward_fn`. The env calls it once per decision, on-device and sync-free. The returned buffer is shared, so treat it as read-only.

**Pricing.** Terms are priced per sim tick: deltas are summed over sub-ticks and rate terms count sub-ticks, so the episode return does not depend on `action_repeat`. `gamma` is per decision. The one exception is `move_reversal`, which counts decisions: a reversal is a change between two of them.

**Terms** (`train.yaml` weights):

| Term | Weight | Notes |
|---|---|---|
| `win_bonus` | 10 | |
| `death_penalty` | −3 | Reads the latched `info["hero_alive"]`, never the obs. |
| `rank_bonus` | 0.5 | Per place above last. |
| `damage_dealt` | 3e-4 | |
| `damage_taken` | −1e-4 | Combat only; the gas has no attacker. |
| `hp_healed` | 1e-4 | Mirrors `damage_taken`, so an out-healed wound is a wash. |
| `kill` | 3 | |
| `cube_pickup` | 0.7 | The early game is where cubes are built: at 60 s the hero won 0.17 with 0–1 cubes, 0.42 with 2–3 and 0.57 with 4–6 (2026-09-25). |
| `survive_per_step` | 0 | |
| `in_zone_per_step` | −0.05 | Flat, so it reads the same at any gas damage. |
| `attack_in_reach` | 0.05 | |
| `gadget_hit` | 0.3 | |
| `move_reversal` | −0.01 | 0 in `RewardConfig`, so a run without the key never pays it. |

**`attack_in_reach`** counts the hero's attacks and supers made while an enemy he can see stands inside his uncharged dash reach: `dash_distance + dash_radius + unit_radius` = 3.77 tiles, the cadence audit's utilization radius. It is capped at one per decision.
- `env._attack_phase` computes it before `start_dash`.
- Gadget throws don't count, but a dash away and a miss are paid too (blunt on purpose).
- A whole episode of in-reach attacks is worth about 3, against 10 for a win.

**`gadget_hit`** counts landings of the hero's gadget spinner that hurt at least one player: one per landing, however many players the blast catches, paid on top of `damage_dealt`.
- `env._projectile_phase` computes it as the hero's hits that `charge_hit` leaves out, since the spinner is the one projectile that charges no super.
- A landing on crates alone, or on nothing, pays nothing: box damage never enters `dmg_by`.
- The 18 s cooldown allows about ten throws a match, so about 3 at most, against 10 for a win.
- A super bolt that hits the same player on the landing tick hides the spinner's hit on that player (rare).

**`move_reversal`** counts decisions whose move bin swung 135° or more from the previous decision's (6 or more of 16 bins), both non-idle (user, 2026-10-07).
- `core/events.move_reversals` reads the previous decision off history slot 0 in `env.step`, before `history.push` overwrites it. So an episode's first decision never counts: the reset empties the ring.
- The 2026-09-30 policy reversed on 26 % of its decisions (measured 2026-10-07), about 1.2 a 120 s match at −0.01. A dodge that saves one 1300-HP hit is worth 0.13 in `damage_taken` alone, so a real dodge still pays and jitter with nobody shooting stops paying.
- The first run trained with it, `runs/mortis_ppo-20261007-161117`, reverses on 11 % (watch) to 16 % (train) of its decisions, on the probe's count (pairs of non-idle moves). The 2026-09-30 policy reverses on 28 % on the same count and seeds (measured 2026-10-10). The rate was already that low at 200M (MODEL_SIZE_STUDY.md).
- The policy sees its last three moves (`hist.move_onehot`), so the cost is a function of what it observes.

**Invariants and guards.**
- The terminal payoff must dominate `survive_per_step`.
- Raising `hp_healed` far enough makes going into the gas and regenerating afterwards profitable.
- `builder.check_reward_is_observable` refuses a `cube_pickup` weight under a spec that cannot see pickups.
- `TERM_NAMES` keeps a stable order, each new term appended (`attack_in_reach`, then `gadget_hit`, then `move_reversal`), so term curves stay comparable across runs.

## 11. Training

[TRAINING.md](TRAINING.md) is the how-to; this section covers the design.

**Build chain** (`training/builder`):
1. `BrawlVecEnv` with the merged spec. `CurriculumManager` is its `params_hook` and shares `sim.gen`, so one seed also reproduces the tier draws.
2. `VecMonitor(BrawlSB3VecEnv)`.
3. `VecNormalize`, on the reward only.
4. MaskablePPO `MultiInputPolicy` through `VecTransposeImage(skip=True)`. SB3 guesses an image's channel axis from its smallest dimension; without the skip, `--smoke` once trained on transposed grids.

`Distribution` argument validation is off process-wide.

**Features extractor.** A padded 3×3 CNN (strides 1, 2, 2; SiLU) plus a SiLU MLP over the float groups. `n_flatten` is measured, never hard-coded, and `normalize_images=False` is required.

**Units.** One SB3 timestep is one decision (0.25 s): `n_steps`, `total_timesteps` and `eval.every_timesteps` all count decisions. A bare `python scripts/train.py` trains the deployable spec.

**PPO (`train.yaml`):**
- 4096 envs × `n_steps` 128; batch 8192; 6 epochs.
- `gamma` 0.998, `gae_lambda` 0.98.
- Entropy 0.005, value 0.5, max grad norm 0.5, `target_kl` 0.025.
- LR linear 2e-4 → 5e-5; clip linear 0.2 → 0.1.
- Policy: CNN output 160, shared MLP [256, 256], pi/vf [384, 384].

The PPO buffer lives in host RAM, proportional to `n_steps` × `n_envs` × agent-obs bytes; recompute it before raising `n_steps`.

**Tiers** multiply `brawlers.yaml` values per (env, bot kind) at each reset. The hero's column is always 1.0.

| Tier | aim noise | reaction | lead | decision period | speed | HP | damage | aggression | hero focus |
|---|---|---|---|---|---|---|---|---|---|
| easy | 2.5 | 2.2 | .25 | 2.0 | .90 | .75 | .70 | .6 | 0 |
| medium | 1.8 | 1.6 | .6 | 1.5 | .95 | .90 | .85 | .8 | .4 |
| hard | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| veteran | .6 | .6 | 1 | 1 | 1 | 1.15 | 1.15 | 1.2 | 1.3 |
| expert | .4 | .45 | 1.25 | .75 | 1 | 1.3 | 1.3 | 1.4 | 1.5 |
| elite | .2 | .3 | 1.5 | .5 | 1 | 1.5 | 1.5 | 1.7 | 1.7 |

- **Hard** is `brawlers.yaml` as authored. **Easy** is in no stage; it is only a yardstick.
- **Elite** caps HP and damage at 1.5× (§2); accuracy, aggression and hero focus carry the rest.
- **Clamping:** the lead and hero-focus products are clamped to [0, 1], because past 1 they mean a different behaviour. The decision period is rounded, with a floor of 1. Aggression is scaled unclamped, since personality clamps each read, and `aggression: 0` is refused.
- **Speed** stays 1.0 above hard, because Mortis is only 6–14 % faster than the ranged bots.
- **What tiers never scale:** any stat that a `SCALAR_FIELDS` budget or the demand bound reads (ranges, shot, shard and dash speeds and distances, ammo, cooldowns).
- The 0.25 s cooldown floor equals one decision. If `action_repeat` ever changes, revisit the floor with it.

**Stages** (window 2000 episodes, at least 2000 since the transition, no demotion, 0.15 gates):

| Stage | Tier mix |
|---|---|
| hard_intro | medium .3, hard .7 |
| veteran_intro | hard .6, veteran .4 |
| veteran | hard .3, veteran .5, expert .2 |
| expert | hard .1, veteran .3, expert .4, elite .2 |
| elite | hard .1, veteran .15, expert .25, elite .5 (no gate) |

- **Transitions:** the window clears at each transition, and each env's first finish afterwards is skipped, because its bots came from the previous stage.
- **Forced advance:** `max_timesteps_at_stage` (125M of 600M) force-advances a stage. Elite needs `total_timesteps` above 4 × the cap.
- **Resume:** the state JSON is rewritten at each transition.

**Eval** runs every 20M decisions and at the start.
- **Episodes:** 300 per tier, one env of tiers × episodes in contiguous blocks pinned by `FixedTierHook`. It is deterministic, reseeded identically each time (seed 999983), with no randomization overlay.
- **Counting:** only each slot's first episode counts. Short episodes are deaths, so counting every finish would bias the rate down.
- **Holdout:** a twin evaluator on the holdout maps logs `eval/holdout_win_rate[_<tier>]` and `eval/holdout_gap`. Read the gap's trend from the at-start row, not its sign. `best_model.zip` is picked on the training-map mean only, since picking on the holdout would spend it.
- **Validation:** `validate_train_config` refuses a holdout map that is unknown, duplicated, a bare string, in the training rotation, or unloadable. Readers that never build the holdout env (`DeployedPolicy.from_run`, `watch.py`, the cadence audit, probes) pass `check_holdout=False`.
- **`gadgets_used`:** throws per episode, counted from the evaluator's own actions by the sim's rule.

**Logging.** `progress.csv` and `log.txt` always, and TensorBoard if installed. `train/*` never resets, while `curriculum/win_rate` resets at each transition. Compare runs on `eval/*`, never on train win rate.

**Cadence audit.** `scripts/audit_attack_cadence.py` takes `--run` (a checkpoint) or `--telemetry` (a deployment CSV). It reports:
- `utilization` = P(attack | legal and a visible enemy in reach);
- the fight-interval histogram;
- `phasing_loss`;
- `long_dash_waiting`;
- `ammo_at_first_attack`.

Values 1, 2 and 4 count as attacks; the gadget does not.

## 12. Performance

Speed is read off a real run's `time/fps` (§2). This section records what shapes speed, not figures.

**Kernel-launch bound.** The sim is bound by kernel launches, not tensor size. Projectile physics and the attack phase dominate a step, and shrinking small (N,E) work buys nothing.

`action_repeat` is cheap: both obs builds, the host copies, the mask and the policy forward are paid once per decision, and each tick runs only the world update (bot visibility included). SB3 has never been the bottleneck.

**Levers in place.**
- **Training config:** the world grid and `raw_los` are off in training.
- **March budgets per call:** the melee cone LOS uses `cone_ray_tiles`, the bots' crate LOS `attack_ray_tiles`, the dash `dash_ray_tiles`, and the projectile wall march `shot_step_tiles` (the farthest a wall-stoppable shot moves in one tick; supers are excluded because they pierce).
- **Bot LOS:** target-only (N,E), instead of (N,E,E).
- **Crate collisions** test only nearby tiles. `_box_grid` is an int16 tile-to-crate map rebuilt by one scatter each tick. `_box_candidates` searches within `box_reach_tiles` and falls back to every crate when the window would be too large. This relies on at most one alive crate per tile, which `boxes.spawn_boxes` guarantees and `check_invariants` enforces.
- **Split shards** are assigned from the slot side: each free slot finds its (shell, arm) by `searchsorted` into the demand cumsum.
- **SB3 side:** `action_masks` uses direct indexing, and `Distribution` validation is off.

**Exactness.**
- The crate-grid and split-spawn paths match the old code to within 1 ulp; tests pin each fast path against a dense reference.
- An ulp in a shard velocity can move a trajectory within a few decisions, so an exact-trajectory test can shift.
- `dmg_ent` and `dmg_by` already differ by 1 ulp between identical CUDA runs (atomic `scatter_add`).

**Declined.**
- CUDA-graph capture: it needs every state and params tensor updated in place, but the curriculum rebinds them.
- A GPU-resident rollout buffer (SB3 internals). Both fall under the repo-code-only rule.
- `torch.compile` stays off: the CPU inductor needs MSVC `cl.exe`, and the CUDA inductor needs Triton. One graph break is known (`torch.randn` with a Generator in the bots' aim noise). `_guarded_tick` turns a dynamo or inductor failure into a clear `RuntimeError`.

**Tried, no gain:**
- hero-column-only dash and gadget marches;
- a narrower melee crate-LOS march;
- skipping the final-obs copy at episode ends;
- trimming the SB3 wrapper, info dicts and callbacks.

**Resources.**
- Nothing grows per step. Per-episode callback stats are `deque(maxlen=window)`; only stage changes and eval points append to lists.
- VRAM is not the constraint.

**If throughput collapses**, check in this order:
1. A hidden host sync (`torch.cuda.set_sync_debug_mode("warn")`).
2. A per-env (N,H,W) gather.
3. Boolean-mask indexing.
4. float64 leaking in through a Python float in `torch.where`.
5. A grid built by gather instead of scatter.
