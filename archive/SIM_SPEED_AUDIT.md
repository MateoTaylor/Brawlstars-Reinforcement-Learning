# Sim speed audit — 2026-09-30

This audit covers three questions:
- where a training step's time goes at the real training config;
- what can be cut before the next run;
- whether logging or obs features are worth removing.

Nothing in the repo was changed. Every number comes from scratch probes that patched the sim in memory.

## Status — 2026-09-30: #1–#4 implemented

#5 (CUDA graphs) and the GPU rollout buffer are not planned: no SB3 internals, no edits to installed packages. Line numbers in the audit below refer to the code before these changes. No speed was measured after the change; `time/fps` in the next run is the measurement.

| # | Where | How it differs from the proposal |
|---|---|---|
| 1 | `config.shot_step_tiles` → `SimParams.shot_step_tiles`, passed as `max_tiles` to the wall march in `projectiles.step_projectiles` | Named `shot_step_tiles`. It covers only shots a wall can stop (`proj_speed`, shard speed); supers pierce, so their speed is not in it. At the default spec it is 0.333 tiles, so the march takes 1 sample plus the endpoint instead of 48 plus it |
| 2 | `observation.include_raw_los` (EnvConfig `obs_include_raw_los`), gating `entities.los_from_hero` and `visibility.los`; `env._build_observation` skips `raw_los` when off | As proposed. Default `true`; `train.yaml` sets `false` |
| 3 | `projectiles._box_grid` / `_box_candidates`; reach from `config.box_reach_tiles` → `SimParams.box_reach_tiles` | The tile→box grid is rebuilt each tick (int16, one scatter) instead of kept from reset, so broken boxes need no bookkeeping. Reach is `max(widest shot + BOX_RADIUS + step/2, widest blast)` = 1.0 at the default spec, a 3×3 window. Falls back to testing every box when the window would hold ≥ `max_boxes` tiles, which test configs with 8 box slots always do |
| 4 | `projectiles._spawn_splits` | As proposed: each free slot finds its (shell, arm) by `searchsorted` into the demand cumsum, then every field is one (N,P) gather |

**Exactness, and how it is pinned**
- #1, #3, #4 have unit tests in `tests/test_projectiles.py`, each against the computation it replaces:
  - `test_projectile_wall_march_budget_matches_full_ray` (full 24-tile ray);
  - `test_box_candidates_match_testing_every_box` (every box);
  - `test_spawn_splits_matches_alloc_slots_assignment` (the old `alloc_slots` assignment, with exact writes; see below).
- #3 relies on at most one alive box per tile. `boxes.spawn_boxes` guarantees it, and `check_invariants` now enforces it (`tests/test_state.py`).
- #2 is pinned by `tests/test_obs_schema.py` (drops exactly the two fields, everything else identical) and `tests/test_configs_files.py` (on in `default.yaml`, off for training).
- **Exact to the intended values, not bit-identical to the old code.** Two things differ by one float ulp:
  - #3 sums box damage with `scatter_add_`. When 3+ hits land on one box in one tick, the sum can differ from the old one in the last bit (addition order). No tie-break or hit set changes.
  - #4 writes each shard's values directly. The old `_write_slots` wrote `old + (new - old)`, which rounds, so old shards carried velocity, position, target and range up to one ulp off the value they were meant to have. The #4 unit test's reference writes the value itself, like the new code, so it pins slot assignment and values exactly but not that old rounding.
  - An ulp in a shard's velocity is enough to send a trajectory a different way within a few decisions. Anything pinned to an exact old trajectory can therefore move.
- **CUDA checks at the training config.** 1024 envs, 1,500 ticks, random hero actions (62k box hits, 153k shards spawned). Each tick was recomputed from a copy of the same state.
  - Against the literal old code (the committed `projectiles.py`, loaded as a sibling module): every field within one ulp. Shard `prj_vel`/`prj_pos`/`prj_target`/`prj_dist_left` differ on nearly every tick (≤ 4.8e-7 / 3.8e-6 / 3.8e-6 / 2.4e-7). `dmg_box` differs on 15 ticks (≤ 4.9e-4 on sums in the thousands).
  - Against the new code's own full-ray, every-box fallback, so #1 and #3 alone: every `prj_*` field, `dmg_box`, `heal_ent` and `charge_hit` identical.
  - `dmg_ent`/`dmg_by` differ by one ulp on a few ticks in both comparisons. The old code, run twice on the same state, differs from itself as often: CUDA's atomic `scatter_add_` order.
  - No tick had two alive boxes on one tile.
- **Test fallout, now fixed: `tests/test_deployment_assemble.py::test_deploy4_assembles_whole` failed after #4.** Its six-decision trajectory first departs from the old one at tick 12 (three shards, one ulp).
  - By decision 5, four live projectiles have `time_to_closest` exactly 0 (everything moving away from the hero reads 0).
  - `obs_select` ranked with `torch.topk`, which leaves the order of ties unspecified. It ordered those four differently over the sim's 192 scattered slots than over the assembler's packed list.
  - The old trajectory passed only because its decision 5 held no order-sensitive tie. The gap was real in deployment too: the tracker's order is unrelated to sim slots, so tied rows reached the policy in a different order than in training.
  - **Fix (user's choice, 2026-09-30): ties go nearest the hero first.**
    - `obs_select._nearest_first` sorts by distance (from `rel_pos`), then stably by `time_to_closest`, and keeps the first K. The order depends on the projectiles, not on their slots; only a tie on both keys stays in slot order.
    - The test passes again. `tests/test_obs_select.py::test_tied_projectiles_go_nearest_first_whichever_slots_hold_them` pins the rule over three slot layouts; the old `topk` fails it.
    - Only the order of tied rows changes. Before, it was whatever `topk` made of the slot layout, which deployment could never reproduce.

**Found in passing, not changed.** A piercing super can fly through the map's border wall and off the map for a tick before its range runs out. The CUDA check counted 458 such slot-ticks, identically in the old code. It is harmless with `debug_checks: false` (terrain lookups clamp). With `debug_checks: true`, `check_invariants` raises on it.

## Bottom line

- **Removing logging or obs features won't buy much.**
  - The full map grid is already off for training (`configs/train.yaml:69`, `include_world_grid: false`), so no time is spent on it.
  - Building obs, copying obs and computing info together cost about 5% of a step.
  - The one obs item worth cutting is `raw_los` (#2, −3%).
- **The time goes to projectile physics.** The projectile and attack phases are ~70% of a step. Most of that is dense work over all 192 projectile slots, while an env has ~5 live projectiles on average.
- **Two small exact changes save 12%** (measured): #1 and #2.
- **Two moderate exact rewrites could save up to another 24%.** That is a measured upper bound; expect somewhat less. They are #3 (box collisions) and #4 (split spawning).
- **No runaway memory or storage risk found.**

## How this was measured

- **Setup.** Real training config: `configs/train.yaml` with 4096 envs on CUDA.
  - The final policy of `runs/mortis_ppo_34maps-20260927-165016` plays.
  - The curriculum is loaded at stage 4 (elite), so episodes look like training.
- **Check against real training.** A timed `learn()` iteration took 69 s collecting plus 15 s training, 84 s in all. The real run logged 79–85 s per iteration.
- **How savings were measured.** Unsynced A/B timings, the way training runs, in interleaved rounds.
  - Figures are ms per decision step, including the policy forward pass and masks.
  - Noise between runs is about ±15 ms (±3%).
- **Per-function timings.** These put a sync around each call and were used only to find candidates. They overstate functions that work on small tensors (see "Not worth doing").
- **"Upper bound".** The item is replaced by a no-op. That is a timing measurement only, never a fix.

## Where a step goes

One decision step is 5 sim ticks for 4096 envs. The synced breakdown below totals 558 ms/step; unsynced, the same step takes ~537 ms.

| Part | ms/step | Largest pieces |
|---|---:|---|
| Projectile phase | 254 | box collision 77, wall march 62, split spawning 39, unit collision 14 |
| Attack phase | 126 | dash body march 57, melee hitscan 31 (22 of it the box line-of-sight march), volleys 8 |
| Bot phase | 54 | `personality.movement` 33 |
| Autoreset | 29 | reset 11.6, second obs build 17.5 |
| Movement | 20.5 | |
| First obs build | 17.5 | `raw_los` is 6.3 of each build |
| Box phase | 12.8 | |
| Final obs/info copy | 8.1 | |
| Everything else | ~36 | history, info, reward, zone, events, SB3 wrapper (~4) |

**Kernel launches.**
- One step launches 26,019 GPU kernels.
- The GPU is busy for only 63% of the step (342 of 539 ms).
- For the rest, it waits for Python to launch the next small kernel.

**Projectile occupancy.** An env has 5.5 live projectiles on average. The worst tick's 99th percentile is 27, and the maximum is 71 of 192 slots.

**Outside the sim step:**
- The policy forward pass, buffer adds and masks cost ~13 ms/step together.
- The training phase takes 15 s of each 84 s iteration.
- Eval takes ~12% of wall-clock (see "Your call").

## Proposals, ranked

### 1. Budget the projectile wall march — exact, small, −9 to −11%

`brawl_sim/core/projectiles.py:644` checks each projectile slot's movement against walls with the full 24-tile ray (48 samples). No projectile moves more than 0.6 tiles in one tick:
- Mortis's super, at 12 tiles/s, is the fastest.
- The fastest bot shots (Spike's shards) move 0.33 tiles per tick.

**Change**
- Add a `proj_ray_tiles` scalar to `SimParams.SCALAR_FIELDS`.
- Derive it from the spec, the same way `dash_ray_tiles` is (`brawl_sim/config.py:676`): the fastest shot any kind can fire, times `dt`.
  - Sources: `proj_speed`, `super_proj_speed`, and shard speed `split_distance / split_seconds`.
  - Use the spec highs, and the low for `split_seconds`.
- Pass it as `max_tiles` at line 644. The march then takes 2 samples instead of 48.

**Why it's exact**
- `wall_hit` is only used for projectiles that are alive, not lobbed and not piercing, and every one of those moves less than the budget.
- Curriculum tiers don't scale projectile speeds.
- The ±10% randomization overlay reaches `_spec_upper` as a range.
- Measured: 0 mismatches over 78.6M compared elements, and 0 live projectiles over budget.
- A test should pin that the budgeted march equals the full one, like `test_melee_los_budget_matches_full_budget_ray`.

**Measured:** 537.5 → 489.5 ms/step (−8.9%). A separate run gave 546.7 → 485.1 ms/step (−11.3%).

### 2. Turn off `raw_los` for training — exact for training, small, −3%

`brawl_sim/env.py:324` computes `perception.raw_los` on both obs builds in every step. It is a full-length wall march over every entity pair, (N,E,E).

**Who reads it**
- Only two obs fields: `visibility.los` and `entities.los_from_hero`.
- Nothing reads those two fields: not the agent spec, the reward, the wrappers or the callbacks.
- The viewer computes its own copy (`scripts/record_rollout.py:45`), so replays are unaffected.

**Change**
- Add an `observation.include_raw_los` flag (default `true`; EnvConfig field `obs_include_raw_los`).
- Gate both fields with it, the same way `obs_include_world_grid` gates `world` (`brawl_sim/core/obs_schema.py:235`).
- Set it to `false` in `train.yaml`, next to `include_world_grid: false`.

**Measured:** 489.5 → 471.2 ms/step on top of #1, which is −3.4% of the original step.

**#1 and #2 together:** 537.5 → 471.2 ms/step (−12.3%). Expect `time/fps` to rise about 10%. Eval speeds up by the same fraction.

### 3. Box collisions: test only nearby boxes — exact with a checked bound, moderate, up to −15%

At `projectiles.py:694-696`, every projectile slot is tested against every box slot each tick. That is 4096 × 192 × 48 = 38M segment–circle tests, 5 times per step.

**Upper bound:** 72.7 ms/step, 15% of the step after #1 and #2.

**Change**
- Boxes never move after spawning (`brawl_sim/core/boxes.py` docstring).
- At reset, build a per-env grid that maps each tile to the box slot on it. In int8, 60×60 per env, that is ~15 MB at 4096 envs.
- Each projectile then tests only the boxes in the 3×3 tiles around it: (N,P,9) instead of (N,P,48).
- The box damage attribution after the test switches from an (N,P,B) mask to candidate indices.

**Why it's exact**
- It is exact while reach stays under 1.5 tiles. Reach is the largest non-piercing shot radius (0.30), plus `BOX_RADIUS` (0.5), plus the largest per-tick step.
- Derive that bound from the spec, as in #1.
- Check first that box spots are tile-centred and that each tile holds at most one.
- Test against the dense version over rollouts.

**Expected saving:** most of the upper bound. I have not measured it beyond that.

### 4. Split spawning: write only the slots that change — exact, moderate, up to −9%

`_spawn_splits` (`projectiles.py:420-533`) runs every tick over 192 shells × 6 arms, which is 1,152 candidate shard slots per env. `_write_slots` then makes 13 gather/where/scatter passes over them, even though a tick has at most a handful of detonations.

**Upper bound:** 43.8 ms/step (9%).

**Change**
- Invert the write. For each destination slot, find which (shell, arm) claimed it. The free-rank arithmetic `alloc_slots` already does gives the answer: a `searchsorted` into the demand cumsum.
- Then write each field with one (N,P) gather, which touches 6× fewer elements.

**Why it's exact**
- The slot assignment and the values are the same.
- The current delta plus `scatter_add_` write can be off by one float ulp; the new one writes the value itself.
- A test should run old and new on the same state and compare every `prj_` field.

**Expected saving:** most of the upper bound.

**#1–#4 together:** if all four land near their bounds, the step drops from ~537 ms to roughly 360–390 ms. That would mean roughly 25–35% more fps. This is an estimate built from the upper bounds.

### 5. CUDA graphs over the tick loop — structural, large; not before the next run

**Why it would help**
- One step launches 26,019 small GPU kernels, and the GPU sits idle ~37% of the time waiting for Python to issue them.
- That is also why cutting elements from small (N,E) tensors does nothing. Running the dash march on the hero column alone removed 90% of its elements and saved nothing measurable.
- Capturing `_run_decision` in a `torch.cuda.CUDAGraph` and replaying it would remove the launch overhead.

**Why it's feasible**
- The sim already follows the no-host-sync rule (`CONVENTIONS.md`).
- torch 2.11 can register the sim's RNG generator with a graph.

**The work**
- Every state and params tensor must be updated in place instead of reassigned. `resample_params` and the curriculum multipliers both rebind params with `setattr`.
- Add a bit-exact eager-vs-graph test.

**Potential:** most of that idle time. Not measured.

### Not worth doing (measured)

| Idea | Result |
|---|---|
| Dash march on the hero column only (57 ms synced) | Exact, no measurable gain: its cost is kernel launches, not tensor size |
| Gadget march on the hero column only | Exact, no measurable gain |
| Melee box line-of-sight march, (N,E,B) | Upper bound 13.9 ms (3%); already budgeted to 6 samples |
| Skip copying the final obs/info at episode ends | No measurable gain |
| Remove the world grid | Already off for training |
| SB3 wrapper, info dicts, callbacks | Under 2% together |

### Your call (trade-offs, not exact)

**Projectile slot count** (`limits.max_projectiles: 192`, `configs/default.yaml:172`)
- Slot count is the biggest multiplier in the step. Each slot costs ~1.1 ms/step.
- At 96 slots: 477.6 → 372.5 ms/step (−22%), and peak GPU memory fell from 2.23 to 1.31 GiB.
- `config.peak_projectile_demand` puts the worst case at 162 slots: 9 Spikes, each with 3 shells in the air, each shell splitting into 6 shards.
- The measured peak is 71 live slots, and the mean is 5.5.
- Overflow drops shots silently. The 162 bound also ignores reloads during a shot's flight, so it is not strictly tight either.
- #3 and #4 shrink the cost of each slot, so this lever matters less after them.
- If you lower it, add a dropped-shot counter to the logs so any thinning shows up.

**Eval** (~12% of wall-clock)
- Each eval point runs two 1,800-env evaluators back to back: training maps, then holdout maps. Together they take ~415 s every 20M steps.
- Every sim fix above speeds eval up by the same fraction.
- Beyond that, eval frequency is the only lever. That is yours to set, and it does not change what the `eval/*` values mean.

### Outside the sim

The training phase takes 15 s per iteration (18%).
- SB3's rollout buffer lives on the CPU, so each of the 384 minibatches is copied to the GPU.
- Each copy takes 11.6 ms, 4.4 s per iteration (~5%).
- A rollout buffer kept on the GPU would remove most of that. It is moderate effort and means changing SB3 internals.

## Memory and storage

**GPU memory is stable.**
- Peak allocation during collection was 2.23–2.26 GiB in every probe, with no growth over hundreds of steps.
- SimState is 82 MB at 4096 envs.
- #3's tile grid would add ~15 MB. Nothing else here adds persistent memory.

**Storage is fine.**
- `runs/` holds 21.3 GB, and 1,077 GB is free.
- A 450M run writes ~3.7 GB of checkpoints: 45 × (63 MB model + 19 MB VecNormalize pickle).
- The pickle is large only because it stores the last obs for all 4096 envs. That is harmless.

## Probe scripts

The probes (`sim_profile.py`, `sim_probe2.py` through `sim_probe4.py`) live in this session's scratchpad, not the repo. This file is the record of their numbers.
