# Test suite audit — 2026-10-20

A systematic pass over the simulation's tests for three things: tests that cannot fail or check
the wrong thing, tests that went stale as the sim moved on, and tests that cost far more than
what they check. Source under `brawl_sim/`, `configs/` and `scripts/` was not touched; every
change is in `tests/`.

## Headline

| | Before | After |
|---|---|---|
| Tests (`-m "not vision"`) | 2925 | 2859 (0 failures) |
| Serial wall time | 32 min (1918 s of test time) | 7 min 17 s (432 s) |
| `-n 8` (pytest-xdist 3.8, already installed), GPU hidden | not used | 1 min 43 s |

Four modules held 79 % of the time at baseline: `test_autoreset` 600 s, `test_maps` 399 s,
`test_env` 297 s, `test_training` 223 s. Every one of them was a rollout run narrow and long
(1-8 envs for hundreds or thousands of decisions). After the audit the same four take 198 s
serial.

**The lever is batching, not the GPU.** On CPU the sim is dispatch-bound: a decision costs
nearly the same at 64 envs as at 8, so "8 envs x 1000 decisions" and "64 envs x 120 decisions"
cover the same env-decisions at a fraction of the wall time. The heavy rollouts were rewritten
that way, keeping (or measuring) what each one has to reach: every env through two episodes and
their autoresets, gadget throws counted, clock-only endings counted, and so on.

## How to run it now

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -u -m pytest -m "not vision" -n 8 --dist worksteal -q
```

From PowerShell (the variable is removed afterwards, so later GPU work in that window still sees
the GPU):

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"; .venv\Scripts\python.exe -u -m pytest -m "not vision" -n 8 --dist worksteal -q; Remove-Item Env:CUDA_VISIBLE_DEVICES
```

- No thread variables are needed: `tests/conftest.py` pins each xdist worker to one torch/MKL
  thread. A serial run keeps the library defaults, and an `OMP_NUM_THREADS` you set yourself
  wins.
- After a red run, `--lf` reruns only what failed and `--ff` runs it first (the cache is
  `.pytest_cache/`, gitignored).
- One file while iterating: `pytest tests/test_<area>.py -q` (most files now finish in seconds;
  see the table below).
- Six tests are marked `gpu`: the four sync-free checks, the SB3 smoke on CUDA and the
  detector's CUDA inference. With the GPU hidden they skip; `-m "not vision and not gpu"` drops
  them outright. With the GPU visible the four sync checks cost ~25 s together and the CUDA smoke ~9 s;
  run them when a change touches the step path and no training run holds the GPU.
- A `testpaths = ["tests"]` line in pyproject.toml would save ~0.5 s of collection (4.3 s vs
  3.7 s, same 2951 tests): the default `norecursedirs` already skips `.claude/` and `.venv/`.

Measured 2026-10-20 with nothing else running (Ryzen 7 9800X3D, 8 cores / 16 threads):

| Run | Wall | Summed test time | Result |
|---|---|---|---|
| `-n 8`, pinned by tests/conftest.py (the command above) | 1 min 43 s | 557 s | 2852 passed, 7 skipped |
| `-n 8`, pinned by `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command line | 1 min 43 s | 567 s | 2852 passed, 7 skipped |
| `-n 8`, unpinned (before the conftest pin) | 3 min 26 s | 1311 s | 2851 passed, 7 skipped, 1 failed |
| serial, GPU visible | 7 min 17 s | 432 s | 2858 passed, 1 skipped |

- **One thread per worker, set in tests/conftest.py.** Unpinned, each of the 8 workers starts a
  full torch pool on 8 cores: the run takes twice as long, and
  `test_vision_classifier.py::test_inference_is_cheap_enough_for_four_parallel_games` (a
  real-time budget) fails. conftest.py now sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to 1 when
  it loads inside an xdist worker, before any test module imports torch. Checked with a probe
  plugin: 1 torch thread in each worker, 8 in a serial run. The budget test can still fail when
  the suite shares the machine with other heavy work.
- The wall time is set by the longest single tests, which no worker can split: test_maps'
  every-generated-map smoke (74 s under `-n 8`, 41 s alone), the `train.py` smoke (61 s), and
  test_env's invariants rollout (35 s).
- Module-scoped fixtures are built once per worker, not once per run: test_nav's distance fields
  cost ~8 s in each worker that draws a nav test, so test_nav sums to 77 s under `-n 8` against
  10 s serial. That does not move the wall time. `--dist loadscope` would keep each module on
  one worker, but then test_training's ~100 s would run on a single worker.
- Hiding the GPU made test_vision_detector.py fail 7 tests. onnxruntime-gpu still lists its CUDA
  provider with no device behind it, the session falls back to the CPU, and the detector's guard
  raised on that even for `device: auto` ("Source note, vision side" below). The guard is fixed
  since 2026-10-21, so the detector tests build from the shipped `auto` again, with no device
  workaround. With the GPU hidden they build on the CPU and the CUDA inference test skips. With it
  visible all 59 pass on CUDA, that test included.

## Where the time went (per module, seconds of test time)

| Module | Before, serial | After, serial | After, `-n 8` (summed) |
|---|---|---|---|
| test_autoreset | 600.1 (5) | 13.5 (4) | 15.6 (4) |
| test_maps | 398.6 (93) | 44.6 (72) | 77.9 (72) |
| test_env | 297.4 (27) | 43.6 (26) | 61.6 (26) |
| test_training | 223.3 (120) | 96.0 (119) | 118.7 (119) |
| test_integration | 37.5 (10) | 26.5 (10) | 8.8 (10) |
| test_sb3_integration | 33.7 (3) | 10.7 (2) | 2.0 (2) |
| test_gadget | 27.7 (40) | 12.2 (40) | 8.3 (40) |
| test_personality | 26.2 (70) | 23.1 (70) | 27.4 (70) |
| test_deployment_shadow | 25.8 (73) | 18.2 (73) | 20.3 (73) |
| test_sb3_vecenv | 21.2 (18) | 4.3 (18) | 4.4 (18) |
| test_compile | 19.9 (4) | 3.8 (3) | 3.3 (3) |
| test_nav | 19.8 (18) | 10.4 (18) | 76.6 (18) |
| test_deployment_grid | 15.6 (58) | 11.4 (58) | 11.6 (58) |
| test_attack_in_reach | 13.6 (17) | 4.2 (17) | 0.8 (17) |
| test_deployment_loop | 12.3 (100) | 10.8 (100) | 10.0 (100) |
| test_deployment_assemble | 10.7 (30) | 7.6 (30) | 9.0 (30) |
| the other 85 modules | 135.2 (2239) | 91.4 (2199) | 111.1 (2199) |
| **all** | **1918.4 (2925)** | **432.3 (2859)** | **567.5 (2859)** |

Seconds of setup + call + teardown, test count in brackets. Both serial columns ran with the GPU
visible, so the six `gpu` tests ran there. test_integration's 26.5 s is mostly its two sync
checks (18.3 s); test_attack_in_reach and test_gadget hold one each (3.5 s and 3.9 s), and
test_sb3_integration's 10.7 s is mostly the CUDA smoke. The `-n 8` column ran with the GPU
hidden, so those skipped. test_nav's 76.6 s there is its module fixture built once per worker
(above). What is left of test_training is mostly real `train.py` runs: the smoke (48 s)
and two short runs that check the holdout columns and the evaluator (12 s and 10 s).

## GPU: why the heavy tests stay on the CPU

- A decision on CUDA costs ~390 ms nearly flat in `n_envs` (kernel-launch bound: hundreds of
  small kernels per tick, five ticks per decision). The tiny-config CPU step costs ~110-180 ms.
  The GPU only wins at >= 64 envs for many decisions, and none of the expensive tests needs that
  once they are batched.
- The GPU is the training GPU. A suite that wanted it would contend with a run, or skip.
- What does belong on the GPU is the sync-free contract (`torch.cuda.set_sync_debug_mode`).
  Those tests were shrunk to the windows that show it — 53.6 s -> 12.3 s for the three existing
  ones in the serial runs above — and one was added for the configuration the GPU actually
  trains with (below).

## Findings fixed

"Mutant" means the code under test was modified in memory (`inspect.getsource` + replace +
`exec`, restored afterwards; no source file was edited) and the test was run against
it: a corrected test must FAIL under its mutant, and the old test passing under the same mutant
is what made the finding real.

### Tests that could not fail (now can)

| File | What was wrong | Evidence |
|---|---|---|
| test_super.py `..._super_charge_comes_from_player_hits_not_box_hits` | autoreset respawned the match before the dashes, so the box sat at 3000 hp and no dash ever hit it | now `autoreset=False`, asserts all 20 dashes hit (7936 damage per decision, charge 0); a box-farming mutant fails |
| test_autoreset.py | EpisodeStats' episode RETURN was never checked | constant-reward tests; mutants "return = last reward", "return not zeroed", "length not zeroed" all fail |
| test_events.py | kill credit could not see a wrong killer or a zone death | scripted kills by bot, by zone, by hero; mutants "no credit mask", "every kill to the hero" fail |
| test_env.py sentinel override | compared nothing (validate_obs on two separate envs) | slot-by-slot state equality; a "-1 is an override" mutant fails, the old test passed it |
| test_env.py batched smoke | the finiteness check ran on ZeroReward's zeros | runs train.yaml's ShapedReward (12 terms, ~490 non-zero rewards) |
| test_env.py clock test | the clock-only branch could go empty silently | counts clock-only endings (3 at seed 0), asserts > 0 |
| test_env.py latency ring | the two pre-latency reads were unasserted | asserts they read idle, plus the head wrap |
| test_state.py | device test needed a GPU to mean anything; NaN test matched any ValueError | `device="meta"`; `match="NaN/Inf"` |
| test_hero.py | ammo test never checked the max clamp; long-dash loop skipped kinds 5-7; dash-through-body never ran `advance_dash` (it lived in test_movement) | refill to max and never past; all bot kinds; landing x 4.67 > 4.5 (a body-stopped dash ends at 3.7) |
| test_training.py | several: reward asserted once after 20 steps; a 21-wide stub mask that predates auto-aim (hid a `[:, -1]` regression); a FixedTierHook "deterministic across resets" test that built a fresh one-tier hook per run, so nothing was compared across anything; 4 of 6 eval tiers | asserts inside the loop; 22-wide mask with the auto-aim column holding the gadget's opposite; one two-tier hook under two global seeds (same noise, and the tiers differ); all six tiers |
| test_deployment_loop.py `test_pin_thread_pools_caps_torch_and_cv2_from_the_config` | asked for 1 torch thread, which is what every xdist worker now starts with (see "How to run it now"), so a cap that never reached torch passed | asks for a count unlike the starting pool; at one thread the old test passes a "`torch.set_num_threads` dropped" mutant, the new one fails it |

### Physics: combat, projectiles, movement, terrain, boxes

| File | What was wrong | Fix, and the mutant that now fails |
|---|---|---|
| test_combat.py | out-of-combat reset was never checked against a bystander | two stopwatches seeded at 5.0: the victim resets, the bystander keeps 5.0 (mutant: no reset) |
| test_combat.py | the melee test could not see a self-hit or a non-swinger dealing damage | entity 1 swings at entity 2, who does not swing back; 1->2 > 0, 1->1 == 0, entity 2's row all 0 (mutant: `not_self` dropped) |
| test_combat.py | nothing pinned the default i-frame rule | new: flag off by default, a COMBAT hit is blocked, a ZONE hit lands (two gate mutants) |
| test_projectiles.py | volley-into-a-full-buffer compared one field | all 7 `prj_*` fields (mutant: allocation rewrites slot 0) |
| test_projectiles.py | earliest-hit-wins geometry never put two bodies in reach on one tick | both reached on the first tick, preconditions asserted, one tick only (Brock's rocket leaves a hazard sphere that would hit both later) (mutants: identity, lowest index) |
| test_projectiles.py | split-damage tests read the fraction back from params | literal 0.5 x full damage (mutant: `* split_fraction` dropped) |
| test_projectiles.py | "shards stopped by walls" passed when no shard spawned at all | open-ground control must deal damage first (mutant: `_spawn_splits` does nothing) |
| test_movement.py | wall-stop bounds checked the centre, not the body | `x <= face - unit_radius` in all 3 tests (mutant: radius 0) |
| test_terrain.py | 10k-tick stress test on 1 env (2.98 s) | 16 envs x 625 ticks, the same entity-ticks (0.76 s; the radius-0 mutant still fails) |
| test_boxes.py | `cubes_per_box` was never varied | set to 3, asserted in params and in the pickup (mutant: constant 1) |
| 4 files | YAML parsed on every helper call (17.9 ms each) | parsed once per module, deep-copied per call (0.05 ms) |

Deleted: test_movement's dashers-pass-through-bodies test (it never ran `advance_dash`; test_hero now checks a real dash) and test_terrain's hxw-allocation test (`test_batched_leading_dims_64x7` covers its shapes). The 5 files: 121 tests in 9.1 s -> 120 in 5.3 s.

### Bot weapons: combat rules, sniper, artillery, rifle, melee, Edgar, Spike

| File | What was wrong | Fix, and the mutant that now fails |
|---|---|---|
| test_sniper / test_rifle / test_artillery | "never misses at zero noise" fired every tick and passed if any of ~15 shots connected | one shot (clip emptied after the first fire), full effective damage asserted: 2320, 5 x 600, 2080 (mutants: aim noise, `proj_count - 1` pellets) |
| test_combat_rules.py (+ test_rifle) | box lateral-hold: the dead victim had zero velocity, so the clamped fallback row read nothing | victim carries (0, 6); the rifle duplicate deleted (mutant: box guard removed) |
| test_combat_rules.py | four no-NaN copies, unseeded, none covering Edgar, Spike or Bull | one seeded test: 8 envs, every bot kind, a bush and a wall, 300 random states (mutant: NaN aim for kinds 5-7, which all 4 copies passed) |
| test_combat_rules.py | the roster table covered 4 of 7 bot kinds | all 7 plus a completeness assert (3 param mutants) |
| test_melee_sweep.py | "distinct fire times" cannot tell which sub-swing fired | sub-swing indices read off the cone directions must be exactly 0..4 (mutant: k clamped, one fires twice) |
| test_melee_sweep / test_edgar | ammo spend checked one-sided (`<= 1`); the facing-freeze test never had a cooldown running | exact spend of 1.0; rifle cooldown set and `cd > 0` asserted (mutants: no ammo booked, freeze gated on `cd` alone) |
| test_edgar.py | box lifesteal could not fail | env-level: box in reach, Edgar's HP stays exactly 3000 (mutant: box damage heals, to 3756) |
| test_melee_sweep.py | `is_swept` checked two kinds | every Kind, swept == {MELEE, EDGAR} (mutant: Bull swept) |
| test_spike.py | each sweep simulated ~35 victims one at a time (3.5 s) | one env per offset in one simulation (0.4 s); all 71 old per-point values match exactly |

Also: the dormant i-frame mask test now says it is dormant (nothing seeds `ent_invuln_t`); `_run_combo` runs 16 ticks instead of 40 (same events); specs parsed once. The 8 files: 81 tests in 22.7 s -> 77 in 12.5 s.

### Training and config leftovers: test_benchmark, test_config, test_configs_files, test_privileged_critic

| File | What was wrong | Fix, and the mutant that now fails |
|---|---|---|
| test_benchmark.py | full-info byte gap compared with a hand-kept constant | records the tensors the diagnostics actually read (mutant: a sixth read) |
| test_benchmark.py | memory estimate's expected value summed the function's own buffers | built from `observation_space` plus the per-env scalars |
| test_benchmark.py | "samples one action" never checked reuse or mutation | spies the sampler and `step`: one sample, 4 steps, same object, unchanged (mutant: resample per step) |
| test_benchmark.py | the CPU warning was unasserted | `capsys` (mutant: silent CPU path) |
| test_config.py | identical bot weights could not catch a positional read | distinct reverse-order weights (mutant: read by position) |
| test_config.py | deep-merge test checked `dt`, which no override touched | the overridden section's untouched sibling (mutant: shallow merge) |
| test_config.py | no-sync test patched `item`/`numpy`/`tolist` only | also `__bool__`/`__float__`/`__int__`, with a guard that the patch is live (mutant: `if mask.any()`) |
| test_config.py | 9 `pytest.raises` with no `match=` | each matched to its message in config.py |
| test_config.py | a renamed-keys scan duplicating `check_removed_keys` | deleted (`build_params` runs it on the shipped files) |
| test_configs_files.py | the overrides never reached `build_params`'s spec | deep-merged into it |
| test_privileged_critic.py | two rollouts and a second 4-env `build_run` | one rollout + update, `build_model` raise on the existing fixture |

The 4 files: 150 tests in 9.8 s -> 149 in 7.0 s.

### Bots, maps and nav: steering, dispatch, perception, maps, spawn, camera, cadence

| File | What was wrong | Fix, and the mutant that now fails |
|---|---|---|
| test_steering.py | a zero or non-unit strafe vector passed; identical bands could not show a broadcast bug | exact unit counter-clockwise vectors; unequal desired/deadband pairs (mutants: normalize dropped, zeros returned, `keepdim` broadcast) |
| test_bot_dispatch.py | fire/aim covered 4 of 7 kinds, compared fire on one slot, never exercised the decision-tick gate | every kind over steps 0-3, both sides reseeded, >= 4 slots must fire (mutant: decision tick ignores `step_count`) |
| test_bot_dispatch.py | the first-tick reference skipped combat's random draws and checked one kind's rate | replays combat's draws, every slot at its own kind's rate, slot 4 an unseen bot (mutant: every slot at the sniper's rate) |
| test_perception.py | "sticky target": the held target was already the nearest, so a non-sticky pick passed | the other target moved strictly nearer; the pick holds, a fresh pick takes the nearer (mutant: always re-pick) |
| test_maps.py | three validate_map rejection tests each tripped a different rule first | one parametrized case per rule, each on a base map that passes first |
| test_map_generate.py | the crate-spot seed never reached the `free` check; tight == standard, so nothing was rejected | 50 seeds (mutant: `free` dropped); a forced-check wrapper (mutant: wrong seed appended) |
| test_spawn.py | spawn-mask test used an all-True mask; aggression ordering untested; reset checked from zeros | full vs partial mask; a 50/50 rush/camper floor; fields dirtied before the reset (mutants: `N = mask.sum()`, `* 0.0`, `zero_` skipped) |
| test_sim_camera.py | float32 only | float64 too, and the constants' dtype in the cache key (2 mutants) |
| test_audit_attack_cadence.py | matched the report's constant text | whole table rows pinned |
| bot_fixtures.py | `cfg_and_params` overrides reached cfg but not the spec `build_params` reads (latent) | deep-merged into both |

Cost: test_nav's `_walk` uses pointer doubling (same end tiles, still catches a planted loop,
~0.7 s a map); the cadence rollout runs 12 decisions instead of 40 (4.0 -> 0.8 s); test_spawn
builds its map bank once instead of 18 times (cached on the cfg fields `build_map_bank` reads);
test_maps shares one module-scoped bank (6 builds -> 1); `nav._CACHE` is swapped with monkeypatch
instead of cleared; YAML parsed once in 4 files. Deleted: a test that only exercised torch
indexing, the 16-spawn test duplicated in test_map_csvs (25 cases), a re-check of generate's own
condition. Stale: the map count (36), a docstring pointing at a missing section, the convergence
test's 0.958 arithmetic (now 20 ticks and an exact heading). The 14 files: 599 tests in 117.5 s ->
577 in 95.2 s, of which the every-map smoke is 40.8 s.

### Stale or duplicated (removed or corrected)

- test_reward.py: three tests that checked only the `runtime_checkable` Protocol (which every
  callable satisfies), a class the file defined itself, or ZeroReward's zeros for NaN.
- test_integration.py: a scan of agent_obs.yaml for privileged fields (`load_agent_spec` already
  raises on one, pinned field by field just above it); an `assert True` with a "30 ticks" comment
  that meant 30 decisions.
- test_env.py: a hand-spawned gadget-hit test duplicated by test_gadget.py's real throw.
- test_action_repeat.py, test_constants.py, test_training.py: comments and literals that no
  longer matched the code (cooldown floor, ammo trickle, Kind 5-7, the curriculum callback).
- The invariants rollouts now draw the gadget (and count its throws, ~190) instead of attack
  values 0/1 only.

### New coverage

- `test_integration.py::test_the_training_configuration_steps_sync_free_on_cuda` (gpu). The
  three existing sync tests build debug_tiny over default.yaml, where everything train.yaml's
  `run.env_overrides` turns on is off (bot nav, tap aim, the endgame hunt, auto-aim, crates hit
  once per dash, the static-projectile read). This one builds the env with `builder.build_env`
  (shaped reward, curriculum hook, randomization), draws all five attack values with the super
  charged and the gadget armed every decision, and kills eight heroes mid-window so the
  autoreset and the curriculum's tier draw run under the check. It passes: ~115 supers, 8
  autoresets, no sync. 13.5 s.

### Observation pipeline, SB3 wrappers, renderers

| File | What was wrong | Fix, and the mutant that now fails |
|---|---|---|
| test_sb3_vecenv.py | "terminal obs matches the final obs" only showed it differed from the new episode; with one env done, a wrong row could not show | envs 0 and 2 finish together; each terminal obs must equal its own row of `final_observation` in every group (mutant: env i handed row i+1) |
| test_sb3_vecenv.py | the info_mode comparison never reached the done branch: no env finished in its 20 decisions | env 0's hero dies before step 3 in every mode, asserted; 6 decisions instead of 20 |
| test_ascii.py | the "hidden" bot stood 20 tiles off, outside the 21x13 agent crop, so the reveal gate never ran; the letters checked (s/a/m/r) missed e/k/b | rendered 3 tiles away with visibility from 20, a revealed control must show it, and the bot's own `_BOT_CHAR` glyph is used (mutant: reveal gate disabled) |
| test_observation.py | `dist_rank` laid out in index order, where a rank equals an order | slots out of distance order, exact rank [0,3,1,2] (mutant: `dist_rank = order`) |
| test_observation.py | a NaN check on the uint8 view | `schema.validate_obs` at the default config |
| test_obs_schema / test_obs_select | "deterministic" meant rendering twice in one process; the checked-in docs were never read | OBSERVATION.md and the AGENT_OBS docs for agent_obs/deploy4/5/6 compared with a fresh render, plus a deploy6 render test (memory, critic groups) |
| test_obs_select.py | an `... or True` assert; the normalization scales compared with the module's own constants | a real check; the scales pinned as literals (20, 20000, 20), which trained checkpoints depend on |
| test_obs_select.py | the zero-padding check held only because the seed-0 camera happens to cover (0, 0) | two partial-selection cases in the fairness-off tie test (mutant: pad only an all-dead selection) |
| test_obs_select.py | the HP and count bounds modelled a switched-off randomization at the tiny preset's 2 enemies; train.yaml's elite HP and 99 cubes exceed them by design | elite HP and max_cubes read from train.yaml at run time and asserted to pass through unclipped (exactly raw / scale); counts at default.yaml's 9 enemies |
| test_viewer.py | the "partition" test checked overlap with the safe rect only | bands inside the map, pairwise disjoint, total area = map minus the safe rect |
| test_gym_single.py | a "300-step" loop over a 60-decision episode, an assert that could not fail, unseeded actions | the clock starts one tick short; one step must truncate, not terminate |
| 10 config helpers | `{**a, **b}` replaced whole preset sections | `config._deep_merge` (no outcome changed) |

**New gate:** the docs comparison fails whenever a spec or schema change lands without its
docs. The message names the command (`python scripts/dump_obs_schema.py [--spec ...]`). All
five docs matched a fresh render today.

Cost and duplicates: the CLI and CUDA smokes run 2 envs / 8 steps / batch 16 / 16 timesteps
instead of 4/32/32/128 (the CUDA one is now 9.4 s); viewer rollouts are cached per (enemies,
steps, seed), 16 -> 4; deleted as duplicates of the SB3 smoke or test_obs_schema: the PPO
plug-in test, `check_env`, a gadget test's `learn`, two set comparisons, a default-config
forward pass, two missing-field tests; `mkstemp` plus cleanup replaces 16 leaked `mktemp` YAMLs
per run; the host-transfer clock is 10 ticks instead of 50. Comments that still called deploy3
and deploy4 the current specs now say otherwise. The 12 files: 220 tests in 44 s -> 214 in 27 s.

## Left for you

- **BRAWL_SIM_DESIGN.md §1 #2: closed by you on 2026-10-21, no fix.** It is now one of §2's
  accepted divergences. A piercing super fired near the border flies off the map before its range
  runs out, and `check_invariants` flags it. So `test_env.py`'s invariants rollout keeps leaving
  the super out of its draw. `test_maps.py`'s every-generated-map smoke draws it, but random play
  seldom charges it (2 of the 25 maps fire one). Both docstrings now name the accepted divergence.
- **Two cost cuts that would weaken coverage** (bots audit #24, #25), not applied. Both were
  measured on 2026-10-21 in scratch copies of the tests. Neither changes the `-n 8` wall time,
  which the slowest tests set.
  - `test_nav.py` body-walk: 7.9 s serial for both maps.
    - Today a body starts a median 32 tiles from its goal, at most 79. The walk runs 725 and 696
      ticks before the last body arrives.
    - Starts drawn from 9-14 tiles instead of >= 9 cut that to 217 and 168 ticks, about 2.1 s.
      Every body still arrives, and every walk still crosses the 8-tile hand-over, where the
      2026-10-07 limit cycle lived.
    - What goes: walks longer than 14 tiles, which follow the field round several walls. Only the
      table-level walk test would still cover those, and it steps tiles with no body radius.
  - `test_personality.py` nav-off halves: 9.8 s over seven tests. Capping each at its nav-on run's
    length + 40 ticks brings them to 5.9 s.
    - The two backing-off runs and the kite run save nothing, because their nav-on halves run a
      fixed 400 ticks.
    - What goes: the controls would show the bot pinned for 2-13 s of game time instead of
      10-30 s. That proves "slower than nav", not "never escapes", so a scenario that plain
      steering escapes slowly would stop failing.
- **Source note, not a test bug:** in `brawl_sim/core/movement.py:48-50` the facing update does
  not check `active`. An auto-aimed dash (attack 4) taken while the move bin points elsewhere
  therefore faces the bin, not the target, from its first tick.
  - Unaffected: the path and the hits. `start_dash` writes the bearing to `ent_dash_dir` and
    `ent_facing` (hero.py:402-404), and `advance_dash` moves along `ent_dash_dir`.
  - Who reads the facing: `hero.facing_vec` in deploy6's obs (which also carries
    `hero.dash_dir`), the three facing fallbacks (idle-bin dash, idle super with no enemy in
    reach, gadget with nothing visible), and the viewer's arrow. Bots neither dash nor read the
    hero's facing.
  - The live `ShadowHero` copies the rule on purpose (`brawl_deployment/perception/shadow.py:622-626`),
    so sim and live agree.
  - Neither side tests it: the auto-aim parity test fires on the idle bin and compares no
    facing. Tests no longer pretend to check it.
- **Source note, vision side: fixed 2026-10-21 (your call).** `ObjectDetector`'s
  silent-fallback guard (`brawl_vision/object_detection/detector.py:307`) used to fire for
  `device: auto` too:
  - `providers_for("auto")` returns None.
  - The None path turns that into `["CUDAExecutionProvider"]` whenever the GPU build is installed.
  - The guard compared that resolved list, so with no visible device `auto` raised, with a
    message telling you to set `auto`.

  **The fix.** The guard now reads `requested`, the list the caller passed before the None path
  runs. `cuda` still insists, and `auto` takes the CPU.

  **Checked unmocked, both detectors, on this machine:**
  - With the GPU hidden, `auto` builds on CPUExecutionProvider and `cuda` raises.
  - With the GPU visible, both build on CUDAExecutionProvider.

  **The test.** `test_auto_takes_the_cpu_when_cuda_cannot_run_and_cuda_refuses_it` forces the
  fallback, so it runs the same with the GPU visible or hidden. It fails both mutants: the guard
  back on the resolved list, and the guard removed. The `_device_kw()` workaround in
  test_vision_detector.py is gone.

  **What it changes live.** `VisionStack.build` (`brawl_deployment/loop.py:431`, `:438`) builds
  both detectors from vision.yaml's `auto`. A broken CUDA install used to stop the loop at
  startup. It now starts it on the CPU (entity 26.1 ms + projectile 47.2 ms a frame, against
  7.2 + 11.0 on CUDA), and nothing in brawl_deployment logs the provider.
- **Shared tick helpers (physics audit F20/F21):** `test_sniper/test_artillery/test_rifle` step
  the world in a non-production order and never book ammo; `test_melee`'s helper uses a single
  cone where production sweeps. A shared helper in `bot_fixtures.py` in env.py's order would make
  those tests exercise the real sequence. Both were confirmed against env.py (timers first at
  L417, ammo and cooldown booked in `_attack_phase` at L617-618, movement at L683; the melee
  sweep schedule at L664-666) but not built: it is a refactor of four files' tick loops, not a
  test fix. Until then, the one-shot tests empty the clip by hand because those helpers book no
  ammo.
- **Done (your call, 2026-10-21):** `scripts/train.py --smoke` now defaults to 64 envs.
  - It took 52 s instead of 133 s at 8 envs, with the same 8192 timesteps, four stage advances
    and both checkpoints, and it reaches elite.
  - The usage line and `_SMOKE`'s comment say so.
  - `test_train_script_smoke_then_watch_the_checkpoint` now runs `--smoke` exactly as it ships,
    with no override.
- Not applied, by choice: test_compile's parity branch (compile cannot build on this machine, so
  only the error branch runs here); deleting test_hero's charged head-on dash test (its pin is
  named in a hero.py comment); jumping test_gadget's 73rd-observation timer (it measures the
  shipped rearm figure, which is its point); shrinking test_training's `restart_schedules` test
  (~3 s, but it rewrites the LR-schedule arithmetic the `--restart-schedules` resume recipe
  rests on).
