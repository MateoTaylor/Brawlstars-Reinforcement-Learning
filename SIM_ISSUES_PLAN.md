# Sim issues seen in watch.py: verification and fix plan

**Status:** BUILT 2026-10-07 and verified, with second and third rounds the same day (BRAWL_SIM_DESIGN.md #27–#30) and a pathfinding fix from the pre-training check (end of §B), then #32 as option C (end of §B). Trained by you, fresh, 2026-10-07 to 10-08: `runs/mortis_ppo-20261007-161117` (600M decisions, ended in the elite stage), which `configs/deployment.yaml` serves since 2026-10-08 (best_model.zip, the 580M eval: 0.333 across the six tiers). §B, just below, has your answers, what was built from them and what it measured. §0–§7 are the plan as written on 2026-10-06.
**Run measured:** `runs/mortis_ppo-20260930-182748/best_model.zip`, expert tier.
**Probe:** [scripts/probes/sim_issues_probe.py](scripts/probes/sim_issues_probe.py). It writes nothing. It runs the checkpoint two ways:

| Mode | What it is |
|---|---|
| `--mode watch` | watch.py's own env: one pinned tier, no gas jitter, argmax. 8 matches, seeds 0–7. Seed 0 is the match `watch.py --tier expert` shows. |
| `--mode train` | the training env: the run's randomization.yaml and sampled actions. 16 envs, seed 7. |

A finding that shows in both modes belongs to the sim, not the viewer.

---

## B. Built, 2026-10-07

Every change sits behind a key whose default.yaml value is the old sim. Finished runs, watch.py on them and their eval/* numbers stay as they were. configs/train.yaml's `run.env_overrides` turns all of them on.

| Your answer (§7) | Built | Key: default.yaml → train.yaml |
|---|---|---|
| 1. "as real pathfinding as possible without compromising training time seriously" | 1.1–1.4 as planned. Flow fields per 3-tile cell ([maps/nav.py](brawl_sim/maps/nav.py)), built at load and cached per map. `policy.path_toward` steers CLOSE, HOLD_RANGE, TO_BUSH and HUNT_BUSH, and the centre field steers both gas terms. A KITE without line of sight closes. | `bots.nav` false → true, with `bots.nav_anchor_tiles` 3 |
| 2. "campers should switch to hunting in final 4" | 2.2: `policy.effective_person` reads CAMPER and TRAPPER as HUNTER once 4 players are left (the hero counts). The camper fire veto lifts with it. | `bots.endgame_players` 0 → 4 |
| 3. "what is the cap?" | 99, the game's own cap ([Brawl Stars on X, 2019-09-20](https://x.com/BrawlStars/status/1174928524039245824)). Neither 3.A nor 3.B is needed at 99: the supply is one cube per crate plus one per death, and no probe match came near it. | `cubes.max_cubes` 16 → 99 |
| 4. "a maximum of 1 hit per target per dash" | 4.1–4.2: `ent_dash_box_hits`, so a dash hits each crate once, as it already hit each unit once. | `boxes.dash_hits_once` false → true; `boxes.hp` 3000 → 8000 |
| 5. "smoothed bot velocity should be solid" … "most ingame bots use tap to attack" | 5.1: `ent_vel_seen`, a 0.25 s EMA of velocity, read by the bots' lead and lateral fire hold. Tap aim: a LEAD or LOB bot fires at the target's current position, as the game's tap auto-aim does ([1](https://brawlstarsconception.fandom.com/wiki/How_to_tell_if_a_player_is_controlled_by_a_bot), [2](https://pro-brawl-stars-tipsntricks.fandom.com/wiki/Aiming)). The tier's aim noise stays. With tap aim on nothing leads, so the EMA only steadies the hold. | `bots.lead_velocity_tau` 0 → 0.25; `bots.tap_aim` false → true |
| 6. "we'll train fresh … I will do this step manually myself" | Nothing launched. | — |

- **The obs divisors stay where the deployed checkpoint has them** (`core/obs_select.py`: 20000 HP, 20 cubes). Under the 99 cap, HP past 20000 reads above 1.0: a 15000-HP enemy with 13+ cubes, or Mortis with 31+. VecNormalize standardizes it, and a fresh run trains on it from the start.
- **Deploy is unchanged.** No key changes the observation or action width. The live loop reads only #32's `observation.projectile_static_speed`, through the `build_agent_obs` it shares with training, and there it changes nothing: the tracker has already zeroed every track under the same 2.0 tiles/s.

### Fixes the probe found while building

Each round ran the probe in both modes with all seven keys on. It found four problems the plan had not foreseen. Each is fixed and pinned by a test:

1. **The body, not the centre line.** A bot is 0.8 tiles wide. Choosing a goal's anchor by a clear centre line still left the body blocked on 3,105 of 102,935 walkable goal tiles. `nav.walk_blocked` now marches both edges of the body, at load for that choice (24 tiles left) and at run time for `policy._walk_clear`.
2. **Crate pulls wait for a walk the body can make** (the same `_walk_clear`), instead of pushing a bot into a wall corner between it and the crate.
3. **The corner rule.** A body off its tile's centre clipped a wall corner beside its next tile and stopped one step short: 20.6 s in one match. `nav.step_dir` first straightens it onto its tile's centre line.
4. **Pockets at the gas edge.** Where the centre field's path leaves the safe rect (`nav.centre_path_inside`, from a per-tile bounding box of that path), the inward push is off.
   - Following the field there led a CAMPER out into the gas.
   - Aiming it straight at the centre held it against the pocket's inner wall for 7 s, until the gas arrived.

### Verification

- **Tests.** 28 new test functions, and one replaced (`test_advance_dash_damages_box_in_capsule` became the once/every-tick pair):

  | File | New |
  |---|---|
  | test_nav.py (new file) | 9 |
  | test_personality.py | 12 |
  | test_combat_rules.py | 2 |
  | test_hero.py | 2 |
  | test_config.py | 2 |
  | test_env.py | 1 |

  The full suite (`-m "not vision"`): 2842 passed, 2 skipped, none failed.
- **Old runs are unchanged.** With every key at its default, a rollout hashes exactly as before (default `502b28ceb0140809`, auto_aim with the 5 s gas step `ed4b81b274a5fc61`). The probe on the old sim reproduces every number in §0–§5.
- **The probe, before → after.** Same checkpoint, expert tier, same seeds; "after" adds the seven keys with `--set`:

  | Measure | watch, 8 matches | train, 16 envs, seed 7 |
  |---|---|---|
  | Hero won | 1 → 0 | 3 → 1 |
  | Median match | 44 → 16 s | 56 → 28 s |
  | Bot gas deaths (every one before was a stalled bot) | 4 → 0 | 9 → 0 |
  | Stalled share of alive ticks, CAMPER / HUNTER / KITE / RUSH / TRAPPER | 38 / 40 / 18 / 9 / 32 % → 0 / 2.5 / 2.8 / 1.0 / 1.9 % | 19 / 47 / 29 / 39 / 15 % → 0.2 / 0.9 / 5.5 / 3.2 / 0.7 % |
  | Stalled share of the mode's own ticks, TO_BUSH | 60 → 3.0 % | 52 → 0 % |
  | … HUNT_BUSH | 41 → 0 % | 35 → 0 % |
  | … CLOSE | 32 → 0.1 % | 51 → 0.9 % |
  | … HOLD_RANGE | 23 → 0.8 % | 37 → 11 % |
  | … WANDER | 24 → 2.8 % | 28 → 4.3 % |
  | … RETREAT (out of scope) | 14 → 24 % | 13 → 12 % |
  | KITE holding with no line of sight (longest stretch) | 25 % (17 s) → 0 | 48 % (98 s) → 0 |
  | Late game reached (≤ 3 bots, hero alive) | 3 → 0 | 6 → 4 |
  | … bot deaths in it to the gas | 3 → none reached | 4 → 0 |
  | … longest quiet stretch, median (max) | 22 s (51 s) → none reached | 11 s (25 s) → 10 s (44 s) |
  | Hero at the cube cap (16 → 99) | 1 → 0 | 2 → 0 |
  | Cubes left near the hero 5 s or more | 2 → 0 | 0 → 0 |
  | Dashes that passed a crate and broke a full one | 21 of 22 → 0 of 26 | 40 of 44 → 0 of 80 |
  | Hero reversals | 33 → 28 % | 29 → 28 % |
  | Bot shots that hit, after a hero reversal / otherwise | 42 / 60 % → 55 / 52 % | 30 / 46 % → 44 / 47 % |

  The hero now dies much sooner, so the late-game rows rest on 4 matches, and on none in watch mode. HOLD_RANGE pathed only its approach in this round; the second round (below) paths its backing off and RETREAT too.
- **Section [4] counts only crates the dash passed** (changed 2026-10-07). The window alone took every crate on the map, so two full crates that bots shot 20 and 54 tiles away read as one-dash breaks. A replay that tagged each crate hit by its source found no crate hit twice in one dash, over all 215 hero dashes. §0 and §4 quote the old measure (25 of 30 and 49 of 73).
- **The cube economy still meets its 2026-09-25 target** (elite tier, 64 matches): the richest bot holds 7+ cubes at 60 s in 5 of the 6 matches still running (12 of 23 before). Paths get bots to their crates: pulled bots stalled more than 1.5 tiles out fell from 32 % to 6 % of pulled decisions.
- **Cost.** On the training device (CUDA, train.yaml's 4096 envs, maps and randomization, random actions, interleaved blocks): 407 → 440 ms per env step, +8.1 %. About 3 % is the bots' own code (an earlier split the same day). The rest comes with the harder game, whose episodes end 16 % more often (44.8 → 51.9 per step), each a reset. The policy's forward passes and the PPO update are unchanged, so a run's fps falls by less than 8 %. Building the flow fields adds about 9 s to the first env a process builds (eval envs reuse them) and 52 MB.

### The bots are much harder

Against the 2026-09-30 checkpoint at the elite tier (64 matches, the cube-economy probe):

| | Old sim | train.yaml's keys |
|---|---|---|
| Win rate | 0.19 | 0.00 |
| Mean rank | 4.59 | 5.66 |
| Median match | 34 s | 18 s |

Bots that used to die pinned on walls now live to reach the hero, and their shots land more often. Expect the next run's early curriculum to be harder. eval/* does not compare across the change (§6).

### Second round: #27–#29, built 2026-10-07

The first round left three items open (BRAWL_SIM_DESIGN.md #27–#29). Your answers the same day:

| Your answer | Built | Key: default → train.yaml |
|---|---|---|
| #27, the pocket camper: "1 tick per bot and bush seems expensive, but I think it's worthwhile." | The gas rules count a bot's room along its way out: its clearance, less how much nearer the gas the centre field's path from its tile comes than the tile itself (`nav.centre_path_dip`, read off the per-tile box the pocket rule already keeps). The CAMPER and TRAPPER flee, the gas push's ramp, and the bush and waypoint margins all read it. On open ground the dip is exactly 0. | `bots.nav` |
| #28, the straight retreat: "yes, add a destination." | A bot backing off, in RETREAT or a HOLD_RANGE too close to its enemy, paths to `policy.retreat_goal`: 6 tiles straight away from the enemy, kept 1 tile inside the safe rect and on the map. It is the one `path_toward` call each bot already makes, with another goal. | `bots.nav` |
| #29, the reversal cost: "add the 0.01 reversal penalty." | `reward.move_reversal`, charged per decision whose move bin swung 135° or more from the previous decision's, both non-idle (`core/events.move_reversals`, read before the history push overwrites the previous one). It counts decisions, not ticks. | `reward.move_reversal` 0 → −0.01 |

- **Tests.** 20 new test functions: test_move_reversal.py (new file) 10, test_personality.py 5, test_nav.py 2, test_steering.py 2, test_events.py 1. The TERM_NAMES pin in test_attack_in_reach.py gains `move_reversal`. The full suite (`-m "not vision"`): 2863 passed, 2 skipped, none failed; that is 21 more than the first round, since one new function runs at both 8 and 16 bins.
- **Old runs are unchanged.** With every key at its default, both rollouts hash exactly as before (`502b28ceb0140809`, `ed4b81b274a5fc61`).
- **The probe, first round → second round.** Same checkpoint and seeds, the seven keys on in both:

  | Measure | watch, 8 matches | train, seed 0 | train, seed 7 |
  |---|---|---|---|
  | RETREAT: stalled share of its ticks | 23.7 → 0 % | 12.4 → 0.4 % | 11.5 → 0 % |
  | HOLD_RANGE: stalled share of its ticks | 0.8 → 1.1 % | 10.0 → 13.9 % | 11.4 → 6.9 % |
  | Bot gas deaths | 0 → 0 | 1 → 0 | 0 → 0 |
  | KITE: stalled share of alive ticks | 2.8 → 1.4 % | 6.2 → 8.5 % | 5.5 → 3.5 % |

  HOLD_RANGE split by the side of its range band, at seed 0: backing off stalls on 0 % of its ticks and closing on 0.1 %. Every stall is inside the band, where only the strafe moves the bot (29.6 % of those ticks). That branch's steering did not change, and the two seeds moved in opposite directions.
- **The reversal count.** In training mode at seed 7 the env's own flag fired on 1,855 of 7,232 decisions (25.6 %); the probe's 28 % counts only pairs of non-idle moves. At −0.01 that is about 1.2 a 120 s match, against 0.13 for one 1300-HP hit dodged.
- One cube in each training seed lay within 2 tiles of the hero for 5 s or more and was never taken (closest 0.68 and 0.70 tiles, against the 0.6 pickup radius; hero uncapped). Nothing here changed pickup; it is the old policy's play in matches that now run differently.
- **Cost.** The first round's A/B again, with both rounds on and nothing else running: 311 → 342 ms per env step, +9.9 % (the first round read +8.1 %, 407 → 440 ms). The time added is the same, 31 ms against 33. The ratio rose because the old sim's step took 24 % less time this run, though its code gained only the reversal flag, so it was the machine that was slower during the first run. The second round's own cost is below what this A/B resolves. Episodes end at the same rates as before (44.8 → 51.8 per step), and a run's fps falls by less than 10 %.

### Third round: #30, built 2026-10-07

Your answer: "#30 build this as long as it's relatively cheap."

- **Built**, under the existing `bots.nav` key. A bot strafing its enemy (CLOSE, HOLD_RANGE or RETREAT) turns its strafe round when the point 1 tile ahead along the strafe blocks a unit (off the map counts) and the point 1 tile behind does not (`personality.advance_strafe`). The sense is latched per entity in `state.ent_strafe_sign` until the next wall. Its reset value, 0, reads as the slot's old ±1, so on open ground nothing changes. The cost is two terrain lookups per bot per tick.
- **Tests.** 3 new test functions in test_personality.py: the turn happens under nav only; it needs the way back open and then holds; an in-band kiter on open ground strafes as before. The full suite (`-m "not vision"`): 2866 passed, 2 skipped, none failed.
- **Old runs are unchanged.** Both rollouts hash as before (`502b28ceb0140809`, `ed4b81b274a5fc61`).
- **The probe, second round → third.** Same checkpoint and seeds:

  | Measure | watch, 8 matches | train, seed 0 | train, seed 7 |
  |---|---|---|---|
  | HOLD_RANGE: stalled share of its ticks | 1.1 → 0 % | 13.9 → 0.2 % | 6.9 → 0.2 % |
  | KITE: stalled share of alive ticks | 1.4 → 1.0 % | 8.5 → 0.6 % | 3.5 → 1.5 % |
  | Bot gas deaths | 0 → 0 | 0 → 0 | 0 → 0 |

  Inside the band at seed 0, HOLD_RANGE stalled on 0.5 % of its ticks, against 29.6 % before.
- **The large A/B.** A GPU smoke on train.yaml, 1536 envs × 1600 decisions, 18,805 episodes. The 09-30 checkpoint plays half the envs and random actions the other half. It was run twice, once with the turn and once without; the random half reads the same as the checkpoint's half:

  | Stalled share of the mode's ticks, checkpoint's half | without the turn | with it |
  |---|---|---|
  | HOLD_RANGE | 6.49 % | 0.57 % |
  | CLOSE | 1.32 % | 0.14 % |
  | RETREAT | 0.37 % | 0.07 % |
  | TO_BUSH | 1.56 % | 1.44 % |
  | WANDER | 3.62 % | 3.48 % |
  | Every bot | 1.81 % | 1.00 % |

  - **WANDER is now 80 % of all stalls.** A wandering bot has no enemy and never strafes, so the turn cannot reach it; its stalls predate both rounds.
  - **No measurable cost.** The run with the turn took 618 s and the run without it 645 s. Both shared the machine with the test suite.

### Pre-training check, 2026-10-07

Before the next run, the sim was checked with every train.yaml key on:

- **The GPU smoke above.**
  - Nothing invalid: no invariant broke except the known #2, a piercing super that spends a tick past the map border (harmless with `debug_checks` off). No NaN or infinite value appeared in the state, the obs or the rewards, and every row always had a legal action.
  - Capacity: projectile slots peaked at 86 of 192, so #8 never overflowed. The highest cube count was 42 of 99.
  - Outcomes: the checkpoint's win rate falls from 0.59 at easy to 0.01 at elite, and random play wins 0.1 %. No episode truncated, because the gas closes first (the longest lasted 620 of 740 decisions). Every reward term reads sane.
- **`scripts/train.py --smoke`** runs end to end.
- **Three independent code reviews of the whole diff.** One rebuilt HEAD's sim beside the working tree and replayed the same seeded default.yaml rollout on both: all 242 shared state and obs fields matched bit for bit, as did the rewards and the done flags. None found a high- or medium-severity bug. One found a real bug it rated low, since it is rare; it is fixed (next item).
- **Fixed: a bot could stall for good a tile or two short of its goal** (`bots.nav` only, so no finished run saw it).
  - **The bug.** On its goal's anchor tile, with the straight walk blocked, `path_toward` pushed the bot straight at the goal. One tile on, the field led it back to the anchor. It swapped between the two every tick, for good. The sampled walk test made it worse: it could read a walk clear from one spot and blocked a step further along.
  - **How often.** A body following `path_toward` from 9 or more tiles out never reached 59 of the 102,935 walkable goal tiles on the 36 maps, nor 3 of the 22,424 bush tiles. In the reviewer's repro an Edgar (2-tile range) chased a hero standing at such a spot. In three of four cases it came no closer than 2.2–3.6 tiles in 75 s: a spot the agent could have learned to stand on.
  - **The fix** (BRAWL_SIM_DESIGN.md §7, Pathfinding):
    - an exact walk test (`nav.segment_blocked`), so a tail of a clear walk is clear;
    - a hand-over chain in `path_toward`: the goal, then the goal tile's centre, then the field, then the anchor tile's centre;
    - orphan anchors: the 56 walkable tiles with no clear body walk from any nearby anchor are made anchors themselves.
  - **After.** 0 of 102,935 and 0 of 22,424. The reviewer's four Edgar cases reach the hero, with and without the dispatcher's smoothing.
  - **Tests.** 4 new test functions: test_nav.py 3 (one runs on two maps), test_personality.py 1. Two existing nav tests were tightened to the exact walk and the orphans. The end-to-end walk fails on the old code (7 and 5 goals never reached on its two maps). The full suite (`-m "not vision"`): 2871 passed, 2 skipped, none failed.
  - **Old runs are unchanged.** Both rollouts hash as before (`502b28ceb0140809`, `ed4b81b274a5fc61`).
  - **Cost.** On CUDA at train.yaml's 4096 envs, interleaved blocks: 423.6 → 435.4 ms per env step, +11.8 ms (+2.8 %). Episodes end at the same rate (52.4 and 52.5 per step). Against the old sim's step (383.5 ms), train.yaml's keys now cost +13.5 %, against +10.5 % before the fix in the same run.
  - **The GPU smoke again**, same harness, seed and checkpoint, 18,842 episodes.
    - Nothing invalid besides the known #2, no non-finite value, and every row always had a legal action. Projectile slots peaked at 103 of 192 and cubes at 37.
    - Outcomes match the first smoke within noise: the checkpoint wins 0.10 overall, 0.57 at easy and 0.01 at elite.
    - TO_BUSH's stalled share halved: 1.44 → 0.74 % in the checkpoint's half, 1.68 → 0.77 % in the random half. Every other mode moved by 0.1 points or less, and every bot together 1.00 → 0.96 %.
    - Bot gas deaths: 7 → 3.
- **Minor, from the same review.**
  - The probe read the gas clearance without the path dip the bots act on. It now passes the bank (`sim_issues_probe.py`); this changes measurement only.
  - Under `bots.nav`, `scripts/play_manual.py` and the replay viewer build nav tables (about 9 s at start-up) for a bank only their renderer reads. Left as is.
- **Found, pre-existing, not blocking.**
  - **#32, `projectiles.time_to_closest` was unbounded raw seconds.** A slow lobbed shell read hundreds of seconds in the sim (1226 at the smoke's peak), where the live loop reads 0. BUILT 2026-10-07 as your option C ("#32: go with option C"): under `observation.projectile_static_speed` (2.0 in `train.yaml`, 0 = off in `default.yaml`), `core/obs_select` reads a projectile slower than 2.0 tiles/s as still, `vel` 0 and `time_to_closest` 0, before `max_slots` ranks the group. That is the live tracker's `STATIC_TILES_S` rule, and the same function builds the live observation, where it changes nothing. `full_obs` keeps the true values for `info` and the reward.
    - Tests: the rule and its ranking (`test_obs_select.py`), the key's loading and refusal of a negative (`test_config.py`), and a parity pin that `train.yaml`'s value equals `STATIC_TILES_S` and that one crawling shell reads the same row through the sim's path and the live tracker (`test_deployment_projectiles.py`). Both rule tests fail with the rule removed, and with only the velocity zeroed.
    - Agent-side probe (CPU, `train.yaml`, 64 envs × 400 decisions, same seed as the before-numbers): worst `time_to_closest` 34.2 → 8.1 s, p99 4.1 → 3.1 s, rows over 5 s 0.77 → 0.41 %, over 10 s 0.15 → 0 %, slow movers 2.1 % → 0. The bound on screen is 9.2 s mid-map, 15.8 s in a corner.
    - The flags-off fingerprints cannot move (the rule touches only the agent's copy of the observation), and re-run they are unchanged. `train.py --smoke` passes with the key at 2.0, and the full suite reads 2874 passed + 2 skipped.
  - **#31, a bot that dies mid-dash can still hit a crate** in the rest of that dash. It is rare and affects crates only. Closed by you on 2026-10-07 ("#31 isn't an issue, no fix needed"); it is now an accepted divergence in BRAWL_SIM_DESIGN.md §2.

### Not built

- **3.A and 3.B:** not needed at the 99 cap.

### Measured after training, 2026-10-10

- **The jitter (#29) fell by more than half, then stopped falling.** Measured on `runs/mortis_ppo-20261007-161117`, expert tier, with the same seeds as §B.
  - Reversals fell from 28 % to 11 % (watch) and 16 % (train), and path efficiency is 0.46–0.49.
  - The rest stopped falling by 200M (12 % / 16 %).
  - The remaining reversals are nearly as confident as other moves.
  - MODEL_SIZE_STUDY.md has the table.

---

## 0. Verdict: all five are sim issues

| # | What you saw | watch mode | train mode | Cause |
|---|---|---|---|---|
| 1 | Bots stuck and dying to the gas; KITE holding with no shot | All 4 gas deaths were bots pushing into a wall. KITE had no line of sight on 25 % of its holding ticks. | All 9; 48 % | Bot steering aims straight at its goal and cannot go around a wall. KITE's mode never checks line of sight. |
| 2 | Long dead late game | 3 of 8 matches reached ≤ 3 bots. That phase ran a median 35 s, with a median longest quiet stretch of 22 s (max 51 s). | 6 of 16; 20 s; 11 s (max 25 s) | Stuck survivors die to the gas far from the agent. Campers hold their bush by design. |
| 3 | Agent bouncing between two cubes | Both cubes sat next to a hero at the 16-cube cap for their whole 68–74 s. | The cap was reached in 2 of 16 | Not a pickup bug. A capped hero cannot take a cube, and the agent cannot see that it is capped. Uncapped, it took 46 of 48 cubes, median 0.4 s. |
| 4 | One dash breaks a crate | 25 of 30 crate touches broke a full crate in one dash | 49 of 73 | A dash damages a crate on every tick it overlaps it (3–5 ticks), and crates have 3000 HP. |
| 5 | Jittery movement | 33 % of decisions reverse direction (≥ 135°). Path efficiency median 0.34. | 29 %; 0.37 | A learned dodge. Bots hit 42 % of their shots after a reversal, 60 % otherwise (30 % vs 46 % in train mode). |

---

## 1. Bot pathing, and KITE holding without a shot

### Evidence
- **Every gas death was a stuck bot.** All 13 (4 watch, 9 train) had pushed without moving for at least half of their last 3 s.
  - They died a median 11–12 tiles from the map border. They were not hiding at the edge: the gas front caught them pinned on a wall.
  - Passive bots are not the only ones. In train mode, 4 of the 9 were KITE and 4 were RUSH.
- **Every moving mode stalls**, and the modes that walk to a goal stall most. Share of each mode's own ticks spent pushing without moving:

  | Mode (goal) | watch | train |
  |---|---|---|
  | TO_BUSH (nearest bush) | 60 % | 52 % |
  | HUNT_BUSH (next bush waypoint) | 41 % | 35 % |
  | CLOSE (enemy) | 32 % | 51 % |
  | HOLD_RANGE (enemy, at range) | 23 % | 37 % |
  | WANDER | 24 % | 28 % |
  | RETREAT | 14 % | 13 % |

  The gas's inward pull is on for 43 % (watch) and 56 % (train) of WANDER's stalls.
- **KITE.** Look at the holding ticks without a shot:

  | | watch | train |
  |---|---|---|
  | Share with no line of sight | 25 % | 48 % |
  | Of those, standing still | 38 % | 59 % |
  | Longest unbroken stretch | 17 s | 98 s |

### Cause
- **Straight-line seeks.** [bots/personality.py](brawl_sim/bots/personality.py) `movement` steers with `steering.seek(pos, goal)`. The goal is the enemy, the nearest bush tile (`perception.bush_scan`, by straight-line distance) or the next bush waypoint.
  - `terrain.resolve_move` slides along a wall one axis at a time, so a line aimed through a wall pins the bot against it.
  - `advance_hunt`'s docstring already describes this ("the hunter grinds against that wall") and patches it with a timeout.
- **The gas pull.** [bots/policy.py](brawl_sim/bots/policy.py) `zone_avoid_contribution` aims at the safe rect's centre. Its vector is a raw offset (length = distance) with weight up to 2.0.
  - Near the gas it outweighs every other term. A wall between a bot and the centre pins it there until the gas arrives.
  - `zone_contribution`, for a bot already in the gas, aims at the nearest safe point. That point can be behind a wall too.
- **KITE.** `_select_mode` puts a KITE in HOLD_RANGE whenever it has an enemy target. Walls never block sight (§2), so a target behind a wall stays its target, and nothing checks line of sight.

### Reopens
§2 "Bots steer; they never pathfind" (original design). The new evidence: all 13 gas deaths in 24 matches were stalled bots. The goal-seeking modes stall on 32–60 % of their ticks.

### Steps
**1.1 Flow fields, built at load time.** Add `brawl_sim/maps/nav.py`. `build_map_bank` calls it only when `bots.nav` is on, so the 36-map CPU test banks pay nothing.
- **Anchors.** One per `bots.nav_anchor_tiles` square cell (3 tiles, so at most 400 on a 60×60 map), at the walkable tile nearest the cell's centre. A cell with no walkable tile gets none.
- **The centre field.** One extra field leads to the zone centre. Its sources are the walkable tiles within 1 of (30, 30), widened until there is at least one.
- **Distances.** Run a wavefront in torch on the bank's device, one map at a time, over (anchors, H, W):
  - Octile costs: 1 straight, √2 diagonal.
  - A diagonal step only when both orthogonal neighbours are walkable (`~blocks_unit`).
  - Repeat until nothing changes.
  - `validate_map` guarantees one connected component, so every walkable tile reaches every anchor.
- **`nav_next`** (M, A+1, H, W) uint8. 0 means "at the anchor"; 1–8 name the neighbour that minimizes step cost plus distance. Break ties toward the straight bearing to the anchor, so paths over open ground don't zig-zag.
- **`nav_anchor_of`** (M, H, W). For a goal tile, the anchor to route through:
  - The candidates are the anchors in the tile's own cell and its 8 neighbouring cells that have a clear straight walk to it. Check them with one batched `terrain.march` over `blocks_unit` at load.
  - Take the path-nearest candidate, or the path-nearest anchor overall if there is none.
  - Reaching that anchor then leaves a straight walk to the goal.
- **Size.** 36 maps × 401 × 3600 bytes ≈ 52 MB of uint8.

**1.2 A path-aware seek.** Add `policy.path_toward(state, bank, cfg, goal, max_tiles)`, returning (N, E, 2):
- If the goal is within `max_tiles` (8) and `_walk_clear` finds the straight walk clear, return `goal − pos`, today's vector.
- Otherwise look up the next tile, `nav_next[map, nav_anchor_of[goal tile], pos tile]`, and return `normalize(next tile centre − pos) × |goal − pos|`.
- At the anchor (code 0), return `goal − pos`.
- The length stays the distance, so every weight in `steering.combine` keeps its meaning.
- Goals farther than 8 tiles always take the field, which caps the march.

**1.3 Use it.**
- **In `movement`.** Make one call per entity per tick, aimed at the mode's goal:
  - the enemy for CLOSE and HOLD_RANGE;
  - `scan.pos` for TO_BUSH;
  - `hunt.pos` for HUNT_BUSH.

  The result replaces `steering.seek` in `seek_dir`, `bush_dir` and `hunt_dir`. `maintain_range` gets an `approach=` vector for its seek side. That is one march per entity per tick, the same kind each loot pull already runs.
- **The gas terms.** `zone_avoid_contribution` and `zone_contribution` take `bank`. They get their direction from the centre field, with no march, and keep today's magnitudes.
- **RETREAT** has no goal and keeps `steering.flee`. Its 13 % is out of scope here.
- **`advance_hunt`** keeps its timeout as a safety net. Fix the docstring line "there is no pathfinder".

**1.4 KITE needs a shot to hold.**
- `Targeting` gains `enemy_los` (N, E). It is the `los` argument `targeting()` already receives. `tgt.los` switches to the crate's line of sight when the target is a crate, so it is the wrong field.
- In `_select_mode`, a KITE with an enemy and line of sight holds range, or retreats when low, as today.
- Without line of sight it takes CLOSE, which now walks around the wall until it has a shot.
- This adds to the 2026-09-21 rule that a KITE holds no farther than its fire reach.

**1.5 Config.**
- default.yaml gets `bots.nav: false` and `bots.nav_anchor_tiles: 3`, which is today's sim. Every finished run names default.yaml by path.
- configs/train.yaml gets `run.env_overrides.bots.nav: true`, the `action.auto_aim` pattern.
- The KITE rule rides on `bots.nav`.

**1.6 Tests.** In a new tests/test_nav.py, plus tests/test_personality.py:
- On every map in default.yaml, `nav_next` never steps into a wall or cuts a corner, and following it from any walkable tile reaches its anchor.
- From `nav_anchor_of`'s anchor, the straight walk to the goal is clear wherever such an anchor exists.
- A CLOSE bot behind a U-shaped wall reaches its target. Today it pins.
- A bot in the gas behind a wall walks out. Today it dies.
- A KITE without line of sight leaves HOLD_RANGE.
- With `bots.nav: false`, `movement` is bit-identical to today.

**1.7 Docs.**
- BRAWL_SIM_DESIGN.md: replace §2's row and update §7's bot section.
- personality.py: update the rule list in the module docstring.

**Expect** in probe section [1]:
- gas deaths of stalled bots near zero;
- stall shares per mode down to a few percent;
- KITE holding without line of sight near zero.

---

## 2. The late game

### Evidence

| | watch | train |
|---|---|---|
| Left at ≤ 3 bots | CAMPER 2, TRAPPER 2, HUNTER 2, KITE 3 | RUSH 7, KITE 5, CAMPER 3, TRAPPER 2, HUNTER 1 |
| How they died | gas 3 (2 CAMPER, 1 HUNTER), hero 2, bot 2 | gas 4 (all stuck RUSH bots), hero 5 |

Seed 0, the match watch.py shows first, spent 83 of its 128 s in this phase.

### Cause
- **Mostly issue 1.** Stuck survivors die to the gas wherever they are pinned. They never get herded toward the centre and the agent.
- **Campers.** A CAMPER holds its bush until the gas is 2 tiles away, and a TRAPPER moves from bush to bush.
  - At the expert tier both fire on sight: aggression 1.4 is at least 1.25, which lifts the camper veto.
  - Neither one comes looking.

### Steps
**2.1 Measure after §1.** With paths, the shrinking safe square herds survivors toward the centre. Run the probe's section [2] with `bots.nav` on before building 2.2, to see how much of the gap is left.

**2.2 Endgame switch (recommended).** Add `bots.endgame_players`: 0 (off) in default.yaml, 4 in train.yaml, your "final 4".
- `personality.effective_person(state, cfg)` reads CAMPER and TRAPPER as HUNTER once `state.ent_alive.sum(1) <= endgame_players`. The hero counts toward the total.
- `_select_mode` and `fire_allowed` use it in place of `state.ent_person`, so the camper veto lifts with it.
- A HUNTER closes on sight and retreats when low. Otherwise it sweeps the bush waypoints it has not searched yet. By then the zone margin has removed the outer waypoints, so the sweep runs through the middle, where the agent is.
- Test: a CAMPER holding a bush switches to HUNT_BUSH when the alive count reaches 4, and to CLOSE on sight. With the knob at 0, nothing changes.

**2.3 Removal (alternative, no code).** Set `camper: 0` (and `trapper: 0`, to drop those too) under `run.env_overrides.bots.personality_weights` in train.yaml. This also loses the early-game ambushers that make checking bushes worth learning.

**Expect** in probe section [2]: fewer gas deaths in the late game and a much shorter longest quiet stretch.

---

## 3. Cubes the agent cannot take

### Evidence
- **Seed 0.** The hero reached the 16-cube cap at 43 s.
  - Two cubes, at (32.0, 26.4) and (31.7, 25.4), lay within 2 tiles of him for 74 s and 68 s. He came within 0.01 tiles of them.
  - He was capped the whole time, and 53 % of his moves there were reversals.
- **Watch mode overall.** 18 % of decisions were made capped with a cube within 3 tiles, and 51 % of those reversed.
- **Uncapped, pickup works.** Train mode took 46 of 48 cubes, a median 0.4 s after first coming near. Watch mode took 29 of 33.

### Cause
- `combat.collect_pickups` (combat.py:219) leaves a cube for an entity at `cubes.max_cubes`. That is the settled cap: §2 sets it at 16, and §4 says "an entity already at the cube cap leaves the cube".
- deploy5 carries no `hero.cubes`, because nothing live reads it.
- The grid's pickup plane draws every cube.
- So a capped agent sees cubes it can never take, and its critic still prices each one at `cube_pickup` 0.7.

### Steps (pick one; the cap stays)
**3.A Hide the cubes a capped hero cannot take (recommended).** It holds whatever the game does at the cap.
- **Sim.** In `observation.build_obs`, under `cubes.hide_when_capped`, hand `_build_grid` the mask `pku_alive & (ent_cubes[:, 0:1] < params.max_cubes.view(-1, 1))`. Its pickup scatter (observation.py:173) uses that mask in place of `pku_alive`. deploy5 has no pickup row group, so the plane is the only place a cube shows.
- **Deploy mirror** in `brawl_deployment/loop.py`, at the grid call on loop.py:927:
  - Keep a sticky `hero_capped` flag. It sets when the HP numeral reads above `base_hp + 15 × hp_per_cube` on two consecutive decisions. For Mortis that is 14000 (8000 + 15 × 400). Read both numbers from the run's config, never hard-code them.
  - Clear the flag at the match gate.
  - While the flag is set, pass `cubes=()`.
- **Why the flag works.** Only a capped hero can read above 14000, because 15 cubes top out at exactly 14000.
  - The flag sets the first time he is near full HP after capping, and regen gets him there within seconds out of combat.
  - Until then the plane still shows cubes. Live capping is rare: 1 of the 8 watch matches.
- **Config.** false in default.yaml, true in train.yaml's `env_overrides`. The loop applies it only when the run's own config has it.
- **Tests.**
  - A capped hero's plane drops a cube, and an uncapped hero's keeps it.
  - The loop's flag sets on two reads above 14000 but not on one, and it clears at a new match.

**3.B Consume at the cap (smaller, sim only).** `collect_pickups` removes the cube for a capped entity and gives it nothing.
- This is right only if the real game does the same, and you know what the game does at the cap. Your call.
- No deploy change.
- The agent still walks to cubes it cannot use, but it can no longer hover over one.

**Expect.**
- Probe section [3]: no cube left near a capped hero for 5 s or more.
- Section [5]: the `capped_cube` row disappears.

---

## 4. Crate HP

### Evidence
- **Scripted, no policy.** One 2000-damage dash against a 3000-HP crate:
  - Crate 0.5–2.5 tiles straight ahead: hit 3–5 times, for 6000–10000 HP.
  - Crate 0.7 tiles to the side: hit once.
- **Rollouts.** A full crate broke in a single dash 25 of 30 times (watch) and 49 of 73 (train).

### Cause
- [core/hero.py](brawl_sim/core/hero.py) `advance_dash` (L450–451) remembers which units a dash has hit (`ent_dash_hits`) but not which crates. A crate inside the dash capsule takes damage on every tick, as §4 says ("crates are hit on every tick").
- With a flat 3000 HP (§2's divergence list), any dash that passes through a crate breaks it.
- Raising HP to 8000 alone would not fix it: 4 ticks still deal 8000.

### Steps
**4.1 New state.** In `core/state.py`, add `ent_dash_box_hits` (N, E, B) bool next to `ent_dash_hits`. It is about 2.6 MB at 4096 envs × 10 entities × 64 crate slots.

**4.2 Hit each crate once per dash.** Copy the logic the unit hits already use:
- `start_dash` zeroes the dasher's row.
- `advance_dash`, when `boxes.dash_hits_once` is on:
  - masks `box_hit` with `~ent_dash_box_hits`;
  - ORs the new hits in;
  - zeroes the row when the dash finishes.

**4.3 Config.**
- default.yaml: `boxes: {n_boxes: 48, hp: 3000, dash_hits_once: false}`.
- configs/train.yaml's `env_overrides`: `boxes: {hp: 8000, dash_hits_once: true}`.
- Dashes to break a crate, at Mortis's 2000 × (1 + 0.1 × cubes):

  | Cubes | Dashes |
  |---|---|
  | 0–3 | 4 |
  | 4–9 | 3 |
  | 10+ | 2 |

**4.4 Re-check the cube economy.** Bots now need 2.7× the damage per crate, which shifts the cube economy.
- Re-run `scripts/probes/cube_economy_measure.py` against the 2026-09-25 target: the richest elite bot holds 7+ cubes at 60 s in more than half the matches.
- If it falls short, the fix is a separate decision. The 5-tile crate rule is settled.

**4.5 Tests.** `tests/test_hero.py::test_advance_dash_damages_box_in_capsule` passes either way. Replace it with a pin: a dash straight through a crate 1.0 tile ahead hits it once with the flag on, and 4 times with it off.

**4.6 Docs.** In BRAWL_SIM_DESIGN.md, §2's divergence line becomes "8000 in training runs", and §4's "crates are hit on every tick" changes to match.

**Expect** in probe section [4]: no one-dash breaks. Even at 16 cubes a dash deals only 5200.

---

## 5. Jittery movement

### Evidence

| | watch | train |
|---|---|---|
| Decisions that reverse (≥ 135°) | 33 % | 29 % |
| … with no enemy in view and no shot nearby | 21 % | 27 % |
| … capped beside a cube (§3) | 51 % | — |
| Path efficiency over 2 s, median | 0.34 | 0.37 |
| Bot shots that hit: after a reversal / after holding or turning | 42 % / 60 % | 30 % / 46 % |

- **Confident, not dithering.** At a reversal, the margin between the policy's top two moves is 0.33–0.37, the same as anywhere else. It rarely bounces straight back (A-B-A: 5–11 % of decisions).
- **Attacks are already near the ceiling.**
  - The hero attacks on about 10 % of decisions. The ammo bar allows about 9 % for dashes, and supers and gadgets make up the rest.
  - Argmax never drops an attack on a split vote, where values 1 and 4 together outweigh "no attack": 0 % in watch mode, 2 % in train mode.
  - The lever is wasted movement, not the attack count.

### Cause
The jitter is a learned dodge.
- Bots lead their shots on the target's velocity at that tick. `policy.target_info` gathers `ent_vel` into `Targeting.vel`, and combat_rules feeds that to `geo.lead_target` and to the lateral-speed hold.
- A reversal right after a shot throws every lead off.
- Nothing charges for a reversal, so the habit spills into moments when nobody is shooting: 21–27 % of decisions with no enemy in view.

### Steps
**5.1 Bots lead on what they have seen.**
- Add `ent_vel_seen` (N, E, 2) to state: an EMA of `ent_vel` with time constant `bots.lead_velocity_tau` seconds. Update it once per tick at the end of `env._movement_phase` (env.py:674), where `ent_vel` is final.
- `target_info` gathers it in place of `ent_vel`, so the lead and the lateral hold both read it.
- `tau` 0 is today's behaviour. 0.25 s (one decision) is a reasonable start in train.yaml.
- This is a modelling choice about how bots aim; the lead of live bots has never been measured.
- Measure it with the probe's reversal/held split. On today's checkpoint the gap should close.

**5.2 A small cost per reversal.** Add a reward term, `move_reversal`.
- **Detect it** in `env.step`, before `history.push`. A reversal is a decision where:
  - `hist_valid[:, 0]` is true;
  - the previous move bin (`hist_action[:, 0, 0]`) and the current one (`action[:, 0]`) are both non-idle;
  - the two bins are at least 6 apart around the circle.
- **Wire it.** Send it to the reward as `info["move_reversal_tick"]`, the same way `attack_in_reach_tick` travels. Add `move_reversal` to `reward.TERM_NAMES` and as a `RewardConfig` field defaulting to 0.0.
- **Size.** Set −0.01 in train.yaml.
  - At today's rate that is about −1.4 per 120 s match.
  - A dodge that saves one 1300-HP hit is worth 0.13 in `damage_taken` alone.
  - So real dodges still pay, and idle jitter stops paying.
- **Test.** A reversal pays exactly the weight. A turn under 135°, an idle bin and the first decision after a reset pay nothing.

**5.3 Your two ideas, assessed.**
- **Action chunking.**
  - It needs a wider action head, so a new spec and a train from scratch, plus a deploy loop that plays chunks back.
  - It gives up 4 Hz reactions, and Mortis depends on dash timing.
  - The dodge would still pay, so the policy would learn to jitter in chunks.
- **A longer frame window.**
  - The policy is not forgetting: its reversals are as confident as any other move, and it rarely bounces back.
  - More history would leave the payoff in place. It is §1 #20, already open, and it is also a retrain.
- **Not proposed: a decoding change**, such as sampling or summing values 1 and 4. The split vote is only 0–2 %.

**Expect** in probe section [5]: fewer reversals, especially with no enemy in view, and higher path efficiency. 5.1 shows on today's checkpoint; 5.2 shows only after training.

---

## 6. Measuring, order and training

**6.0 Give the probe `--set` first.**
- Add `--set KEY=VALUE` (repeatable) to the probe, using `parse_overrides` from brawl_sim/training/config.py (the parser behind `scripts/train.py --set`).
- The overrides apply on top of the run's own train.yaml.
- Each sim fix can then be measured on today's checkpoint before any training:

```bash
.venv/Scripts/python.exe scripts/probes/sim_issues_probe.py runs/mortis_ppo-20260930-182748/best_model.zip --tier expert --mode watch --episodes 8 --set run.env_overrides.bots.nav=true
```

For train mode, swap in `--mode train --envs 16`.

**Order.**
1. Crates (§4): the smallest change, and self-contained.
2. Cubes (§3).
3. Pathing and KITE (§1): the biggest.
4. Late game (§2): measured after §1, which may close part of it.
5. Jitter (§5): 5.1 can be checked on today's checkpoint; 5.2 shows only after training.

After each step:
- run the full suite (`-m "not vision"`);
- run the probe in both modes with that step's `--set`;
- update the BRAWL_SIM_DESIGN.md rows the step changes.

**Training.**
- **No forced restart.** Nothing here changes the obs or the action width, so no step forces a train from scratch.
  - The next run can fine-tune from this one (TRAINING.md "Resuming") or start fresh. Your call.
  - This policy has learned stuck bots and one-dash crates, so a fine-tune has more to unlearn.
- **Old runs stay put.** Every change sits behind a key whose default.yaml value is today's sim. Old runs, watch.py on them and their eval/* numbers stay as they are.
- **eval/* is not comparable across the change.** A new run's eval/* is scored on a different sim. Compare across the change with the probe, or by evaluating the old checkpoint under the new settings.

---

## 7. Decisions for you

Answered 2026-10-07: §B's first table has each answer and what was built from it.

1. **Pathing.** Reopen §2's "bots never pathfind" for §1's flow fields?
2. **Late game.** The endgame switch at 4 players (2.2), or remove campers (2.3)?
3. **Cubes.** 3.A (hide them) or 3.B (consume them at the cap)?
4. **Crates.** 8000 HP and one hit per dash, in training runs only (train.yaml), with default.yaml left at 3000?
5. **Jitter.** 5.1's τ (0 keeps today's bot aim) and 5.2's weight (0 turns it off).
6. **Next run.** Fine-tune or train fresh?
