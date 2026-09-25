# Sim Overhaul for the Next Run -- Build Plan

**The step-by-step checklist is [SIM_OVERHAUL_STEPS.md](SIM_OVERHAUL_STEPS.md).** This file is the
reasoning behind it; the checklist is what to do next.

Companion to [BRAWL_SIM_BUILD_PLAN.md](BRAWL_SIM_BUILD_PLAN.md) and [bot_overhaul.md](bot_overhaul.md),
written in the same shape: numbered steps, each independently buildable and testable, each with its
own files / implementation notes / acceptance. Written 2026-09-17 against the tree as it stood that
day (last landed change: "deployment cleanup"). Every file and line reference below was checked
against that tree; re-verify line numbers before editing, they drift.

**The five changes, in the operator's words, and the phase that delivers each:**

| # | Change | Phase |
|---|---|---|
| 1 | Bot difficulty is accuracy + aggression, not HP + damage. HP/damage still scale, capped at **1.5x at elite**. At **hard and above, bots favor attacking the hero when the hero is in view**. | **B** |
| 2 | The deployed agent attacks slowly. Confirm the sim is not teaching it to over-conserve attacks in fights; fix what that finds. | **A** |
| 3 | No memory, but **3 frames of local (4-tile) history**: map state around the agent, its own past actions + hp + ammo, nearby enemies. | **H** |
| 4 | **~10 new maps** in the game's style; **fences removed from the sim**. | **M** |
| 5 | The **gadget**: new button, 18 s cooldown, starts charged; a spinner flies up to 2 tiles toward the nearest enemy in 0.2 s and deals 2000 in a radius-1 sphere. | **G** |

Phase **I** (integration) ties them into one new agent-obs spec, one `train.yaml`, the docs, the test
suite and the run checklist. §10 gives the order of work and the dependency graph.

**Standing rules this plan is written under** (from `CONVENTIONS.md`, the design docs and the
operator's earlier decisions; none is re-opened here):

- Hot path stays sync-free: no `.item()`, `.any()`, host loops, or data-dependent shapes inside
  `_run_tick`. Anything per-decision may run in `env.step` outside `_run_tick`, still without a sync.
- `obs_select._MAX_GROUPS = 6` (one host transfer per group), and `sb3_features` accepts **at most
  one uint8 group**. Every observation change below fits inside both.
- **Deploy parity rule:** a field enters the deployable spec only if `brawl_deployment` can supply it
  from `brawl_vision` or from the shadow hero's proprioception. Each obs change below names its
  supplier.
- Mortis's `attack_cooldown: 0.35` and the reload pause are **settled**. Nothing here changes either
  number or measures the pause. Phase A works *around* the cooldown, not on it.
- Ordinary attacks and the super are aimed along the move bin (design §4.4). The gadget's
  nearest-enemy aim is the game's own rule for *that* button, so it is not a sim-side auto-aim.
- No checkpoint is reused. Every width change below is a train-from-scratch, like bot_overhaul D11.
- Bots' `sight_tiles: 14` and bush concealment define "in view". Walls never block sight (bird's-eye).

---

# 0. Decisions

**SETTLED** = fixed by the operator's brief. **PROPOSED** = my call where the brief left room; the
default applies unless overruled, and the step that depends on it says so.

| # | Decision | Resolution |
|---|---|---|
| **S1** | Elite HP / damage cap | **SETTLED: 1.5x.** `elite.hp: 2.0 -> 1.5`, `elite.damage: 1.75 -> 1.5`, every lower tier at or below it (§3, Step B4). |
| **S2** | What "accuracy" means as a knob | **PROPOSED: the four knobs that already exist** -- `aim_noise`, `reaction_delay`, `lead_target`, `decision_period`. They are already the accuracy axes; B4 pushes them harder at elite rather than inventing a fifth. |
| **S3** | What "aggression" means as a knob | **PROPOSED: one new per-kind `SimParams` float `aggression` (1.0 = today's bot)** read by `bots/personality.py` in exactly three places: the retreat HP threshold, the KITE hold distance, and the CAMPER fire veto (Step B3). A tier multiplies it like `hp`. |
| **S4** | How "favor attacking the player" is expressed | **PROPOSED: one new per-kind float `hero_focus` in [0, 1]**, a distance discount on the hero in `perception.select_target`: the hero is picked when `dist_hero * (1 - hero_focus) < dist_nearest_other`, and a bot **switches** to the hero when that holds even if its sticky target is still valid. 0 reproduces today's rule bit for bit. Base 0.5 in `brawlers.yaml` (hard), scaled to 0.85 at elite (Step B2). |
| **S5** | Attack cadence: is there a structural sim cap? | **FOUND, yes (§1.1):** the mask is evaluated only at decision boundaries, so a 0.35 s cooldown against a 0.25 s decision means **one dash per 0.50 s, 70% of the rate the cooldown allows**. Whether that is what the operator saw on the emulator is what Steps A1/A2 measure first. |
| **S6** | The cadence fix | **PROPOSED: a fire latch (Step A3).** An attack is *legal* when the cooldown and dash will both expire inside the coming decision window, and it *fires* on the first sub-tick where the gate opens. Still at most one attack per decision. Chain becomes 0, 0.35, 0.75, 1.10 s (mean 0.375) instead of 0, 0.50, 1.00. **Gated on A1/A2's result**: build it only if the sim's own utilization is high and the cap is what limits it. Requires the deployment mirror (Step A4). |
| **S7** | Gadget in the action space | **PROPOSED: widen the attack column 3 -> 4** (`0 none, 1 attack, 2 super, 3 gadget`), same move bot_overhaul D1 made for the super. Keeps `action.shape == (N,2)`, `_held`, `act_buf`, the mask layout. The cost -- gadget and dash cannot share one decision -- is small: with a 0.35 s cooldown every other decision is attack-masked anyway, and that is where a gadget press is free. |
| **S8** | Gadget gate | **PROPOSED: `alive & gadget_cd <= 0`, nothing else.** Not ammo, not `attack_cd`, not `dash_t` -- the game lets a gadget go mid-dash. |
| **S9** | Gadget projectile representation | **PROPOSED: `Proj.GADGET_SPINNER`, class `ARTILLERY`** (D8's "travels, no damage in flight, resolves at its landing point"). Landing point = hero pos + min(2 tiles, distance to target) along the target bearing, clipped by walls at spawn (Step G2). Flight time fixed at 0.2 s (speed derived), like `proj_flight_seconds`. |
| **S10** | Who the spinner targets | **SETTLED (brief): the nearest enemy.** PROPOSED refinement: nearest *alive enemy revealed to the hero* (the same `vis[:, 0, :]` the observation's `enemy_revealed` uses, so the sim never aims at something the agent could not see); **facing** when nothing is revealed. |
| **S11** | Gadget self-damage | **PROPOSED: never.** The spinner lands 0-2 tiles from the hero with a 1-tile blast; `step_projectiles`' artillery blast currently *can* hit its owner (D4 left Grom able to). Extend the hazard's owner exclusion to `GADGET_SPINNER` or the hero eats his own gadget. |
| **S12** | Gadget vs boxes, super charge, concealment, long dash | **PROPOSED:** boxes in the blast take damage (the blast code already does this and the game's gadget damage breaks crates); gadget hits **do not** charge the super; using the gadget **does** break concealment and out-of-combat (like any attack); it does **not** reset the long-dash idle timer (that mechanic is about his *attack*). Each is one line either way -- see A-G2..A-G4. |
| **S13** | History representation | **PROPOSED: two halves.** (a) A `history` float32 group: for each of the last 3 decisions, the action taken (move one-hot 17 + attack one-hot 4), hp, ammo fraction, displacement of the hero since then, and a valid flag: **78 floats**. (b) **Three extra grid channels** `enemy_hist1..3`: enemies that were revealed at that past decision, rasterized at their *world* positions into the *current* view, restricted to a 9x9 (4-tile Chebyshev) block around the hero. Past terrain is not stacked -- it is static, and the displacement scalar already carries what a shifted terrain crop would. |
| **S14** | Nearby enemies as a vector too? | **PROPOSED: no.** The raster gives tile-resolution trajectories; a per-frame top-K vector would need a sort per frame and add ~30 floats for sub-tile precision the CNN does not need. Revisit if ablation says otherwise (§9 R4). |
| **S15** | Fences | **SETTLED (brief): gone from the sim.** PROPOSED mechanics: `f` leaves `CHAR_TO_TILE` (the loader refuses it), every CSV is converted, README/tests updated. **`Tile.FENCE` stays in the enum** because `brawl_vision`'s 5-class terrain classifier, its label files and the just-retrained `terrain.pt` all index it; removing the member would force a vision retrain for a class that appears in four cells of one clip. The deployment's class-to-channel table maps a predicted FENCE to **water's row** (blocks units, passes shots, `is_water`), the one trained-on combination with a fence's physics (Step M1). |
| **S16** | Map source | **PROPOSED: a seeded generator (`scripts/gen_maps.py`) with its 10 outputs saved as CSVs and validated like any other map.** Hand-authoring ten 60x60 grids is a day of typing with no reproducibility; a generator plus a style-band validator gives the 10 and can give 30 later. Seeds and family per map go in the maps README. |
| **S17** | Map symmetry | **PROPOSED: 8 of 10 point-symmetric (180 degrees), 2 mirror-symmetric (left-right).** The reference screenshots are mostly point-symmetric; the two mirrors keep the agent from learning "the far corner is my corner rotated". |
| **S18** | Where the gadget cooldown comes from at deployment | **PROPOSED: proprioception only.** It starts charged at the match gate and runs 18 s from each modelled press, exactly how the shadow already owns `attack_cd`. No CV read of the button in this pass (§9 R3). |
| **S19** | The match gate anchor | **FOUND (§1.5): it is the gadget button**, chosen *because* the policy never pressed it. After Phase G the policy presses it, and the button's cooldown sweep may drop its ring score under the 0.45 gate. **Step G6 measures that before choosing** between re-anchoring on `super`/`attack` or a 2-of-3 vote. Not optional: a wrong answer ends every match 18 s in. |
| **S20** | New spec name | **`configs/agent_obs_deploy4.yaml`**, by the same "a spec path is what a run names, never edited" rule that made deploy2 and deploy3 new files. |

**Assumptions I am proceeding on** (guessing wrong means rework, not a blocked step):

| # | Assumption | Why | If wrong |
|---|---|---|---|
| A-A1 | The emulator accepts a dash input the instant the previous cooldown ends, and drops (does not buffer) one that lands earlier. | Nothing measured either way; `attack_cooldown: 0.35` was fitted assuming no buffering. | If it buffers, Step A4's timing margin is a non-issue and the lift-commit change becomes optional. If it drops, A4's margin analysis is load-bearing. |
| A-G1 | Gadget damage scales with power cubes the way attacks do (`stats.effective_damage`). | Every damage source in the sim scales; one exception is a trap. | One gather; use `base` instead of `effective`. |
| A-G2 | Gadget hits do not charge the super. | Matches how gadget damage works in the live game for most brawlers. | Drop the exclusion in Step G2 (one mask). |
| A-G3 | Walls stop the spinner (landing point clipped at spawn). | The alternative teaches a through-wall poke that may not exist in Nulls Brawl; a clip costs one `terrain.march`. | Delete the march in Step G2. |
| A-G4 | Using the gadget does not reset `ent_attack_idle_t` (long dash). | The long dash is about *attacking*; the brief calls the gadget a separate button. | Add `gadget_fire` to the `attacked` OR in `_attack_phase`. |
| A-H1 | The deployed loop can hold three decisions of enemy detections in map-frame coordinates with drift small enough to rasterize at tile resolution. | Odometry is bounded per frame (`max_shift_tiles: 2.0`) and re-anchors; 0.75 s of drift is well under a tile in the measured clips. | Restrict `enemy_hist_k` to k = 1 (0.25 s) and zero the rest; the spec shape does not change. |
| A-M1 | 60x60 stays the map size. | Every map, `MapBank`, the view maths and the deployment's map frame assume it. | Not in scope. |
| A-B1 | Bots keep `bots.fight_each_other: true`. | Hero focus is a *preference*, not "ignore everyone else"; without infighting nine bots at elite would be nine bots on one target with no counterplay. | Nothing to change; `hero_focus: 1.0` already expresses "always the hero when visible". |

---

# 1. What the audit found

The facts each phase is built on. Verified in the tree on 2026-09-17.

## 1.1 Decision timing, and the cadence cap

`dt = 0.05`, `action_repeat = 5`: a decision is 5 sub-ticks, 0.25 s. `env._run_decision`
([env.py:336](brawl_sim/env.py#L336)) runs sub-tick 1 with the action and sub-ticks 2..5 with `_held`
([env.py:384](brawl_sim/env.py#L384)), which zeroes the attack column -- **one decision is at most one
attack attempt**, and the attempt is on sub-tick 1 only.

`hero.action_mask` ([hero.py:35](brawl_sim/core/hero.py#L35)) is evaluated at phase 3 of sub-tick 1:
`ready = alive & attack_cd <= 0 & dash_t <= 0`; `fire_ok = ready & ammo >= 1`; `super_ok = ready &
super_ready`. Mortis: `attack_cooldown 0.35`, `dash_duration 0.30`.

Walk one chain. Dash on tick 0 sets `attack_cd = 0.35` (phase 6, after phase 2's decrement). Ticks
1-4 decrement it: at the next decision boundary `attack_cd = 0.15`, `dash_t = 0.05`. Both > 0, so the
mask refuses. Ticks 5-9: `attack_cd` reaches 0 on tick 7 (t = 0.35 s) but nobody can act until the
boundary after tick 9. Next dash: tick 10, t = 0.50 s.

| | per dash | 3-ammo burst | dashes / s |
|---|---|---|---|
| Cooldown allows | 0.35 s | 0.70 s | 2.86 |
| **Sim, today** | **0.50 s** | **1.00 s** | **2.00** |
| Sim with Step A3's latch | 0.35 / 0.40 / 0.35 / 0.40 ... (mean 0.375) | 0.75 s | 2.67 |

So the sim cannot *express* "attack as soon as the dash animation completes"; the best a policy can
learn is one attack per two decisions. This is a **structural cap**, not a learned habit -- but it is
also not the only candidate for what the operator saw, which is why Phase A measures before it
changes anything (§2).

Other levers that touch cadence, all left alone: the reload pause (settled), `action_latency_seconds:
0.001` (0 ticks -- the sim trains with zero input latency while the emulator's attack lands ~167 ms
after the decision, §1.5; see §9 R1), the long dash (`long_dash_seconds: 4.5` rewards *not* attacking
for 4.5 s with a 2x dash -- game-accurate, so it stays, but A1 measures how often the agent waits for
it with an enemy in reach).

## 1.2 Bots: what scales today, and what does not exist

- **Targeting** ([bots/perception.py `select_target`](brawl_sim/bots/perception.py)): sticky nearest
  visible other entity, hero and bots alike, within `bots.sight_tiles: 14`. **No notion of the hero.**
  A bot that has a valid sticky target keeps it until it dies or leaves sight.
- **Fire** (`bots/combat_rules.combat` + `policy.fire_gate`): data-driven from per-kind `SimParams`
  (`fire_needs_los`, `fire_range_fraction`, `attack_arc_rad`, `aim_model`, `aim_noise_*`,
  `lead_target_fraction`, `reaction_delay`, `decision_period_ticks`). CAMPER adds a veto
  (`personality.fire_allowed`: silent until seen).
- **Movement** (`personality._select_mode` -> `_MODE_WEIGHTS`): per-`Person` mode machine; the only
  HP-dependent behaviour is `RETREAT_HP_FRACTION = 0.35` (HUNTER and KITE retreat below it). KITE
  holds `desired_range_fraction * attack_range`.
- **Tiers** ([training/config.py `TIER_FIELDS`](brawl_sim/training/config.py)): `aim_noise,
  reaction_delay, lead_target, decision_period, move_speed, hp, damage`, applied per (env, bot kind)
  by `curriculum.TierApplier` via `_FLOAT_TARGETS` (multipliers) plus the `lead_target` clamp and
  `decision_period` rounding. **Shipped elite: hp 2.0, damage 1.75, aim_noise 0.25, reaction 0.35,
  lead 1.5, decision_period 0.5.** `tests/test_training.py` pins tier order (weakest to strongest on
  every knob) and pins elite to "the operator spec".
- Per-kind params that a kind omits resolve to **0** (`config._resolve_value`); tier multipliers are
  multiplicative, so any new knob a tier should scale needs a **non-zero base in `brawlers.yaml`**.

## 1.3 The observation pipeline

- `core/observation.build_obs` builds the full dict; `_build_grid` ([observation.py:133](brawl_sim/core/observation.py#L133))
  rasterizes 12 channels into a window given an `origin`; entity channels are `_scatter_count`s of
  world positions into that window. **Any world-position list can be scattered into the current
  view the same way** -- that is what makes S13(b) cheap.
- `core/obs_select.py`: YAML spec -> groups; `_MAX_GROUPS = 6`; the grid group selects channels by
  name from `_CHANNEL_INDEX`; float groups flatten trailing dims (`hero.pos_norm` (N,2) -> 2 cols);
  `normalize` divides by unit (`tiles` by map size, `hp` by 20000, `count` by 20).
- `configs/agent_obs_deploy3.yaml` (the run that is deployed today): **5 groups** -- `self` (20
  floats), `enemies` (9 x 7 fields), `projectiles` (12 x 5), `zone` (2), `grid` (10 x 13 x 21). One
  group free.
- `wrappers/sb3_features.BrawlFeaturesExtractor`: the one uint8 group -> 3x3 CNN; every float group
  flattened -> MLP `[256, 256]`. Channel count and float width come from the space, so **S13 needs no
  extractor change**.
- `core/obs_schema.OBS_SCHEMA` must list every `build_obs` field (`tests` pin it both ways);
  `scripts/dump_obs_schema.py` regenerates `docs/OBSERVATION.md` and `docs/AGENT_OBS.md`.
- Deployment: `perception/assemble.ObservationAssembler` fills each group from an explicit
  supplier and **raises on a missing one** (`_require`); `_put_self`
  ([assemble.py:252](brawl_deployment/perception/assemble.py#L252)) is a literal table of `hero.*`
  names read off `ShadowHero.observe()`. `perception/grid.GridBuilder.build` takes the tracked
  enemies/projectiles/crates/cubes and scatters them.

## 1.4 Maps and fences

Interior tile shares of the shipped CSVs (border excluded):

| map | wall | bush | water | fence | spawns | boxes | in rotation |
|---|---|---|---|---|---|---|---|
| open | 2.8% | 1.8% | 0 | 0 | 16 | 16 | yes |
| bushy | 0 | 25.3% | 0 | 0 | 16 | 16 | yes |
| skull_creek | 7.8% | 20.2% | 6.4% | 0 | 16 | 16 | yes |
| feast_or_famine | 2.7% | 26.0% | 0 | 2.2% | 16 | 16 | yes |
| scorched_stone | 14.9% | 28.2% | 0 | 4.5% | 14 | 44 | yes |
| island_invasion | 8.4% | 35.9% | 22.9% | 0 | 16 | 32 | yes |
| walled | 16.1% | 0 | 3.2% | 0.8% | 16 | 16 | no (no bush) |

The reference screenshots (six real Solo Showdown maps): point-symmetric layouts; wall clusters of
2x2 to ~4x6 with grass skirts on one or two sides; small ponds (radius 2-4) and one or two narrow
channels; crates scattered singly and in pairs; two maps with a water border. Nothing like `open`'s
emptiness and nothing like `island_invasion`'s 23% water except at the edge. **Fences appear in none
of them.** Style bands for the generator are derived from this in Step M2.

Fence touch points in the sim: `constants.Tile.FENCE`, `TILE_BLOCKS_UNIT`, `CHAR_TO_TILE["f"]`,
`render/ascii.py` (`=` glyph), `render/viewer.py` (colour), `maps/README.md` legend,
`tests/test_map_csvs.py::test_walled_has_water_and_fences`, three CSVs. In vision:
`terrain/labeling.CLASSES` (5 classes incl. FENCE), the classifier weights' class-name check, label
JSONs, `terrain/palette.py`, `truth.py`'s wall-vs-fence metric, `scripts/vision_*`. In deployment:
`perception/grid.py`'s class -> `[blocks_unit, blocks_projectile, is_bush, is_water]` table, built
from the sim's `TILE_BLOCKS_*` tables.

## 1.5 Deployment facts that constrain the plan

- Perception 12 Hz, decisions every 3rd tick (§1.2 of the design doc). An attack press is **three
  ticks: down, drag, lift** (`control/buttons.py`); the lift fires ~167 ms after the decision. The
  shadow (`perception/shadow.py`) models the dash at the *decision*, so shadow time runs ~170-200 ms
  ahead of the game on every attack. Today's 0.50 s cadence hides that; a 0.35 s cadence would not
  (§2, Step A4).
- `ShadowHero.attack_mask()` reproduces `hero.action_mask`'s formula and is evaluated *before* the
  decision's timers -- conservative by construction. `policy.BrawlPolicy._mask` is `n_move + 3` wide.
- `ShadowHero.check_ammo` is the desync canary: grace period after an attack, a one-pip tolerance
  when the shadow is above the read, three strikes to resync, `resync()` reseeds `attack_cd` to a
  full cooldown ("assume not ready"). Each resync therefore costs up to one cooldown of attacks.
- **The match gate is anchored on the gadget button** (`brawl_deployment/data/control_calibration.json`
  `match_gate.anchor: "gadget"`, threshold 0.45, chosen for its 5.8x/1.8x margin *and* because the
  policy never pressed it). `MatchState.update` exits after `exit_samples` consecutive sub-threshold
  frames and the loop then releases every contact. Button centres (device px): attack (1676, 997),
  super (1463, 998), **gadget (1560, 901)**.
- Telemetry (`loop.TickRow`) records per decision `move_bin`, `attack` (what was *sent*), `ammo_shadow`,
  `ammo_cv`. It does not record the mask or whether an enemy was in reach -- Step A2 adds both.

---

# 2. Phase A -- attack cadence

Goal: know *why* the deployed agent attacks slowly before changing anything, then remove the sim-side
cap if the measurement says the cap is what binds.

## Step A1 -- sim-side cadence audit

**Files:** `scripts/audit_attack_cadence.py` (new), `tests/test_audit_attack_cadence.py` (new, on a
scripted rollout, not a checkpoint).

Roll the deployed checkpoint (`runs/mortis_deploy3_elite-*/best_model.zip`, deterministic) in the sim
at the elite fixed tier for >= 200 episodes with `tick_hook` recording, per decision:
`hero.can_attack`, `hero.attack_cd`, `hero.dash_t`, `hero.ammo`, `hero.long_dash_ready`,
`hero.attack_idle_t`, the chosen attack column, and **`enemy_in_reach`** = any enemy alive, revealed
to the hero, within `dash_distance + dash_radius + unit_radius` (~3.9 tiles).

Report (one table, one histogram each, written to `runs/audit/`):

1. **Utilization** `P(attack | legal & enemy_in_reach)`. High (>= 0.8) means the policy attacks when
   it can; low means learned conservation.
2. **Inter-attack interval** during fights (consecutive attacks with `enemy_in_reach` throughout).
   Expect a spike at exactly 0.50 s and a second cluster at ~3 s (reload).
3. **Phasing loss**: share of decisions with `enemy_in_reach & ammo >= 1 & attack_cd in (0, 0.20]`
   -- the decisions the latch (A3) would unlock.
4. **Long-dash waiting**: share of fight decisions with `attack_idle_t >= 4.0 & enemy_in_reach &
   legal & no attack`.
5. **Ammo at first attack of a fight** (does it enter fights empty because it spent ammo on boxes?).

**Acceptance:** the script runs on CPU (the GPU may be busy with a training run -- check before using
it); the test feeds a hand-scripted policy (attack whenever legal) and asserts utilization 1.0 and an
interval histogram with mass only at 0.50 s -- which is also the regression pin for the cap itself.

## Step A2 -- deployment-side cadence telemetry

**Files:** `brawl_deployment/loop.py` (`TickRow`, `_decide`), `scripts/audit_attack_cadence.py`
(`--telemetry runs/deploy/<run>` mode), `tests/test_deployment_loop.py`.

Add to `TickRow`: `attack_legal` (the tuple the shadow handed the policy, encoded as an int bitmask),
`attack_cd_shadow`, `enemy_in_reach` (same radius as A1, off `tracked.enemies` in tiles), `resync`
(bool, this decision triggered one). All scalars, per the telemetry rule.

The audit script then computes A1's five statistics from a telemetry file so the two sides are
compared **on the same definitions**. Add a sixth, deployment-only: **resyncs per minute in fights**
and the ammo error that tripped each.

**Reading the result (the decision tree for A3/A4):**

| Sim utilization | Deploy intervals vs sim | Cause | Do |
|---|---|---|---|
| high | ~equal (0.50 s spike both) | the structural cap (§1.1) | **A3 + A4** |
| high | deploy longer | shadow/mask/resync side | A4's lift-commit + canary review first; A3 after |
| low | any | learned conservation | reward/curriculum review (§9 R2) before A3; A3 still worth having |

## Step A3 -- the fire latch (sim)

**Only after A1/A2 say the cap binds.** Do Phase G's Step G3 first: both edit `action_mask`,
`decode_action` and `_run_decision`, and G3 widens the column the latch reads.

**Files:** `brawl_sim/core/hero.py` (`action_mask`, `decode_action`, `tick_timers`),
`brawl_sim/core/state.py` (`_ENTITY_FIELDS`), `brawl_sim/env.py` (`_run_decision`, `_decode`,
`_attack_phase`), `brawl_sim/config.py` (`EnvConfig.fire_latch: bool`, default **true** for the new
run, `false` reproduces today), `configs/default.yaml`, `brawl_sim/core/obs_schema.py`
(`hero.can_attack` description), `tests/test_hero.py`, `tests/test_action_repeat.py`,
`tests/test_env_step.py` (or wherever the "one attack per decision" pin lives -- `grep -n "_held"
tests/`).

**Semantics.** Let `W = (action_repeat - 1) * dt` (0.20 s): the latest sub-tick of a window at
which a latched press can still fire.

```
# hero.action_mask, latch on
ready    = alive & (attack_cd <= W) & (dash_t <= W)          # was: <= 0
fire_ok  = ready & (ammo >= 1)                               # ammo only rises inside a window: a clip
super_ok = ready & super_ready                               #   legal now is legal on the fire tick
```

New state field `ent_fire_latch (N,E) i64` (0 none / 1 attack / 2 super / 3 gadget) and
`ent_fire_latch_t (N,E) i64` (sub-ticks left). Hero row only is ever written; bots keep 0.

```
# env._decode (phase 3), every sub-tick
col = effective_action[:, 1]                                 # _held zeroes this on sub-ticks 2..K
set = col > 0                                                # sub-tick 1 only, in practice
latch   = where(set, col, where(latch_t > 0, latch, 0))
latch_t = where(set, action_repeat, clamp(latch_t - 1, min=0))
fire      = (latch == 1) & can_attack_now                    # the OLD strict gates: cd<=0, dash_t<=0, ammo>=1
super     = (latch == 2) & can_super_now
gadget    = (latch == 3) & can_gadget_now
# _attack_phase clears latch (and latch_t) on the row that fired this tick.
```

Why `latch_t` counts `action_repeat` ticks from the *arrival* of the bit rather than "until the window
ends": with `action_latency_ticks > 0` the fire bit surfaces late in the window through `act_buf`,
and a window-end clear would give it nothing to wait on. Each press gets exactly one window of
chances, wherever that window starts. `_held` still zeroes sub-ticks 2..K, so a press can only be
*set* once per decision -- "one decision, at most one attack attempt" survives unchanged.

**Why the strict gates stay inside `_attack_phase`.** The mask says "legal by the end of this
window"; the phase-6 `can_attack` safety net says "legal right now". Keeping both is what makes an
override or a bot unable to force an early attack, exactly as today.

**Acceptance:**
- Chain test: scripted hero attacking every decision with a target in reach produces dashes at ticks
  0, 7, 15, 22, 30 (0.35 / 0.40 alternating). With `fire_latch: false` it is 0, 10, 20 (today's pin).
- `_held` test still holds: one attack per decision even with `action_repeat=5` and the bit held high.
- A press latched on sub-tick 1 whose cooldown does *not* expire in the window never fires, and the
  latch is 0 at the next boundary (no carry-over).
- `hero.can_attack` in the observation equals the new `fire_ok` (it is what the policy is told).
- `test_action_repeat.py`'s invariance tests updated for the new timing; `check_invariants` unchanged.
- `graphify update .` after the change.

## Step A4 -- the fire latch (deployment mirror)

**Files:** `brawl_deployment/perception/shadow.py` (`attack_mask`, `act`, `_tick`, new `commit`),
`brawl_deployment/control/buttons.py` (`press` becomes *arm*; the down is emitted by `settle()`),
`brawl_deployment/loop.py` (`_decide`, the per-tick `settle` call), `brawl_deployment/config.py`
(`validate`: the latch needs `W` in ticks), `tests/test_deployment_shadow.py`,
`tests/test_deployment_control.py` (the buttons tests live here), BRAWL_DEPLOYMENT_DESIGN.md §4.4 / §6.3 (append, do not rewrite).

Two changes, and the second is what makes the first safe:

1. **Mirror the latch.** `attack_mask()` uses `attack_cd <= W & dash_t <= W`. `act()` arms a press;
   `_tick()` fires the modelled attack on the first sub-tick where the strict gate opens within
   `action_repeat` sub-ticks, else drops it. `Buttons` sends the **down on the perception tick on
   which the shadow fired**, then drag, then lift as today.
2. **Commit the dash at the lift, not at the decision.** Today the shadow starts `attack_cd`/`dash_t`
   when the decision is made, ~170-200 ms before the game does. With a 0.50 s cadence the next lift
   always lands after the real cooldown; at 0.35 s the margin is **zero** (§1.1 arithmetic: real
   cooldown ends at `S_A + L + 0.35`, next lift lands at `S_A + 0.35 + L`). `Buttons.settle()` calls
   `shadow.commit()` when it emits the lift; `_tick` starts the timers there. Shadow time then trails
   the game only by input latency, and the margin is ~`L - latency` (>= 100 ms).

   Side effect, and it is an improvement: `hero.dashing`/`dash_t` in the deployed observation now
   describe the dash that is actually on screen.

**Measurement gate before enabling on a live run:** `scripts/measure_reload.py`-style job: a chain
of six attack presses at the latched cadence against a wall in a real match (not Training Grounds --
see the live test venue rule), counting ammo drops through the ~1 s HUD lag. Six presses, six pips
consumed (over two clips), zero resyncs. If presses are dropped, A-A1 is wrong in the bad direction:
keep the latch in the sim (it is still the better training signal) but make the deployed shadow's
`W` a config knob and set it to 0 until the timing is understood.

**Acceptance:** shadow unit tests mirror A3's chain test tick for tick; a buttons test asserts the
down is deferred to the fire tick and the lift still lands two ticks later; loop test asserts a
decision with a latched-but-never-fired press sends nothing and logs it.

---

# 3. Phase B -- difficulty: accuracy + aggression

## Step B1 -- the two new per-kind params

**Files:** `brawl_sim/config.py` (`PER_KIND_FIELDS`, `validate`), `configs/brawlers.yaml` (every bot
block; the hero block gets nothing), `tests/test_config.py`, `tests/test_configs_files.py`.

```
("hero_focus", "hero_focus", _F32),    # [0,1]; 0 = plain nearest-target. Base 0.5 for every bot kind.
("aggression", "aggression", _F32),    # 1.0 = today's bot. Must be > 0 for any bot kind (validate).
```

`brawlers.yaml`: `hero_focus: 0.5` and `aggression: 1.0` on each of the seven bot kinds, with a
comment naming the tier multiplier that scales them. `validate()` rejects a bot kind with
`aggression <= 0` (a missing key resolves to 0 and would divide by zero in B3) and `hero_focus`
outside [0, 1]. The `_REMOVED_KEYS` pattern is not needed (nothing is renamed).

**Acceptance:** `load_config` on every preset in `configs/` passes; a bot block without
`aggression` fails with a message naming the kind; `test_single_archetype_preset_pins_sniper` still
passes.

## Step B2 -- hero focus in targeting

**Files:** `brawl_sim/bots/perception.py` (`select_target`), `brawl_sim/bots/policy.py`
(`all_bot_intents` passes `params`), `tests/test_perception.py` / `tests/test_bots_policy.py`.

`select_target(state, vis, cfg)` becomes `select_target(state, vis, params, cfg)`:

```
focus = stats.gather_kind(params.hero_focus, state.ent_kind).unsqueeze(-1)       # (N,E,1)
scale = ones_like(dist); scale[:, :, HERO] = 1 - focus[..., 0]                    # discount column 0 only
dist_eff = where(candidates, dist * scale, inf)
nearest = argmin(dist_eff, -1); has_candidate = candidates.any(-1)
picked = where(has_candidate, nearest, -1)
prefer_hero = has_candidate & (nearest == HERO) & (focus[..., 0] > 0)             # visible AND wins the discount
target = where(prefer_hero, HERO, where(current_valid, current, picked))
```

- `focus = 0` reproduces today's `where(current_valid, current, picked)` exactly (`prefer_hero`
  false everywhere) -- pin that with the existing tests untouched.
- `prefer_hero` **overrides stickiness**: a bot fighting another bot turns on the hero the moment the
  hero is in view and wins the discounted comparison. Without this, "favor the hero" would only
  apply to bots that happened to be idle.
- The hero's own row (slot 0) is computed and discarded as everywhere else.
- Sight is unchanged: `vis` is still `bot_visibility(...)`, so a concealed or >14-tile hero is not a
  candidate. "If the player is in view" is exactly the existing `vis[:, e, HERO]`.

**Acceptance:** with `hero_focus = 0.5`, hero at 7 tiles and another bot at 4, the bot picks the
hero (3.5 < 4); at hero 9 tiles it keeps the bot (4.5 > 4); a sticky bot target is dropped for the
hero the tick the hero becomes visible at a winning distance; with `hero_focus = 1.0` the hero is
chosen whenever visible; with `0.0` every existing `select_target` test passes unchanged.

## Step B3 -- aggression in the personality layer

**Files:** `brawl_sim/bots/personality.py` (`_select_mode`, `movement`, `fire_allowed`),
`brawl_sim/bots/policy.py` (`targeting`: `desired_range`), `tests/test_personality.py`.

`aggression` (gathered per entity once in `movement`, passed down) enters in three places and no
others:

| where | today | with `aggression = a` |
|---|---|---|
| retreat threshold (`_select_mode`, HUNTER + KITE) | `hp_frac < 0.35` | `hp_frac < clamp(0.35 / a, 0.05, 0.90)` -- elite (1.7) retreats below 21%, easy (0.6) below 58% |
| KITE hold distance (`policy.targeting` -> `desired_range`) | `desired_range_fraction * attack_range` | `* clamp(1 / a, 0.6, 1.4)` -- aggressive kiters hold closer, timid ones farther |
| CAMPER fire veto (`fire_allowed`) | silent until seen | silent until seen **and** `a < 1.25`; at 1.25+ a camper fires on sight |

Nothing else: `_MODE_WEIGHTS` stays a constant table (it is gathered as one tensor; scaling a column
per entity would cost a second gather for a marginal behavioural change). Personality *sampling* is
untouched -- aggression is per (env, kind) via the tier, personality is per entity via
`spawn.sample_personalities`; the two compose.

**Acceptance:** with `a = 1.0` every `test_personality.py` case passes unchanged; a HUNTER at 25%
HP with `a = 1.7` is in CLOSE, with `a = 1.0` in RETREAT; a CAMPER unseen with `a = 1.5` fires,
with `a = 1.0` does not; `desired_range` for a KITE Brock scales as the table says.

## Step B4 -- the tier table

**Files:** `brawl_sim/training/config.py` (`TIER_FIELDS` += `aggression`, `hero_focus`, both default
1.0), `brawl_sim/training/curriculum.py` (`_FLOAT_TARGETS` += both; `hero_focus` clamped to [0, 1]
after scaling, like `lead_target`), `configs/train.yaml`, `tests/test_training.py`
(`test_shipped_tiers_are_listed_weakest_to_strongest`: `aggression` and `hero_focus` never fall;
`test_shipped_elite_tier_is_the_operator_spec`: the new spec below).

| tier | aim_noise | reaction | lead | decision_period | move_speed | **hp** | **damage** | **aggression** | **hero_focus** (x0.5 base) |
|---|---|---|---|---|---|---|---|---|---|
| easy | 3.0 | 2.2 | 0.25 | 2.0 | 0.90 | 0.75 | 0.70 | 0.6 | 0.0 -> 0.00 |
| medium | 1.8 | 1.6 | 0.60 | 1.5 | 0.95 | 0.90 | 0.85 | 0.8 | 0.4 -> 0.20 |
| hard | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 -> 0.50 |
| veteran | 0.6 | 0.6 | 1.0 | 1.0 | 1.0 | 1.15 | 1.15 | 1.2 | 1.3 -> 0.65 |
| expert | 0.4 | 0.45 | 1.25 | 0.75 | 1.0 | 1.30 | 1.30 | 1.4 | 1.5 -> 0.75 |
| **elite** | **0.20** | **0.30** | 1.5 | 0.5 | 1.0 | **1.50** | **1.50** | **1.7** | **1.7 -> 0.85** |

Reading it: below hard, accuracy and aggression fall faster than HP/damage (an easy bot is a bad
shot that runs away, not a paper bot); above hard, accuracy and aggression keep climbing while
HP/damage plateau at 1.5. Stages and gates (`hard_intro -> veteran_intro -> veteran -> expert ->
elite`, advance at 0.35, window 2000, 75 M cap) are unchanged -- the *content* of a tier changed, not
the walk. `hard` is still "brawlers.yaml as authored", which is now a bot that favors the hero.

**Acceptance:** `test_training.py` passes with the new pins; `CurriculumManager` applies both new
multipliers to bot columns only (extend `test_manager_applies_tier_multipliers_to_bot_columns`);
`scripts/train.py --dry-run` (or the equivalent config-only path) builds; an eval at the elite fixed
tier shows the win-rate drop you would expect from a harder elite (record the number in the run
notes; there is no target, it is the new baseline).

---

# 4. Phase G -- the gadget

## Step G1 -- constants, params, state

**Files:** `brawl_sim/constants.py`, `brawl_sim/config.py` (`PER_KIND_FIELDS`, `validate`),
`configs/brawlers.yaml` (hero block), `brawl_sim/core/state.py`, `brawl_sim/core/hero.py`
(`tick_timers`), `CHARACTER_DETAILS.md` §1 (append the gadget), `tests/test_constants.py`,
`tests/test_state.py`, `tests/test_hero.py`.

- `Proj.GADGET_SPINNER = 7`; `N_PROJ_KINDS = 8`; `PROJ_CLASS_OF[GADGET_SPINNER] = ProjClass.ARTILLERY`.
  Then follow the adding-a-brawler checklist's projectile half: `obs_schema` one-hot widths
  (`projectiles.kind_onehot` 7 -> 8), `render/viewer.py` and `render/ascii.py` projectile tables,
  assets README, `grep -rn "N_PROJ_KINDS\|len(Proj)\|== 7" tests/`.
- Per-kind params, all 0 for every bot (= "no gadget", the `super_charge_hits: 0` convention):

  ```
  ("gadget_cooldown", "gadget_cooldown", _F32),          # 18.0
  ("gadget_range", "gadget_range", _F32),                # 2.0 tiles
  ("gadget_flight_seconds", "gadget_flight_seconds", _F32),  # 0.2
  ("gadget_damage", "gadget_damage", _F32),              # 2000
  ("gadget_radius", "gadget_radius", _F32),              # 1.0
  ```
  `validate()`: if `gadget_cooldown > 0` then range, flight and radius must be > 0.
- State: `ent_gadget_cd (N,E) f32` in `_ENTITY_FIELDS`. `zero_` gives 0 = **ready**, which is the
  brief's "starts fully charged"; nothing in `spawn` needs to set it. `hero.tick_timers` decrements
  it every tick (clamped at 0) for every entity; bots' stays 0.
- `CHARACTER_DETAILS.md` §1: the gadget's five numbers and the S10-S12 rules, so the source of truth
  for stats stays the one file.

**Acceptance:** `N_PROJ_KINDS == len(Proj)`; `check_invariants` passes with the new field; a hero
kind loads with the five numbers; a bot kind resolves all five to 0.

## Step G2 -- the spinner: spawn and resolution

**Files:** `brawl_sim/core/projectiles.py` (new `spawn_gadget`; `step_projectiles` owner exclusion
and charge exclusion), `brawl_sim/core/hero.py` (new `gadget_target`), `brawl_sim/env.py`
(`_attack_phase`, `_projectile_phase`, `_bookkeeping`), `tests/test_projectiles.py`,
`tests/test_gadget.py` (new).

**`hero.gadget_target(state, vis, params, cfg) -> (dir (N,E,2), travel (N,E))`** for every entity
(computed for all, read for the hero; bots have `gadget_cooldown 0` and never fire one):

```
revealed = vis[:, :, :] & alive.unsqueeze(1) & not_self                   # (N,E,E): what e can see
dist = |pos_j - pos_e|; dist_eff = where(revealed, dist, inf)
nearest = argmin(dist_eff, -1); has = revealed.any(-1)
to = normalize(pos[nearest] - pos_e); dir = where(has, to, facing_vec)     # facing when nothing is revealed
travel = where(has, minimum(gadget_range, dist[nearest]), gadget_range)
hit, hit_pos, _ = terrain.march(bank.blocks_proj, map_id, pos_e, dir, travel, cfg)     # A-G3
travel = where(hit, |hit_pos - los_step * dir - pos_e|, travel)
```

For the hero `vis[:, HERO, :]` is the same mask `enemy_revealed` in the grid uses, so the spinner
never targets what the agent cannot see (S10). `terrain.march` is the same call `step_projectiles`
makes per projectile; one more (N,E) march per tick is in budget.

**`projectiles.spawn_gadget(state, fire, origin, dir, travel, damage, params, cfg)`** -- a sibling
of `spawn_supers`, one slot per firing entity via `alloc_slots`, written with `_write_slots`:
`target = origin + dir * travel`, `vel = dir * travel / gadget_flight_seconds`, `dist_left = travel
+ eps`, `aoe = gadget_radius`, `damage`, `kind = GADGET_SPINNER`, `cls = _ARTILLERY`, `pierce =
false`. `step_projectiles`' existing artillery path then does the rest: `arrived` (dot <= 0) after
exactly 4 ticks, `det_pos = prj_target`, blast on entities and boxes within `prj_aoe`. `travel` may
be tiny (enemy on top of the hero): `arrived` is true on tick 1 and it detonates in place -- test it.

**Two edits inside `step_projectiles`:**

1. Owner exclusion (S11): `no_self = is_hazard | (prj_kind == GADGET_SPINNER)`;
   `owner_ok = ~no_self.unsqueeze(-1) | (prj_owner.unsqueeze(-1) != entity_idx)`. Grom's
   self-detonation behaviour (D4) is untouched because the flag is per kind.
2. Charge exclusion (S12, A-G2): return one more `(N,E,E)` bool, `charge_hit`, that is
   `dmg_by > 0` **minus** the gadget's contributions. `env._projectile_phase` passes it up and
   `_bookkeeping` derives `hits` from `charge_hit` for the projectile phase instead of from
   `dmg_by_total > 0`. (Dash hits and volleys keep charging as they do today.)

**`env._attack_phase`:** decode yields `gadget_fire`; `can_gadget = alive & (gadget_cd <= 0) &
(gadget_cooldown > 0)`; on fire set `ent_gadget_cd = gadget_cooldown`; add `gadget_fire` to the
`attacked` OR that breaks concealment and out-of-combat (S12) but **not** to the long-dash reset
(A-G4); `damage = stats.effective_damage(...)` scaled by `gadget_damage / base_damage` -- or add a
`stats.effective_gadget_damage` that applies the same cube bonus (A-G1); call `spawn_gadget`.

**Acceptance (`tests/test_gadget.py`):**
- Enemy revealed at 1.5 tiles: spinner lands on it, enemy takes 2000 at tick 4, hero takes 0.
- Enemy at 2.8 tiles: lands at 2.0, enemy (0.8 from the landing point) takes 2000.
- Enemy at 3.5 tiles: lands at 2.0, enemy takes 0.
- Two enemies inside the radius: both take 2000; a box inside takes 2000.
- Enemy behind a wall at 1.5 tiles: landing point clipped at the wall, enemy takes 0.
- Nothing revealed: flies 2 tiles along facing.
- Hidden (bushed, unrevealed) enemy at 1 tile and a revealed one at 2: targets the revealed one.
- `super_charge` unchanged after a gadget-only hit; changed after a dash hit.
- `ent_gadget_cd == 18.0` after firing, decrements 0.05 per tick, and the mask refuses until 0.
- Reveal timer set on gadget use; `attack_idle_t` not reset.
- `max_projectiles` accounting: a gadget consumes one slot for 4 ticks.

## Step G3 -- action space: 3 -> 4

**Files:** `brawl_sim/config.py` (`action_nvec = (n_move_bins + 1, 4)`), `brawl_sim/core/hero.py`
(`action_mask`: `attack = stack([no_fire_ok, fire_ok, super_ok, gadget_ok])`; `decode_action`:
`gadget = attack_col == 3`), `brawl_sim/env.py` (`_decode`, `_attack_phase` signature, `_held`
unchanged -- it zeroes the whole column), `brawl_sim/core/obs_schema.py` (`action_mask` width),
`brawl_sim/wrappers/sb3_vecenv.py` (`action_masks`), `scripts/play_manual.py` (a key for the
gadget), `scripts/record_rollout.py` / `scripts/watch.py` if they decode the column, `docs/`.

`gadget_ok = alive & (gadget_cd <= 0) & (gadget_cooldown > 0)` -- **not** ANDed with `ready` (S8).
Then `grep -rn "n_move_bins + 1, 3\|, 3)\b\|(N, 20)\|\[0, 1, 2\]" tests/ brawl_sim/ brawl_deployment/`
for roster-sized literals -- the adding-a-brawler memory says these pass by luck.

**Acceptance:** `env.action_spec()` reports the new nvec; `action_mask` is `(N, 21)`; a gadget
chosen while masked is a no-op (the phase-6 safety net); `MaskablePPO` builds against the widened
space in `test_training.py`'s smoke test; one attack per decision still holds for column value 3.

## Step G4 -- observation fields

**Files:** `brawl_sim/core/observation.py` (`build_obs` hero block), `brawl_sim/core/obs_schema.py`
(`OBS_SCHEMA`, `_HERO_DESCRIBE_FIELDS`), `scripts/dump_obs_schema.py` (run it), `docs/OBSERVATION.md`
(regenerated), `tests/test_observation.py`, `tests/test_obs_schema.py`.

```
hero.gadget_ready        (N,)  bool      gadget_cd <= 0 and the kind has a gadget
hero.gadget_charge_frac  (N,)  fraction  1 - gadget_cd / gadget_cooldown  (1.0 = ready; mirrors super_charge_frac)
```

Both go into the `self` group of the new spec (Step I1). Deploy parity: supplied by
`ShadowHero.observe()` (Step G5), proprioception only (S18).

**Acceptance:** schema round-trip test passes; both fields appear in `docs/OBSERVATION.md`.

## Step G5 -- deployment: shadow, buttons, policy

**Files:** `brawl_deployment/perception/shadow.py`, `brawl_deployment/control/buttons.py`,
`brawl_deployment/control/calibration.py` / `match_state.Calibration.button("gadget")`,
`brawl_deployment/policy.py`, `brawl_deployment/loop.py` (`Controls` build, `TickRow.attack`
docstring), `brawl_deployment/perception/assemble.py` (`_put_self` table += two fields),
`configs/deployment.yaml` (nothing new; the gadget centre is a measurement already in the JSON),
`tests/test_deployment_shadow.py`, `tests/test_deployment_control.py` (the buttons tests live here), `tests/test_deployment_policy.py`,
BRAWL_DEPLOYMENT_DESIGN.md §4.4 (append a gadget paragraph; keep the "Gadgets are not emitted"
sentence with a dated strike-through, the way §4.4 itself was revised).

- **Shadow:** `ShadowParams.gadget_cooldown`; `gadget_cd` timer; `reset()` sets it 0 (charged at the
  gate); `_tick` decrements; `attack_mask()` returns **four** legals; `act()` accepts 3 and queues a
  gadget for the next sub-tick (no `attack_cd`, no ammo); `observe()` adds `gadget_ready` and
  `gadget_charge_frac`; `resync()` leaves `gadget_cd` alone (no CV evidence about it; a tap on a fixed
  button is the most reliable input we send). `_last_attack_at` is **not** touched by a gadget: the
  ammo canary's grace period is about the ammo bar, and the gadget does not move it.
- **Buttons:** `origin(3) = cal.button("gadget")`; a gadget press is a **tap** -- down on one tick,
  up on the next, no drag (the game aims the gadget itself, which is exactly S10). `settle()` gets
  the two-step variant; `aim()` raises for action 3; the off-screen radius check does not apply.
  Contract unchanged: one press per decision, a press in flight finishes first.
- **Policy:** `_mask` is `n_move + 4`; `attack_legal` is a 4-tuple; `Decision.attack` in 0..3;
  `check_spaces` compares the widened nvec; the "no-fire must be legal" guard unchanged.
- **Loop:** nothing structural; `row.attack` now ranges 0..3.

**Acceptance:** shadow test: gadget legal at reset, illegal for 18 s after `act(…, 3)`, legal again
at 18.0; attack legality unaffected by a gadget; buttons test: action 3 emits down at the gadget
centre then up, and never a move; policy test: a 4-wide mask reaches `predict`; the calibration
script's dry run still refuses a spec/checkpoint mismatch.

## Step G6 -- the match gate anchor (measure, then decide)

**Files:** `scripts/deploy_calibrate.py` (a `--probe-gadget` job), `brawl_deployment/match_state.py`,
`brawl_deployment/data/control_calibration.json` (`match_gate`), `tests/test_deployment_capture.py` (where `MatchState` is tested).

1. **Measure:** in a real match, tap the gadget once and record the gadget anchor's ring score every
   tick for 20 s, alongside the attack and super anchors' scores. (The venue rule applies: not
   Training Grounds.) Save the trace under `runs/audit/`.
2. **Decide by the trace:**
   - Gadget score never drops under 0.45 during the cooldown: keep the anchor, document the trace.
   - It does: switch `match_gate.anchor` to whichever of `super` / `attack` held the wider margin in
     the trace (super's refit score is 0.987, attack's 0.960; super changes look when charged, attack
     is under a floating stick -- the trace, not the guess, picks), **or** implement a 2-of-3 vote
     over the three anchors in `MatchState.update` (`_above`/`_below` per anchor, in-match while at
     least two agree). The vote is the robust answer if no single button is clean.
3. The `_comment` in the JSON that says "the one button the policy never presses" is rewritten
   either way; it is now false.

**Acceptance:** a `MatchState` unit test replays the recorded trace and stays in-match throughout;
the loop's first live dry run (`control.backend: null`) after Phase G shows no gate exit at the
first gadget decision.

**Deferred 2026-09-21** (the operator): G6 runs after the training run, with every other
deployment step, and still lands before any live run of the new checkpoint. Training does not
need it: the gate reads the emulator's screen, and the sim has no gate.

---

# 5. Phase H -- three frames of local history

## Step H1 -- ring buffers and the push

**Files:** `brawl_sim/core/state.py` (`_STATE_FIELDS`), `brawl_sim/core/history.py` (new),
`brawl_sim/env.py` (`step`, `_build_observation`, `_autoreset`), `brawl_sim/config.py`
(`EnvConfig.history_frames: int = 3`, `history_radius_tiles: int = 4`), `configs/default.yaml`,
`tests/test_history.py` (new).

State (K = `history_frames`, newest at index 0; all zero at `zero_`, which is the correct "no
history" value):

```
hist_valid       (N,K)     bool
hist_action      (N,K,2)   i64    (move_bin, attack) applied in the window that ENDED at that boundary
hist_hp          (N,K)     f32
hist_ammo        (N,K)     f32
hist_pos         (N,K,2)   f32    hero world pos at that boundary
hist_enemy_pos   (N,K,E,2) f32    every entity's world pos at that boundary
hist_enemy_seen  (N,K,E)   bool   alive & revealed_to_hero at that boundary (slot 0 = hero, always false)
```

**Push timing, and why it is where it is.** `history.push(state, action, vis)` runs at the **top of
`env.step(action)`**, before `_run_decision`: it shifts every ring by one (K = 3, so two slice copies)
and writes slot 0 from the state *the policy just observed* plus the action it chose against it.
So at the next observation, frame 1 is "one decision ago I was here with this hp/ammo, I saw these
enemies there, and I chose this" -- which is what the network needs. Autoreset zeroes the rings for
the rows it resets (they are state fields), so the reset observation has `valid = [0,0,0]`, the
next `[1,0,0]`, then `[1,1,0]`, then `[1,1,1]`. The action written is the **policy's action**, which
under the mask equals the applied one; deployment writes the *modelled* action for the same reason.

`vis` is the `(N,E,E)` visibility `_build_observation` already computes; stash it on the env
(`self._obs_vis`) rather than recomputing. Outside `_run_tick`, so `torch.compile` never sees it;
sync-free (slices and `where`, no host reads).

**Acceptance:** valid-flag sequence after reset as above; `hist_action[:,0]` equals the last
`step()` argument; `hist_pos[:,0]` equals the hero position of the previous observation; rings
survive `action_repeat` changes unchanged (they are per decision); `tests/test_env_step.py`'s
timing/cost pins still hold (push cost at n_envs=4096 is measured in Step I4).

## Step H2 -- observation fields and the three grid channels

**Files:** `brawl_sim/core/observation.py` (`build_obs`, `_build_grid` 12 -> 15 channels),
`brawl_sim/core/obs_schema.py`, `brawl_sim/core/obs_select.py` (`_CHANNEL_INDEX`; confirm the
flatten path handles rank-3 fields -- `hero.pos_norm` is rank 2 today), `docs/` (regenerate),
`tests/test_observation.py`, `tests/test_obs_select.py`.

```
hist.valid          (N,K)      bool
hist.move_onehot    (N,K,17)   unitless   one-hot of hist_action[...,0]
hist.attack_onehot  (N,K,4)    unitless   one-hot of hist_action[...,1]
hist.hp             (N,K)      hp         normalized by 20000 like hero.hp
hist.ammo_frac      (N,K)      fraction
hist.displacement   (N,K,2)    tiles      hist_pos - hero.pos (0 where invalid)
```

Flattened: `3 * (1 + 17 + 4 + 1 + 1 + 2) = 78` floats -- the `history` group (Step I1). Note the
`tiles` normalizer divides by map size, so a one-tile displacement reads 0.017; that is the existing
convention for `entities.rel_pos` and the deploy-spec magnitude test is written against it. Keep it.

Grid: three new channels `enemy_hist1..3`, each a `_scatter_count` of `hist_enemy_pos[:, k]` masked
by `hist_enemy_seen[:, k] & (chebyshev(hist_enemy_pos[:, k] - hero.pos) <= history_radius_tiles)`
into the **current** `view_origin`. World positions scattered into the current window are aligned to
the hero's current position by construction -- no re-centering, no per-env roll. Outside the 9x9
block the channel is zero, which is the brief's "4-block radius".

**Acceptance:** an enemy revealed at world (10, 10) two decisions ago shows in `enemy_hist2` at the
cell for (10, 10) relative to the *current* view, and in nothing else; a hidden-then enemy shows
nowhere; an enemy 5 tiles away then shows nowhere (radius); `hist.displacement` is the negative of
the hero's motion; with `valid = 0` every history field is 0; the 12-channel grid tests updated to 15.

## Step H3 -- extractor and spec checks

**Files:** `brawl_sim/wrappers/sb3_features.py` (no code change expected; add a test),
`tests/test_sb3_features.py`.

The uint8 group grows to 13 channels in the deploy spec (10 + 3), the MLP input grows by 78. Add a
test building the extractor from `agent_obs_deploy4.yaml` (Step I1) and asserting the CNN's
`in_channels` and the MLP's `in_features` against the spec, so a spec edit that silently changes
widths fails here rather than at the first `model.learn`.

## Step H4 -- deployment mirror

**Files:** `brawl_deployment/loop.py` (`DeployLoop`: a `deque(maxlen=3)` of decision snapshots),
`brawl_deployment/perception/assemble.py` (`_put_history` + a `history` supplier; `_put_view`
passes `enemy_history`), `brawl_deployment/perception/grid.py` (`GridBuilder.build(...,
enemy_history: list[list[xy]])` scatters past map-frame enemies with the same 4-tile block),
`tests/test_deployment_assemble.py`, `tests/test_deployment_grid.py`, `tests/test_deployment_loop.py`.

Snapshot per decision, taken **after** `shadow.act` (so it holds the modelled action): `(move_bin,
modelled attack, hero hp read or last good, shadow ammo, hero map-frame pos, [enemy map-frame pos
for every tracked enemy this decision])`. Cleared at the match gate's entry. A missed HP read carries
the previous value (the sim never has an invalid frame mid-episode; a zero would be a lie). Past
enemy positions come from the tracker's map-frame output, which is what the current `enemies` group
already uses (A-H1).

**Acceptance:** assembler test: three synthetic snapshots produce the same 78-float vector the sim
produces for the same numbers (build both, compare); grid test: a past enemy at map (x, y) lands in
`enemy_hist_k` at the same cell the sim's `_scatter_count` would put it; loop test: the first three
decisions after the gate carry `valid = [0,0,0], [1,0,0], [1,1,0]`.

---

# 6. Phase M -- maps, and no fences

## Step M1 -- remove fences from the sim

**Files:** `brawl_sim/constants.py`, `brawl_sim/maps/csv/{feast_or_famine,scorched_stone,walled}.csv`,
`brawl_sim/maps/README.md`, `brawl_sim/render/ascii.py`, `brawl_sim/render/viewer.py`,
`tests/test_map_csvs.py`, `tests/test_maps.py`, `tests/test_constants.py`,
`brawl_deployment/perception/grid.py` (class table), `tests/test_deployment_grid.py`.

- `constants.py`: drop `"f"` from `CHAR_TO_TILE` (`load_map_csv` then raises on it -- pin that).
  **Keep `Tile.FENCE = 4`** with a comment: "not a sim tile since 2026-09; retained because
  `brawl_vision.terrain.labeling.CLASSES` and the trained terrain classifier index it" (S15). Keep it
  in `TILE_BLOCKS_UNIT` (a fence does block movement; the table is also what vision's scorer reads).
- CSVs: convert `f` -> `#` where the cell 8-touches an existing `#`, else `f` -> `.`. Re-run
  `validate_map` and the README's rules by eye on the ASCII render (`scorched_stone` has 3x3 fence
  blocks next to walls that become 3x3 wall blocks -- check no corridor closes). `walled` (test-only):
  `f` -> `#`.
- README: delete the legend row; add one line under design rules: "fences are not a sim tile".
- Renderers: entries keyed by `Tile.FENCE` may stay (they are dead but harmless); delete the `f`
  from `ascii.py`'s char table if it maps chars rather than tiles.
- Tests: `test_walled_has_water_and_fences` -> `test_walled_has_water`; new
  `test_no_map_contains_fences` over every CSV; `test_every_character_is_known` now covers the
  refusal.
- **Deployment:** in `grid.py`'s class table, override the FENCE row to **water's** values
  `[blocks_unit=1, blocks_projectile=0, is_bush=0, is_water=1]` with a comment: the policy will never
  have trained on `blocks_unit & ~blocks_projectile & ~is_water`, and water is the trained-on
  combination with a fence's physics. (Mapping it to WALL would tell the agent a fence stops bullets.)
- Vision: **no change**. `terrain.pt`, `CLASSES`, labels and `vision_score_map.py` keep their five
  classes; only the sim's map vocabulary shrinks.

**Acceptance:** every CSV loads; no `f` anywhere under `maps/csv`; `load_map_csv` raises on `f`;
deployment grid test for a FENCE cell returns water's row; the full `tests/test_vision_*.py -m "not
vision"` set is untouched and green (11 s).

## Step M2 -- the generator

**Files:** `scripts/gen_maps.py` (new), `brawl_sim/maps/generate.py` (new: the library the script and
the tests share), `tests/test_map_generate.py` (new).

Deterministic in `(seed, family)`. Pipeline, on a 60x60 char grid:

1. **Border**: ring of `#`. For the two water-border maps, a 2-tile `~` moat inside it with four
   3-tile floor gaps (one per side, off-centre so the moat is not a symmetric funnel).
2. **Stamps**, placed in the top half (rows 1-29) and mirrored -- 180 degrees for point symmetry
   (`(r,c) -> (59-r, 59-c)`), left-right for mirror maps (`(r,c) -> (r, 59-c)`):
   - wall cluster: rectangle 2x2 .. 4x6, optionally L/T-shaped, with a 1-tile bush skirt on 1-2
     random sides (the "wall with grass attached" motif in every screenshot);
   - pond: filled ellipse radius 2-4, 1-tile bush rim on 50% of the perimeter;
   - channel: 1-2 wide water strip, 6-14 long, straight or one bend;
   - bush patch: blob of 6-30 cells;
   - crate spot `X`: single or pair, 1 tile from cover;
   - centre feature (point-symmetric maps only): one of {2x2 wall + 4 bushes, small pond, open
     cross}.
3. **Spawns** `S`: 16, on a ring 6-8 tiles inside the border at near-uniform angles, each nudged to
   the nearest floor cell; `spawn_points` sorts by angle anyway.
4. **Repair pass**: remove any wall run longer than 8 straight tiles by punching a 2-tile gap; fill
   1-wide dead-end corridors; connect components by carving the shortest floor path between the two
   largest (rare after the run cap).
5. **Validate**: `loader.validate_map` (border, 8-32 spawns, 8-64 boxes, one unit-passable
   component), `bush_waypoints <= 63`, plus the **style bands** and **pocket check** below. Reject and
   reseed on failure; report the seed that passed.

**Style bands (interior shares), per family:**

| family | count | wall | bush | water | boxes | symmetry |
|---|---|---|---|---|---|---|
| standard | 6 | 7-12% | 20-30% | 3-8% | 20-32 | 5 point, 1 mirror |
| open | 1 | 4-6% | 12-18% | 0-3% | 16-24 | point |
| dense | 1 | 8-12% | 32-38% | 2-5% | 24-36 | mirror |
| water-border | 2 | 6-10% | 20-28% | 12-20% (incl. the moat) | 20-32 | point |

**Pocket check** (the README's "no enclosed pockets", made mechanical): from every floor cell, a BFS
bounded to radius 6 must reach >= 40 floor cells. **Bush spread**: at least one bush cell in >= 80%
of the 10x10 `hunt_cell_tiles` cells (so HUNTER's waypoint sweep has somewhere to go on every map).

**Acceptance:** `test_map_generate.py` generates each family at a fixed seed and asserts the bands,
symmetry, the pocket check, `validate_map`, and byte-for-byte determinism across two runs; the
script prints the seed and shares for each output.

## Step M3 -- the ten maps and the rotation

**Files:** `brawl_sim/maps/csv/<ten names>.csv`, `brawl_sim/maps/README.md` (table: name, family,
seed, shares), `configs/default.yaml` (`world.maps`), `tests/test_map_csvs.py` (`DIMENSIONS` -- it is
a dict the parametrization reads, so each new map is one entry).

Names, in the repo's style (`skull_creek`, `feast_or_famine`): e.g. `twin_ponds`, `stone_fort`,
`cross_creek`, `hollow_ring`, `dry_gulch`, `reed_marsh`, `broken_wall`, `quiet_lake`, `thorn_field`,
`split_river`. Pick by looking at the ASCII render of each candidate seed; the generator produces
more candidates than needed on purpose. `world.maps` becomes the six current maps plus these ten
(sixteen; `MapBank` is padded to the longest and every map is 60x60, so no cost beyond the bank's
memory). `open` stays in rotation as the deliberate outlier it already is.

**Acceptance:** all `test_map_csvs.py` cases pass for sixteen maps; `test_maps.py`'s bank tests pass
with `len(map_names) == 16`; a 1000-step smoke episode runs on each new map with `check_invariants`
on; ASCII renders of the ten are reviewed by a human against the screenshots (this is the one
acceptance step that is a judgment call -- record which seeds were rejected and why in the README).

## Step M4 -- map-overfitting eval

**Files:** `brawl_sim/training/config.py` (`EvalConfig.map_names: tuple | None`), `scripts/train.py`
(the eval callback builds its env with it), `configs/train.yaml` (`eval.holdout_maps: [two of the
ten]`), `tests/test_training.py`.

Hold two of the ten out of `world.maps` and evaluate on them only. The number that matters for the
"agent overfits to training maps" complaint is **eval win rate on held-out maps vs on training
maps**; without this eval the next run cannot say whether the ten maps helped.

**Acceptance:** the eval callback logs `eval/holdout_win_rate` alongside the existing metric; a
config with a holdout map that is also in `world.maps` is rejected.

---

# 7. Phase I -- integration and the run

## Step I1 -- `configs/agent_obs_deploy4.yaml`

**Files:** the spec, `tests/test_configs_files.py` (`DEPLOY_SPECS` += deploy4; the pickups table;
a `test_deploy4_differs_from_deploy3_by_exactly_the_history_and_gadget_additions` pin, in both
directions like every pair in that file), `docs/AGENT_OBS.md` (regenerate).

Six groups (the maximum):

| group | dtype | width | change from deploy3 |
|---|---|---|---|
| `self` | f32 | 22 | + `hero.gadget_ready`, `hero.gadget_charge_frac` |
| `enemies` | f32 | 9 x 7 | none |
| `projectiles` | f32 | 12 x 5 | none |
| `zone` | f32 | 2 | none |
| **`history`** | f32 | 78 | new: `hist.valid, hist.move_onehot, hist.attack_onehot, hist.hp, hist.ammo_frac, hist.displacement` |
| `grid` | u8 | 13 x 13 x 21 | + `enemy_hist1, enemy_hist2, enemy_hist3` |

Header comment in the file, in the style of deploy3's: why a fifth file, what changed, which
supplier provides each new field at deployment (Steps G5, H4), and that the width changes are a
train-from-scratch.

## Step I2 -- `configs/train.yaml`

- `run.agent_obs: configs/agent_obs_deploy4.yaml`.
- Curriculum tiers from Step B4; stages unchanged.
- `eval.holdout_maps` from Step M4.
- Rewards unchanged apart from R2's `attack_in_reach` (added 2026-09-21, §9) (the gadget's damage
  is paid through `damage_dealt`; `check_reward_is_observable`
  passes because deploy4 sees pickups). One optional line for the run notes: log `gadgets_used` per
  episode through `events` so the eval can say whether the policy uses the button at all.

## Step I3 -- docs, schema, graph

`python scripts/dump_obs_schema.py`; `graphify update .` (the CLAUDE.md rule; AST-only); regenerate
`docs/AGENT_OBS.md` against deploy4 as well as `agent_obs.yaml` if the script only does the latter
(extend it with `--spec`); append the dated decisions to BRAWL_DEPLOYMENT_DESIGN.md §9/§10 rather
than editing resolved items; `CHARACTER_DETAILS.md` gadget block (G1).

## Step I4 -- test suite and throughput

- `pytest tests/ -x -q -n auto` -- the full suite is ~13 min, 95% of it `brawl_sim`; the vision
  files can be run alone in 11 s and should be untouched by everything here except M1.
- `scripts/benchmark.py` at `n_envs=4096` before and after: **removed 2026-09-21** by the
  operator, who watches training's `time/fps` instead (SIM_OVERHAUL_STEPS.md I4.2 has what to
  read it against). Kept as the list of suspects if it drops: the additions are one (N,E) march
  per tick (G2), two (N,E) gathers per tick (B2/B3), the per-decision history push (H1) and
  three scatters per observation (H2). The removed budget was <= 3%. If the history push shows
  up, the (N,K,E,2) ring is the suspect -- store enemies at (N,K,E) x int16 tile coordinates instead.
- Check for a running `scripts/train.py` before any GPU job (the GPU is shared).

## Step I5 -- the run checklist

1. Phase A's audit numbers recorded (pre-change baseline, sim and deploy).
2. `control.backend: null` dry run of the deployed loop against the new spec: no gate exit at the
   first gadget decision (G6), `history` and `self` groups assembled without `_require` errors (H4,
   G5), telemetry rows carry the A2 fields.
3. Train. The first stage's win rate will start lower than the last run's (harder `hard`); the gate
   is unchanged at 0.35, so the walk self-paces.
4. Deploy `best_model.zip`; re-run the cadence audit on the resulting telemetry and compare to A1/A2.

**Reordered 2026-09-21** (the operator): 3 runs first. 2 and 4 wait until training is complete
and G6 has landed; SIM_OVERHAUL_STEPS.md I5 is the live checklist.

---

# 8. Trade-offs made explicit

| choice | what it costs | what it buys | alternative not taken |
|---|---|---|---|
| Gadget as attack-column value 3 (S7) | no dash+gadget in one 0.25 s decision | zero plumbing change to `_held`, `act_buf`, masks, deployment `Decision` | a third action column `(N,3)`: independent gadget, but every consumer of `action.shape == (N,2)` changes |
| Fire latch (S6) | shadow timing margin shrinks to ~100 ms; A4's lift-commit is mandatory | the sim can express the game's real cadence; +33% dashes/s in a burst | shorter decisions (0.20 s): perception must hit 15-20 Hz, which is measured not to fit |
| History as rings in state + raster channels (S13) | +78 floats, +3 channels, a per-decision push | no extractor change, fits in 6 groups, world-aligned by construction, deployable from data the loop already has | SB3 `VecFrameStack` on the dict: triples the host transfer and stacks everything, including 13x21 terrain, three times |
| Past enemies only as raster (S14) | sub-tile precision lost for past frames | no per-frame sort, no identity tracking needed on either side | top-K vector per frame: +30 floats and a tracker dependency at deployment |
| `hero_focus` as a distance discount (S4) | "favor" is a ratio, not a probability; tuning is by a single number | sync-free, one gather, exact today-behaviour at 0, overrides stickiness cleanly | a Bernoulli per re-pick: needs RNG in `select_target` and never overrides a sticky target |
| Fences: enum kept, vocabulary removed (S15) | a dead enum member with a comment | no vision retrain; the just-retrained `terrain.pt` stays valid | deleting `Tile.FENCE`: a 4-class retrain and relabel for a class seen in four cells |
| Generated maps (S16) | a generator is code to maintain; style is a band, not an eye | reproducible, validated, extendable to 30 maps; ten maps in an afternoon | hand-authored CSVs: better-looking, not reproducible, ~a day of typing |
| Gadget cooldown by proprioception (S18) | a dropped gadget tap is invisible for 18 s | no new CV reader; mirrors how `attack_cd` already works | a ring-score reader on the gadget button: real work, and G6 measures the very signal it would need |

---

# 9. What to revisit as the system grows

- **R1 -- action latency parity.** The sim trains at 0 ticks of input latency; the emulator's attack
  lands ~167 ms after the decision (three 12 Hz ticks) and movement ~one tick. A per-column latency
  (`attack` 3-4 ticks, `move` 0-1) in `act_buf` would close the biggest remaining timing gap. Not in
  this plan because A4's lift-commit is the cheaper half of it and must come first.
- **R2 -- if A1 says utilization is low.** Look at `death: -3` against `damage_dealt: 3e-4` (a
  death costs the same as 10 000 damage dealt), and at whether the long dash is being farmed. A
  small per-decision `attack_in_reach` shaping term is the blunt fix; changing `long_dash_seconds`
  is not on the table (game-accurate). **Taken 2026-09-21** (A1 measured 0.339; the operator
  approved the term): `reward.attack_in_reach: 0.05`, see SIM_OVERHAUL_STEPS.md Step R2.
- **R3 -- CV read of the gadget button.** The ring-score trace G6 records is exactly the signal a
  reader would use; if S18's silent-drop failure mode shows up in telemetry (a gadget modelled,
  no damage spike within 0.5 s, repeatedly), build the reader and add the gadget to `resync`.
- **R4 -- ablate the history group.** Train with `history` present but zeroed (valid = 0 always) on
  the same seed and compare held-out win rate; if it does not move, the raster channels alone may be
  the whole gain, and 78 floats can go.
- **R5 -- bots with gadgets and supers.** G1/G2 are kind-agnostic on purpose (as D2 was); a
  `gadget_*` block on a bot kind plus a `gadget_fire` decision in `combat_rules` is the whole change.
- **R6 -- history radius and depth.** 4 tiles / 3 frames are the brief's numbers; `history_frames`
  and `history_radius_tiles` are config for a reason. Six frames at 2 tiles is a different memory.
- **R7 -- map count.** The generator's families are bands; once M4's holdout eval exists, the
  question "do 30 maps beat 16" is a two-line config change and a run.

---

# 10. Order of work and dependencies

```
M1 (fences)  ─────────────────────────────────────────────────────────┐
B1 → B2 → B3 → B4 (difficulty)  ──────────────────────────────────────┤
A1 → A2 (audit; needs a checkpoint + one deploy telemetry file)        │
G1 → G2 → G3 → G4 ──┬──────────────────────────────→ G5 → G6 (deploy)  ├→ I1 → I2 → I3 → I4 → I5
                    └→ A3 (latch, sim; gated on A1/A2) → A4 (deploy)   │
H1 → H2 → H3 (needs G3's 4-wide attack one-hot) ──→ H4 (deploy)        │
M2 → M3 → M4 (maps) ──────────────────────────────────────────────────┘
```

- **Parallel tracks:** M, B, A1/A2 and G1-G2 touch disjoint files and can proceed together. H2's
  `attack_onehot` width and G3's column width are the one coupling between H and G.
- **Sim before deployment within each phase:** every deployment step (A4, G5, G6, H4) mirrors a
  sim step and is tested against it; do not start one until its sim half's tests are green.
- **Smallest useful first day:** M1, B1-B2, G1, A1 -- four independent, well-pinned changes that
  make everything after them concrete.
- The estimates below are for a model working from this document with the tree open; they exclude
  the training run itself.

| phase | steps | size |
|---|---|---|
| A | A1, A2 (audit) / A3, A4 (latch) | S / M |
| B | B1-B4 | M |
| G | G1-G6 | L (G2 and G5 carry most of it) |
| H | H1-H4 | M-L |
| M | M1-M4 | M (M2 is the bulk) |
| I | I1-I5 | S, plus the run |
