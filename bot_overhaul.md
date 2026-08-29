# Bot & Hero Overhaul — Build Plan

Companion to [BRAWL_SIM_BUILD_PLAN.md](BRAWL_SIM_BUILD_PLAN.md), written in the same shape (numbered
steps, each independently buildable and testable, each with its own files/acceptance/implementation
notes). The source of truth for *what* each character should do is
[CHARACTER_DETAILS.md](CHARACTER_DETAILS.md); this document is *how* the simulator gets there.

**Three goals, in priority order:**
1. The hero fights bots that behave like the real game's brawlers (Brock, Grom, Buzz, Shelly).
2. The result generalizes — adding a sixth brawler should be a YAML row, not a new Python module.
3. Bot cost stays bounded. Training throughput must not regress.

---

# 0. Decisions — all settled (2026-08-18)

| # | Decision | Resolution |
|---|---|---|
| **D1** | How the hero's Super enters the action space | **Widen the attack dim `2 → 3`** (`0=none, 1=attack, 2=super`). Keeps `action.shape == (N,2)`, every wrapper, `act_buf`, `_held`, and the MaskablePPO mask layout intact. |
| **D2** | Do bots get Supers? | **Hero only, for now.** Build the charge/super machinery kind-agnostically (a bot super is then a config block), but configure it only for `hero_mortis`. |
| **D3** | Brock's sphere tick schedule | **t = 2s and t = 4s.** Falls out of Step C3's stateless `prj_age` schedule for free. |
| **D4** | Does the sphere stack / hurt its owner? | **Stacks; never hurts its owner.** Two overlapping spheres deal 1392/tick. |
| **D5** | Does the sphere respect walls? | **No.** It is a ground area — it damages anything inside its radius, LOS irrelevant. |
| **D7** | Attack animations | **There are no animations.** `attack_cooldown` simply does double duty: for its duration you can neither attack **nor reload**. Minimum **0.25s** for everyone; longer for Mortis (0.35s — he has to dash) and Buzz (1.0s — he fires five shots). See §1.2, this removes an entire subsystem from the plan. |
| **D8** | How hazards are represented | **One projectile buffer, three motion/damage CLASSES** — `PROJECTILE` (bullet: travels, dies on contact), `ARTILLERY` (travels, deals no damage in flight, resolves at its landing point), `HAZARD` (doesn't move, doesn't disappear on contact, damages anything standing in it). Replaces the `prj_lobbed` bool; **no new buffer, no new obs group**. |
| **D9** | Buzz's per-sub-swing arc | **0.65 rad** — the value that makes "at most 3 of 5 connect" true (arithmetic in §3). |
| **D10** | Shelly's spread | **30° radius = 60° total** → `proj_spread_rad: 1.047`. See §3's note — `proj_spread_rad` is a total span in this codebase, so the half-angle is doubled to get it. |
| **D11** | Checkpoint compatibility | **Irrelevant** — no existing trained models will be reused. Widen `Proj` and the obs freely. |
| **D12** | Regen | **The official rule: 13% of max HP per second, starting 3s after last attacking or taking damage.** `regen.max_hp_fraction_per_second: 0.2 → 0.13`, delay unchanged. Ground truth, not a tuning knob — see §1.4. |
| **D13** | Does the agent see a projectile's class? | **Yes** — the agent must be able to tell lobbed / hazard / bullet apart. `projectiles.class_onehot` (3-wide) goes into `agent_obs.yaml`'s projectiles group. |
| **D15** | Zone lethality | **Nobody survives more than ~5 seconds in the zone.** That is a *proportional* requirement — it must hold for a 6000 HP Brock and a 34000 HP fully-cubed Buzz alike — so zone damage becomes **20% of max HP per second**, replacing the flat `zone.dps`. See Step B3. |
| **D16** | Regen | **Closed, no change beyond D12's 0.13.** The regen-vs-DPS concern §1.4 raised is not a problem: the heal is fast once it starts, but starting it costs a long window with *no attacking and no damage taken*, and any attack or hit resets it. That is already exactly how `ent_out_of_combat_t` behaves (reset by `combat.apply_damage` **and** by `env._attack_phase`). The disengagement dynamic is intended; **the zone is what punishes it** (D15). |

| **D14** | Projectile budget | **May be raised if measurement justifies it, within reason.** Plan: keep `max_projectiles: 128`, measure the real steady-state peak after Step C3 (hazards occupy slots for 4s), raise only to what that shows plus headroom. Measured cost is ~0.12 ms/step per slot at `n_envs=1024`, so 128→192 is ~+8 ms/step (+4%) and 128→256 is ~+16 ms (+8.6%, the whole Phase 0 saving). **Raising `P` does not change the agent's input width** — `agent_obs.yaml`'s projectiles group is `max_slots: 12` nearest-K — so this is purely a sim-time tradeoff. |

**Assumptions I am proceeding on** (called out because guessing wrong means rework, not because they
block Step A0):

| # | Assumption | Why | If wrong |
|---|---|---|---|
| A1 | **Buzz's 5-hitscan sweep costs 1 ammo, not 5.** | CHARACTER_DETAILS says "time to reload one **bullet**: ~1 second", and one attack press = one ammo is how the game works. | 5× change to his ammo economy; §1.4's 1260 DPS becomes ~250 sustained. Would make him a non-threat. |
| A2 | **Brock's sphere spawns wherever the rocket dies** — wall, unit, box, *or* max-range expiry — not only on a unit hit. | Matches the real rocket (it detonates on anything) and is the more general rule. A 5.33 tiles/s rocket at 8 tiles range expires often, so this materially changes hazard density. | Fewer hazards, and Brock loses most of his area denial. |
| A3 | **Grom keeps `attack_range: 7.33`.** | CHARACTER_DETAILS doesn't specify a range for him; every other Grom stat is given. | One-line config change. |
| A4 | **The Super does not charge itself** — super hits count toward the *next* charge normally. | No reason to special-case it. | Marginal. |

---

# 1. Observations on the current codebase

## 1.1 What the current design assumes

The simulator is well-factored, and the factoring is along one specific axis: **`Kind` is both "which
brawler" and "which weapon archetype", and each archetype is a Python module.**

- [`brawl_sim/constants.py:19`](brawl_sim/constants.py#L19) — `Kind` has exactly 5 members, and
  `N_KINDS = 5` sizes every per-kind `SimParams` tensor's `K` axis and every `kind_onehot`.
- [`brawl_sim/bots/policy.py:359`](brawl_sim/bots/policy.py#L359) — `_archetype_modules()` hardcodes
  a `(Kind, module)` pair per archetype; `all_bot_intents` runs all four `combat()` functions over
  all `(N,E)` entities and selects by kind.
- Each archetype module is **already fire/aim only** (the Step 41 combat/movement split). Movement is
  personality-driven and fully shared. This is the single best thing about the current design for our
  purposes: **the overhaul only has to touch the combat half.**

The consequence: the four archetype modules
([`sniper.py`](brawl_sim/bots/sniper.py), [`artillery.py`](brawl_sim/bots/artillery.py),
[`melee.py`](brawl_sim/bots/melee.py), [`rifle.py`](brawl_sim/bots/rifle.py)) are each ~40 lines and
differ only in **which predicates they AND onto `shared.fire_gate`** and **which aim model they
use**. That is a table, not four modules — which is exactly what makes goal (2) achievable cheaply.

## 1.2 The five gaps, named

Every stat in CHARACTER_DETAILS.md is either a `configs/brawlers.yaml` edit or one of these five
structural gaps. Nothing else is required.

**Gap 1 — reload is unconditional.** [`hero.tick_timers`](brawl_sim/core/hero.py#L63) does
`ammo += dt/reload_seconds` every tick with no gate at all (its own docstring even says "reload
continues during a dash"). Under **D7**, firing must pause ammo accrual for `attack_cooldown` seconds.

**D7 deleted what used to be the largest gap here.** An earlier draft of this plan proposed an
`attack_anim_seconds` param and an `ent_attack_anim_t` state field to model "I am mid-attack". That
subsystem is gone: `ent_attack_cd` **already is** that timer. It already exists, is already
decremented in `tick_timers`, and already gates re-attacking in three places (`_attack_phase`'s
`can_attack`, `hero.action_mask`'s `fire_ok`, `policy.fire_gate`). D7 adds exactly one term to one
line — the reload gate — and everything else follows. Concretely:

| what CHARACTER_DETAILS asks for | how it's expressed |
|---|---|
| can't attack again for a while | `attack_cooldown` — **already works** |
| reload pauses for that window | one `& (ent_attack_cd <= 0)` in `tick_timers` |
| "+0.1s after", short frames for most | absorbed into the 0.25s floor |
| Buzz's 5 shots over 1.0s | `attack_cooldown: 1.0`, sub-swings clocked off `ent_attack_cd` |
| Mortis has to dash | `attack_cooldown: 0.35` (> `dash_duration` 0.30) |
| Grom's 1.25s shell flight | **not a cooldown** — `proj_flight_seconds`, already implemented. He can throw again while one is airborne. |
| Brock's 1.5s travel | **not a cooldown** — `proj_speed`, already implemented |

Net: **zero new state fields and zero new per-kind params** for the whole "attacks take time" concern
— including Buzz's sweep clock, which Step C2 reads straight off `ent_attack_cd`.

**Gap 2 — nothing persists on the ground.** Damage comes from exactly four sources (dash capsule,
melee cone, projectile, zone), all of which resolve on the tick they occur. Brock's 0.75-radius
sphere ticking twice over 4s has no home.

Per **D8**, the fix is to notice that the projectile pipeline is *already* a two-class system and to
make the class explicit. `prj_lobbed` today means exactly "travels, but does not collide with units,
walls, or boxes in flight" — `non_lobbed = ~state.prj_lobbed` gates every collision branch in
[`step_projectiles`](brawl_sim/core/projectiles.py#L325). Replacing that bool with a three-value
`prj_class` gives:

| class | moves | damages in flight | dies on contact | resolves by |
|---|---|---|---|---|
| `PROJECTILE` | ✅ | ✅ | ✅ | first unit/wall/box hit, or range expiry |
| `ARTILLERY` | ✅ | ❌ | ❌ | arriving at `prj_target` → AoE + splits |
| `HAZARD` | ❌ | ✅ (on a schedule) | ❌ | tick count exhausted |

`HAZARD` reuses the AoE damage test the detonation path already has
([`projectiles.py:396`](brawl_sim/core/projectiles.py#L396) — `ent_dist <= prj_aoe`), so the new code
is a gate and a schedule, not a new damage system. **No new buffer, no new obs group, no new tick
phase.** This is strictly less machinery than the separate `haz_*` buffer originally proposed, and it
generalizes better — any future weapon that leaves a puddle, mine, or turret is a class value plus
config.

**Gap 3 — a swing is one instantaneous cone.** [`melee_hitscan`](brawl_sim/core/combat.py#L70) reads
`state.ent_facing` and resolves one cone on the tick it fires. Buzz needs five, at rotating angles,
spread across his 1.0s cooldown.

**Gap 4 — a projectile hits once and dies.** [`step_projectiles`](brawl_sim/core/projectiles.py#L316)
kills a projectile on its first unit hit (`_earliest_only` → `unit_kill`) and on any wall
(`wall_kill`). Mortis's Super must pass through **both**, and must not hit the same victim twice —
which needs per-projectile hit memory, the same thing `ent_dash_hits` already is for dashes.

**Gap 5 — the hero has no ability state.** No charge counter, no super, no "have I been idle long
enough" flag. `hero.action_mask` returns a fixed `{"move", "attack"}` pair and `decode_action` reads
exactly one fire bit.

Two smaller ones worth naming:

- **`MAX_PROJ_PER_ENTITY = 3`** ([`projectiles.py:26`](brawl_sim/core/projectiles.py#L26)) is a hard
  Python constant. Shelly fires 5. See §4 Step B2 — this one has a trap in it.
- **`ent_out_of_combat_t` is the wrong stopwatch for the long dash.** It is reset by taking damage
  *and* by attacking ([`env.py:554`](brawl_sim/env.py#L554)); Mortis's long dash charges off "time
  since I last **attacked**" only. A Mortis being shot at while not attacking should still charge it.

## 1.3 Measured performance baseline

Measured on this machine (RTX 5070 Ti / sm_120), `configs/default.yaml`, `n_envs=1024`,
`action_repeat=5`, so every number below is **per `step()`, i.e. 5 sim ticks**:

```
native throughput           4,757 steps/sec      215.3 ms/step

_bot_phase          46.35 ms  (24.9%)     _dash_phase          5.60 ms  ( 3.0%)
_attack_phase       35.29 ms  (18.9%)     _death_phase         3.46 ms  ( 1.9%)
_movement_phase     28.67 ms  (15.4%)     _pickup_phase        2.95 ms  ( 1.6%)
_projectile_phase   23.36 ms  (12.5%)     _box_phase           2.12 ms  ( 1.1%)
_autoreset          19.57 ms  (10.5%)     ...
_observe            13.52 ms  ( 7.2%)     sum of 17 phases   186.46 ms
```

> **⚠ Measurement caveat, learned the hard way during Phase 0.** These per-phase numbers are only
> comparable **within one process**. Across processes this box varies by ±20% or worse: repeated
> identical runs produced 3,105–5,731 steps/sec, and after Phase 0 landed, phases nobody had touched
> (`_movement_phase`, `_projectile_phase`) appeared to *double*. Root cause found with `nvidia-smi`:
> **the GPU sits at ~69% utilization with ~25 compute processes attached even when no benchmark is
> running**, so absolute timings depend entirely on what else the machine is doing.
>
> That contention was later identified as an unrelated foreground workload and cleared. Two further
> confounds showed up even on an **idle** GPU, and both are worth knowing before trusting any number
> here:
> - **The env must be reset to a fixed seed before every measurement.** Timing successive `step()`
>   batches without resetting times different *points in the episode* (entities dead, zone shrunk),
>   which swamped the effect and produced an impossible ordering (an optimization measuring
>   *slower* than its own absence).
> - **There is strong monotonic drift across a long benchmark** — within one mode, successive
>   samples ran 179 → 271 ms. Use **min of N**, not mean or median; the minimum is the least
>   contaminated sample and it orders the four A1/A2 on-off combinations correctly.
>
> - **MEMORY must be measured process-scoped, never with `nvidia-smi`.** Added in Step E2 after
>   the same trap caught a third time: `nvidia-smi --query-gpu=memory.used` reports the WHOLE
>   DEVICE, so a training peak sampled that way included an unrelated foreground application and
>   read **14.6 GB / 92%** when the training process itself peaked at **8.78 GB / 55%**. That
>   produced a confident and completely wrong "this will OOM if anything else touches the GPU"
>   warning. `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` count only the calling
>   process and are immune; they also catch the peak INSIDE a backward pass, which any sampling
>   loop will miss. The same applies on the host side: `GlobalMemoryStatusEx`/free-RAM deltas are
>   system-wide, while summing `arr.nbytes` over the real arrays is exact and process-scoped.
>
> Every Phase 0 result below is therefore: single process, both arms, fixed-seed reset per
> measurement, min of 5, idle GPU. Measured that way the parts and the whole agree — A1 (~11.5 ms)
> + A2 (~4.3 ms) predicts 15.8 ms/step, and the end-to-end A/B measured **194.4 → 179.0 = 15.5 ms
> (8.0%)**. Step E2's re-baseline must follow the same protocol or it is not comparable.

**Where it stands after Phase 0–2's C2** (idle GPU, min of 4, fixed-seed reset per measurement):
**229 ms/step**, up from ~179 ms at the end of Phase 0. That rise is *earned*, not regression:

| | ms/step | vs. §1.3 baseline | why |
|---|---|---|---|
| `_bot_phase` | 56.3 | −− | A2's target-LOS still in effect |
| `_movement_phase` | 46.9 | +18 | unchanged code; more entities alive longer at the new TTK |
| `_projectile_phase` | 37.2 | +14 | B2's 128→192 buffer, and B1's slower projectiles living ~2× longer |
| `_attack_phase` | 33.2 | −2 | A1's ray budget, and C2's sweep folded into the existing call |
| `_observe` + `_autoreset` | 40.8 | −− | A3 not active here (default config keeps the world grid) |

The projectile and movement phases grew because the *simulation now does more*: Brock's rocket
travels 1.5 s instead of 0.57 s, Shelly fires 5 pellets instead of 3, and fights last longer. That
is the cost of the real game, and it is what Phase 0's headroom was harvested to pay for.

Micro-benchmarks of the individual hot tensors, **per tick** (`E=10`, `ray_steps=48`):

```
melee_hitscan, full 48-step rays        4.150 ms      <- dominates _attack_phase (7.7 ms/tick)
melee_hitscan, 6-step rays              1.951 ms
raw_los (N,E,E)                         1.391 ms      <- bot phase
target-only LOS (N,E)                   0.404 ms
bush_scan                               1.465 ms
hunt_waypoint                           1.273 ms
visibility (N,E,E)                      0.578 ms
```

**Three findings, all of which are free wins** (details and step numbers in §4 Phase 0):

1. **The bot phase computes a full `(N,E,E)` LOS matrix and uses one column of it.**
   [`all_bot_intents`](brawl_sim/bots/policy.py#L409) calls `perception.raw_los` (all pairs), and the
   only consumer is [`targeting`](brawl_sim/bots/policy.py#L205)'s
   `los_enemy = gather(los, 2, idx)` — LOS to each entity's *own* target. Nothing else in the bot
   phase reads it. Measured: **1.391 → 0.404 ms/tick**.

2. **`march()` always takes `cfg.ray_steps` samples regardless of the query distance.**
   `ray_steps = ceil(max_ray_tiles / los_step_tiles) = 48`. It computes all 48 sample points, then
   masks the ones past `max_dist` via `in_range` ([`terrain.py:114`](brawl_sim/core/terrain.py#L114)).
   `melee_hitscan`'s cone is **3 tiles** — 6 steps' worth — and it pays for 48, every tick, for every
   `(N,E,E)` entity pair *and* every `(N,E,B)` entity-box pair, whether or not anyone is swinging.
   Measured: **4.150 → 1.951 ms/tick**.

3. **`obs["world"]` is built twice per step and the agent never reads it.**
   [`obs_select`](brawl_sim/core/obs_select.py#L326) only ever indexes `full_obs["view"]`;
   `configs/agent_obs.yaml`'s grid group is `view_channels`, egocentric. The full-map 60×60×12 grid is
   built in `_observe` and again in `_autoreset`. Turning it off moves `_observe` 13.52 → 9.69 ms and
   `_autoreset` 19.57 → 16.95 ms — **~6.5 ms/step (3.5%)**. Honest caveat: end-to-end throughput at
   this scale is ±20% run-to-run, so this shows up in the phase attribution but **not** above noise in
   the steps/sec number. It is still 44 MB of pointless traffic per build at `n_envs=1024`.

Findings 1+2 are worth **~16 ms/step (8.6%)**, which — see Step B2 — is almost exactly what the
projectile-buffer growth would otherwise cost. Harvest first, then spend.

## 1.4 Time-to-kill, and why regen is no longer the lever

**D7 changes the arithmetic**, so this is computed post-D7. Today sustained fire is one shot per
`reload_seconds` (the cooldown overlaps the reload). Once firing *pauses* the reload, it is one shot
per `attack_cooldown + reload_seconds`:

| | now | after (D7-inclusive) | change |
|---|---|---|---|
| sniper → **Brock** | 4800/1.8 = **2667** | (2320 + 1392 sphere)/(0.25+2.1) = **1579** | −41% |
| artillery → **Grom** | 3200/1.4 = **2286** | 2080/(0.25+2.3) = **816** | −64% |
| melee → **Buzz** | 1500/0.8 = **1875** | (≤3 × 840)/(1.0+1.0) = **≤1260** | −33% |
| rifle → **Shelly** | (3×1280)/1.7 = **2259** | (5×600)/(0.25+1.75) = **1500** | −34% |
| **Mortis** | 2000/2.4 = **833** | 2000/(0.35+2.25) = **769** | −8% |

Note Buzz: his 1.0s cooldown is by far the largest, so D7 costs him half his sustained output and he
flips from *up* 34% to *down* 33%. That is the correct consequence of a five-shot sweep being a real
commitment, but it is the number most likely to need revisiting — flagged for Step E2.

Time-to-kill roughly **doubles**: Mortis needs 8–13s of sustained contact to drop a 6000–10000 HP
bot, or 3–5 landed dashes. **That is accurate to the real game** — Mortis is a slow-burn assassin who
wins through repeated engagement, not burst — and it is the direct consequence of using real HP, real
damage, and real reload together. It is not a bug to tune out.

Burst is unaffected and remains the hero's real weapon: 3 ammo at 0.35s apart is **6000 damage in
~0.7s**, enough to delete a 6000 HP Brock or Grom from full in one engagement window.

**D12 settles regen at the real rule (13% max HP/s after 3s), and D16 closes the question.** Every
bot's sustained DPS now sits at or below the hero's own 1040 HP/s regen rate:

| | sustained DPS (post-B1) | vs. hero regen 1040 HP/s |
|---|---|---|
| Brock | 1105 | just above |
| Shelly | 1029 (1500 once B2 gives her 5 pellets) | at |
| Grom | 904 | below |
| Mortis | 889 | below |
| Buzz | 840 | below |

**This is intended, not a bug.** The heal is fast once it starts, but *starting* it costs a full
window with no attacking and no damage taken, and any attack or hit resets it -- which is already
exactly how `ent_out_of_combat_t` works (`combat.apply_damage` resets it, and so does
`env._attack_phase`). Sustained DPS is therefore the wrong comparison: nobody trades at sustained
rates, because trading resets the clock for both parties. Burst decides fights -- Mortis's 3-dash
opener is 6000 damage in ~0.7s, enough to delete a full-health Brock or Grom.

**What that leaves is disengagement, and the zone is the designed answer (D15).** A bot that fully
breaks contact does out-heal any single attacker; the zone is what makes breaking contact cost more
than staying put. Hence Step B3: zone damage becomes **20% of max HP per second**, so nothing
survives more than ~5s in it regardless of HP or cube count.

Two smaller items still land in E2, from a measured rollout rather than a guess:
1. **Episode length.** 3000 ticks = 150s. With TTK roughly doubled, confirm a 10-player lobby still
   resolves rather than truncating.
2. **`cubes.damage_bonus_per_cube`** -- at `max_cubes: 16` and 0.10/cube that is a 2.6x damage
   multiplier, and it matters more now that base damage is lower. It is the intended late-episode
   stalemate-breaker.

---

# 2. Design principles

**P1 — Mechanics are data, not modules.** Every new behavior gets a per-kind `SimParams` field and a
generic implementation. A brawler that does not use a mechanic sets its field to 0 and pays nothing
but a masked `torch.where`. This is the same discipline `split_distance`/`split_damage_fraction`
already follow (every non-artillery kind resolves them to 0 and `_spawn_splits` becomes a no-op write).

**P2 — Nothing new goes in the bot decision phase.** Every new mechanic resolves in phases 6/9
(attack, projectile), which are already dense over slots. `policy.fire_gate` already reads
`ent_attack_cd`, so after D7 the bot phase gains **literally nothing**. `_bot_phase` is 24.9% of the
step and must not grow.

**P3 — Prefer widening an existing structure over adding a new one.** D7 and D8 are both applications
of this: the attack timer already existed (`ent_attack_cd`), and the projectile class already existed
as a bool (`prj_lobbed`). The only genuinely new buffer in the whole plan is `prj_hits` `(N,P,E)`
bool, which is the same shape and pattern as the existing `ent_dash_hits`.

**P4 — Checkpoints are expendable (D11), so obs churn is not a constraint.** Widen freely; still
batch related widenings into one step to keep each step's acceptance criteria clean.

**P5 — Each step is independently revertible and independently testable.** Phase 0 changes no
behavior at all (bit-identical outputs). Phase 1 changes only numbers. Phases 2–3 add one mechanic
each, gated so that setting its config field to 0 restores the previous behavior exactly.

**P6 — Tests assert against `params`, never against literals.** The existing suite mostly does this
already (`attack_range = params.attack_range[0, Kind.BOT_MELEE].item()`), which is why a stat change
does not cascade. Preserve it — the exceptions are exactly where the drift is (§4 Step A0).

---

# 3. Per-character change table

**Pure `configs/brawlers.yaml` edits** (no code):

| kind | field | from | to | note |
|---|---|---|---|---|
| `bot_sniper` (Brock) | `base_hp` | 5000 | **6000** | |
| | `base_damage` | 4800 | **2320** | |
| | `reload_seconds` | 1.8 | **2.1** | |
| | `attack_range` | 8.67 | **8.0** | |
| | `proj_speed` | 14.0 | **5.33** | 8 tiles ÷ 1.5 s |
| | `attack_cooldown` | 0.50 | **0.25** | D7 floor; the 1.5s travel is `proj_speed`, not a cooldown |
| `bot_artillery` (Grom) | `base_hp` | 6000 | **6000** | unchanged |
| | `base_damage` | 3200 | **2080** | |
| | `split_damage_fraction` | 0.5 | **1.0** | splits now deal full damage |
| | `proj_flight_seconds` | 0.9 | **1.25** | |
| | `reload_seconds` | 1.4 | **2.3** | |
| | `attack_cooldown` | 0.45 | **0.25** | D7 floor; he can throw again while one shell is airborne |
| `bot_melee` (Buzz) | `base_hp` | 10000 | **10000** | unchanged |
| | `base_damage` | 1500 | **840** | per sub-swing |
| | `attack_range` | 3.00 | **2.67** | "same as Mortis" = `dash_distance` |
| | `reload_seconds` | 0.8 | **1.0** | |
| | `attack_cooldown` | 0.30 | **1.00** | D7 — the sweep *is* the cooldown |
| `bot_rifle` (Shelly) | `base_hp` | 7600 | **7800** | |
| | `base_damage` | 1280 | **600** | per pellet |
| | `proj_count` | 3 | **5** | ⚠ see Step B2 |
| | `attack_range` | 8.67 | **8.0** | |
| | `proj_speed` | 9.0 | **6.4** | 8 tiles ÷ 1.25 s |
| | `proj_spread_rad` | 0.18 | **1.047** | D10 — see note below |
| | `reload_seconds` | 1.7 | **1.75** | |
| | `attack_cooldown` | 0.40 | **0.25** | D7 floor |
| `hero_mortis` | `reload_seconds` | 2.4 | **2.25** | |
| | `attack_cooldown` | 0.30 | **0.35** | D7 — must exceed `dash_duration` 0.30 |

**D10 note — `proj_spread_rad` is a TOTAL span, not a half-angle.** `spawn_volley` computes
`fan_offset = proj_spread * (k - (count-1)/2) / (count-1)`
([projectiles.py:186](brawl_sim/core/projectiles.py#L186)), so for `k = 0..4` the offsets run from
`-spread/2` to `+spread/2` — the field is the full cone width. A **30° radius** is therefore
`proj_spread_rad: 1.047` (60° total). At 8 tiles that puts the outermost pellets ~8 tiles apart, so
Shelly is lethal point-blank and negligible at range — which is what a shotgun should be, but flag it
if you meant 30° total (`0.524`) instead.

**New per-kind fields** (each defaults to 0 = "this brawler doesn't have it"). Note there is **no
`attack_anim_seconds`** — D7 removed it:

| field | kind | value | mechanic |
|---|---|---|---|
| `hitscan_count` | Buzz | 5 | Gap 3 — swept melee. 1 (or 0) = today's single cone. |
| `hitscan_sweep_rad` | Buzz | 1.222 | 70° |
| `attack_arc_rad` | Buzz | **0.65** | D9 — derivation below |
| `split_seconds` | Grom | 0.5 | replaces deriving shard flight from `split_distance` |
| `on_hit_area_radius` | Brock | 0.75 | Gap 2 — spawns a `HAZARD`-class projectile on impact |
| `on_hit_area_damage` | Brock | 696 | per tick |
| `on_hit_area_ticks` | Brock | 2 | |
| `on_hit_area_interval` | Brock | 2.0 | D3 → ticks at t=2s and t=4s |
| `long_dash_seconds` | Mortis | 4.5 | Gap 5 |
| `long_dash_multiplier` | Mortis | 2.0 | |
| `super_charge_hits` | Mortis | 5 | Gap 5 |
| `super_range` / `super_damage` / `super_heal` | Mortis | 10.0 / 1800 / 1800 | |

**Global** (`configs/default.yaml`): `regen.max_hp_fraction_per_second: 0.2 → 0.13` (D12).

**D9 derivation — Buzz's per-sub-swing arc.** 5 hitscans spanning 70° = 1.222 rad gives 4 gaps, so
spacing = 1.222/4 = **0.3054 rad** (17.5°). A stationary target at a fixed bearing is caught by every
sub-swing whose center lies within the half-angle `arc/2`, i.e. by `floor(arc / spacing) + 1` of them.
For a maximum of 3, we need `arc / spacing ≥ 2` → `arc ≥ 0.611`. **0.65 rad** (37°) gives
`0.65 / 0.3054 = 2.13` → at most 3 connect, with consecutive cones overlapping — exactly
CHARACTER_DETAILS' "should overlap slightly, but it is unrealistic players are hit by more than 3."
Max damage per full sweep: **2520**. (Approximate in practice: the target moves and Buzz turns during
the 1s.)

---

# 4. The steps

## PHASE 0 — Audit tests, then harvest headroom (no behavior change)

Do these first. A1–A3 are provably output-identical, so any test failure is a real bug in the change —
which only works if the suite is trustworthy going in, hence A0.

### Step A0 — Audit and re-baseline the test suite

**Files:** `tests/*.py`

The suite is **~10.5 minutes** for a full run, so treat it as one background job, not a loop.

**The good news, established by audit:** the tests overwhelmingly assert against `params`, not against
literals — e.g. `attack_range = params.attack_range[0, int(Kind.BOT_MELEE)].item()`
([test_melee.py:102](tests/test_melee.py#L102)),
`desired = policy.RANGE_FRACTION_BY_KIND[...] * attack_range`
([test_artillery.py:130](tests/test_artillery.py#L130)). Restatting a brawler does **not** cascade
through them. The drift is concentrated in a small number of places, all found:

| location | problem | action |
|---|---|---|
| [test_action_repeat.py:236](tests/test_action_repeat.py#L236) | `assert 0.30 <= cooldown <= 0.50` for **every** brawler | **Breaks under D7** (Buzz 1.0, floor 0.25). Rewrite to `cooldown >= 0.25`. |
| [test_action_repeat.py:241](tests/test_action_repeat.py#L241) | `assert cooldown < reload_seconds`, rationale *"the cooldown must not cap sustained fire"* | **Rationale inverted by D7** — the cooldown now caps sustained fire deliberately. Also numerically false for Buzz (1.0 vs 1.0). **Delete and replace** with a test that firing pauses ammo accrual for exactly `attack_cooldown`. |
| [test_action_repeat.py:238](tests/test_action_repeat.py#L238) | `assert cooldown > cfg.agent_dt` (strict) | D7's floor is *exactly* `agent_dt` (0.25). Relax to `>=`; `_held` already guarantees one attack per decision regardless. |
| [test_melee.py:58](tests/test_melee.py#L58) | comment says *"within attack_range (2.0)"* — the real value is **3.0** | Already stale today. Fix the comment; the assertion is fine. |
| [test_action_repeat.py:195](tests/test_action_repeat.py#L195) | docstring asserts *"every attack_cooldown is now 0.30-0.50s"* | Stale after D7. Test itself still valid (it forces 0.05 locally). |
| [test_projectiles.py:122,209](tests/test_projectiles.py#L122) | comments pin `proj_count=1` / `proj_count=3` | Update for Shelly's 5. |
| [test_hero.py:130](tests/test_hero.py#L130) | zero-to-full ammo regen | **Survives** (never attacks, so `attack_cd` stays 0). Add a sibling that *does* attack. |
| `test_constants.py`, `test_obs_schema.py` | `N_PROJ_KINDS == 4`, schema widths | Mechanical updates in Step B1/C3. |

**Do in this step:** fix the stale comments, and convert any remaining literal-valued assertion to
read from `params` (P6). **Do not** yet change the assertions whose *values* Steps B1/C1 will change —
those move with their own step, so a red test always points at the step that broke it.

**Acceptance:** full suite green before any behavior change. A written list of which test files each
later step is expected to touch (the §5 matrix's `tests/` rows).

> **✅ DONE.** Baseline: `2 failed, 693 passed` in 11m39s. Both failures were **pre-existing** and in
> `test_training.py`, not in the files this audit predicted:
> - `test_shipped_curriculum_advances_at_50_percent_win_rate` asserted `advance_win_rate == 0.5` for
>   every stage; the shipped config has used 0.2/0.3 for some time. **Already failing before any
>   uncommitted edit.** Rewritten to assert the invariants that survive retuning — non-terminal
>   stages are escapable (`0 < rate < 1`), thresholds are non-decreasing, terminal is `None` — since
>   the exact threshold is a tuning knob the overhaul will move again.
> - `test_shipped_curriculum_walks_easy_to_hard` asserted the terminal stage contains no `medium`;
>   the config's terminal stage is now a deliberate mixture (`{medium: 0.3, hard: 0.5, elite: 0.2}`),
>   which is an anti-overfitting choice. Rewritten to assert the walk's real shape: `easy` share
>   non-increasing **and** combined `hard`+`elite` share non-decreasing.
>
> The two predicted stale comments were both fixed by **derivation, not recommenting**
> ([test_melee.py](tests/test_melee.py), [test_combat.py](tests/test_combat.py)): each now asserts
> its range precondition off `params`, so a restat fails loudly instead of silently making the test
> pass for the wrong reason. Verified: `98 passed`.

### Step A1 — Per-call ray budget on `march`

**Files:** `brawl_sim/core/terrain.py`, `brawl_sim/core/combat.py`, `tests/test_terrain.py`,
`tests/test_combat.py`

`march()` takes `steps = cfg.ray_steps` unconditionally. Give it an optional `max_tiles` argument;
when passed, use `ceil(max_tiles / cfg.los_step_tiles)` samples instead. Thread it through
`line_of_sight(..., max_tiles=None)`. Call sites that know their own bound pass it:

- `combat.melee_hitscan` — both the `(N,E,E)` and the `(N,E,B)` LOS calls are bounded by
  `attack_range` (≤ 3 tiles today, 2.67 after Step B1) → 6 steps instead of 48.
- `bots/policy.targeting`'s box LOS — bounded by `attack_range` (≤ 8.67) → 18 steps.
- `perception.raw_los` for bot use — bounded by `cfg.bots_sight_tiles` (14) → 28 steps.
- `env._build_observation`'s `raw_los` stays at the full budget: it feeds `obs["visibility"]["los"]`,
  which is documented as a whole-map physical answer.

**Correctness note:** the "one extra sample exactly at `max_dist`" behavior
([`terrain.py:100`](brawl_sim/core/terrain.py#L100)) must be preserved — it is what makes short rays
work at all, and short rays are the entire point of this step.

**Acceptance:** for every existing call site, `march(..., max_tiles=d)` and `march(...)` return
identical `(hit, hit_pos, hit_t)` whenever `max_dist ≤ max_tiles` elementwise. `melee_hitscan` drops
from 4.150 → ~1.95 ms/tick at `n_envs=1024`. Full suite green.

> **✅ DONE — 4.273 → 1.970 ms/tick, a 2.17× speedup** (paired, min of 5, idle GPU). At
> `action_repeat=5` that is **~11.5 ms/step**.
>
> *An earlier run of this measurement reported 2.9×. That was taken while the GPU was ~69% busy
> with an unrelated workload, which inflated the slow arm. 2.17× is the idle-machine number and
> supersedes it — see §1.3's caveat.*
>
> Implemented as `terrain.march(..., max_tiles=None)` + `terrain.ray_steps_for()`, with
> `line_of_sight` passing it through. `None` keeps the full `cfg.ray_steps` budget, so every other
> call site is untouched and bit-identical.
>
> **The budget had to be a Python scalar, which forced a design choice worth recording.** It sizes a
> tensor dimension, so it can never be device data — and `params.attack_range.max()` is a host sync
> that has no safe moment to run (`resample_params` executes inside `step()` via autoreset). Solved
> with `config.cone_ray_tiles(spec)`: a static bound derived from the **spec** (taking `high` off any
> randomization range), computed once in pure Python, valid for every reset the env will ever do.
> Stored as `params.cone_ray_tiles`, the one non-tensor field on `SimParams`.
>
> It counts **only kinds with `attack_arc_rad > 0`**. Including ranged kinds would raise the bound
> from 3.0 to 8.67 tiles and give back most of the saving — for rows `melee_hitscan` discards anyway.
>
> **The subtle part, now pinned by a test.** A 6-step ray returns a *wrong* `los` for pairs more than
> 3 tiles apart — a wall 10 tiles away is simply never sampled. That is sound only because `los` is
> consumed in exactly one place, ANDed with `in_cone`, and `point_in_cone` already rejects everything
> past `attack_range`. Every pair whose LOS answer changed was already excluded, so `can_hit` is
> bit-identical. `test_melee_los_budget_matches_full_budget_ray` asserts that equality directly
> against a full-budget reference over randomized states on a wall-heavy map, and guards against
> passing vacuously (it fails if the cone budget ever stops being shorter than the full one).

### Step A2 — Bot phase uses target-only LOS

**Files:** `brawl_sim/bots/policy.py`, `brawl_sim/bots/perception.py`, `tests/test_bot_dispatch.py`

Replace `all_bot_intents`' `los = perception.raw_los(state, bank, cfg)` — an `(N,E,E)` matrix of
which 90% is discarded — with a new `perception.target_los(state, bank, cfg)` returning `(N,E)`: one
ray per entity, to its own `ent_target`. `targeting()` takes that directly instead of gathering a
column out of the matrix.

**Note:** `targeting`'s signature changes (`los` becomes `(N,E)`), and its
`los_enemy = torch.gather(los, 2, idx)` line disappears. Nothing else in the bot phase reads `los` —
verified: `personality.movement` never receives it.

**Acceptance:** `tgt.los` is elementwise identical to the old `gather(raw_los, 2, idx)` over 1000
randomized states. Bot-phase LOS cost 1.391 → 0.404 ms/tick.

> **✅ DONE — 1.428 → 0.575 ms/tick, a 2.48× speedup** (paired, min of 5, idle GPU). At
> `action_repeat=5` that is **~4.3 ms/step**.
>
> *An earlier run reported 4.7×, measured under GPU contention; 2.48× supersedes it.*
>
> `perception.target_los` replaces the `(N,E,E)` matrix with `(N,E)` rays and additionally
> takes A1's ray budget bounded by `cfg.bots_sight_tiles` (28 steps instead of 48), with a `<= 0`
> guard that restores the full budget when the sight limit is disabled.
>
> Also picked up the box-LOS bound via a new `config.attack_ray_tiles(spec)` sibling to
> `cone_ray_tiles` (max `attack_range` over **all** kinds, since any archetype can shoot a box).
>
> **The `bots_sight_tiles` bound rests on a `select_target` property that had to be verified, not
> assumed:** `all_bot_intents` passes it `bot_visibility(...)`, and BOTH of its paths are gated on
> that range-limited matrix — the nearest-visible re-pick *and* the stickiness check. So a live
> `ent_target` is within sight range **this tick**, not merely when it was acquired; a target that
> walks out of range is dropped, not held. If that ever changes, this bound silently breaks.
>
> `test_target_los_matches_the_raw_los_column_it_replaced` pins the equivalence on every row that
> has a target, and asserts untargeted rows actually occur so the exclusion is not silently
> untested — which caught a real gap: on a 20×20 map with `sight_tiles: 14` every *living* entity
> always has a target, so the test has to kill entities to generate the case at all.
>
> **Regression this step surfaced, worth recording:** A1's `params.cone_ray_tiles` broke
> `SimParams`' documented "every field is a tensor" contract, and
> `test_reset_envs_partial_mask_leaves_other_rows_bit_identical` caught it by walking `__slots__`
> and calling `.clone()`. Fixed by making the exception explicit — `SimParams.SCALAR_FIELDS`, an
> enumerated allowlist for *structural* scalars (things that size a tensor dim or drive Python
> control flow and so can never be device data). A new test pins both directions: an undeclared
> non-tensor field fails, and a declared scalar that becomes a tensor fails.

### Step A3 — Stop building the world grid for training

**Files:** `configs/train.yaml`, `configs/default.yaml` (comment), `README.md`

Set `observation.include_world_grid: false` in the training config. The agent provably cannot see it
(`obs_select` only indexes `obs["view"]`); it is built twice per step and discarded.

**Acceptance:** `obs_schema.validate_obs` passes with the field absent (the `conditional` machinery
already handles this — `presets/debug_tiny.yaml` sets it false today). `_observe` + `_autoreset` drop
~6.5 ms/step. Any tool that *does* want the world grid (`render/viewer.py`, `scripts/watch.py`) keeps
it on via its own config.

> **✅ DONE — 188.35 → 178.85 ms/step, 9.5 ms (5.0%), plus 73 MB less peak VRAM** at `n_envs=1024`
> (min of 5, idle GPU, identical reset per measurement). Better than the ~6.5 ms projected. The VRAM
> saving scales with `n_envs`, so at `configs/train.yaml`'s `n_envs: 4096` it is roughly 290 MB.
>
> **Scoped to training via `run.env_overrides`, not applied to `configs/default.yaml`.**
> `training/builder.py` feeds `env_overrides` into BOTH the `EnvConfig` and the `spec` dict, so it
> is the correct hook. Leaving the default ON means `render/viewer.py`, `scripts/watch.py`, and
> `scripts/record_rollout.py` are untouched — they genuinely want the whole map.
>
> Verified end-to-end through the real training stack, not just the config loader: the built env
> reports `obs_include_world_grid: False`, and the agent's observation is unchanged at
> `{self: (21,), enemies: (9,17), projectiles: (12,13), zone: (7,), grid: (10,20,40)}` with a
> `venv.step()` completing normally.

---

## PHASE 1 — Schema and data groundwork

### Step B1 — Rename, restat, and widen the enums

**Files:** `brawl_sim/constants.py`, `configs/brawlers.yaml`, `brawl_sim/config.py`,
`brawl_sim/core/obs_schema.py`, `brawl_sim/render/{viewer,ascii}.py`, `CHARACTER_DETAILS.md`,
`tests/test_constants.py`, `tests/test_config.py`, `tests/test_configs_files.py`

Pure data + naming. **No new mechanics** — every §3 "pure edit" row lands here, and the four archetype
modules are untouched.

1. `Kind` members keep their integer values (slot order is load-bearing everywhere) but get accurate
   comment lines naming Brock/Grom/Buzz/Shelly. **Do not rename the enum members** — `BOT_SNIPER` etc.
   are referenced in 20+ places and the archetype *role* is still the right abstraction.
2. `Proj` gains the one member the later steps need, **now, in one go** (D11/P4): `SUPER_BOLT`.
   `N_PROJ_KINDS` 4 → 5, `projectiles.kind_onehot` widens, agent input width changes once.
   **No `HAZARD_AREA` member** — under D8 a hazard is a projectile *class*, not a weapon identity;
   Brock's sphere keeps `prj_kind = SNIPER_BOLT` so the obs and renderer still know whose it is. This
   is the same reasoning `_spawn_splits` already applies to shards ("Same `Proj` kind as the shell
   that spawned them: they are the same weapon",
   [projectiles.py:299](brawl_sim/core/projectiles.py#L299)).
3. New `ProjClass` IntEnum in `constants.py` (`PROJECTILE=0, ARTILLERY=1, HAZARD=2`), declared here
   but not yet *used* — Step C3 does the `prj_lobbed → prj_class` swap. Declaring it now keeps the
   enum churn in one step.
4. Apply every stat from §3's first table, plus D12's
   `regen.max_hp_fraction_per_second: 0.2 → 0.13` in `configs/default.yaml`.
5. Update `configs/brawlers.yaml`'s header block — its conversion notes (`CastingRange / 3`, the
   `attack_cooldown` DELIBERATE DIVERGENCE section) describe the *old* roster and will be actively
   misleading otherwise.

**Acceptance:** existing tests pass with updated expected values. A 200-step rollout on each map runs
without NaN. `describe_obs` shows the new HP/damage. Deliberately **not** yet accurate: Buzz still
swings once for 840 (not 5×), Brock leaves no sphere, Grom's splits still resolve by distance.

### Step B2 — Projectile budget for 5-pellet volleys

**Files:** `brawl_sim/core/projectiles.py`, `brawl_sim/config.py`, `tests/test_projectiles.py`,
`tests/test_config.py`

`MAX_PROJ_PER_ENTITY = 3 → 5`. **The trap is in `config.validate`**, not in the constant:

```python
if cfg.max_projectiles < cfg.n_entities * max_proj_count * N_SPLITS:   # 10 * 5 * 4 = 200
```

This bound assumes every entity simultaneously fires its max volley *and* every projectile splits.
That is impossible — only artillery splits, and it fires `proj_count: 1`. Naively satisfying it means
`max_projectiles: 128 → 256`, and **that is measurably expensive**: `_projectile_phase` goes
25.74 → 41.64 ms/step (+16 ms, +62%), which is the entire Phase 0 saving spent on headroom nobody
needs.

**Do this instead (D14):** make the bound per-kind-aware —
`max over kinds of proj_count × (1 + splits_for_that_kind)`, summed over the roster — and *measure*
the real steady-state peak before changing `max_projectiles`. Back-of-envelope: 9 Shellys at 5 pellets
living 1.25s on a 1.75s reload ≈ 3.6 live each ≈ 32 total. `P = 128` is very likely ample for bullets
alone.

**Per D14, raising `P` is permitted where measurement justifies it.** Cost is ~0.12 ms/step per slot
at `n_envs=1024`; the agent's input width is unaffected (`max_slots: 12` nearest-K). The ladder:
`128` (free) → `192` (+8 ms, +4%) → `256` (+16 ms, +8.6%, spends the entire Phase 0 saving). **Take
the smallest rung the measurement supports**, and take it after Step C3, not here — hazards hold slots
for 4s and are the dominant consumer.

**Acceptance:** `alloc_slots` correctly places 5 simultaneous pellets per shooter with no collisions.
A diagnostic counter reports peak *and* 99th-percentile live projectiles across a full 3000-tick
episode at `n_envs=256` on every map, broken down by class; `max_projectiles` is then set from that
measurement plus headroom, not from the formula. Dropped volleys (`alloc_slots` returning `ok=False`)
are counted and must be ~0 in steady state.

> **✅ DONE. `max_projectiles: 128 → 192`, and the validate formula was replaced rather than
> patched.**
>
> The old bound — `n_entities * max(proj_count) * N_SPLITS` = 200 — was wrong in *both* directions:
> it assumed every entity fires a max-width volley AND that every projectile splits (no kind does
> both; only artillery splits and it fires `proj_count: 1`), while never modelling the thing that
> actually fills the buffer, which is **projectiles persisting across ticks**. It would have
> rejected a workable config and accepted a broken one.
>
> `config.peak_projectile_demand` replaces it with burst × lifetime:
> `volleys_alive = min(max_ammo, floor(lifetime / attack_cooldown) + 1)`, times `proj_count`, times
> splits; roster worst case is the hero plus `n_enemies` of the most demanding bot kind, since
> `randomize_enemy_types` can legitimately draw an all-one-archetype lobby. For the post-B2 roster
> it returns **135** — matching the hand analysis exactly — so 128 genuinely was too small and
> volleys would have been silently thinned (`alloc_slots` drops the shot; it does not raise).
>
> **Measured peaks (256 envs × 1200 ticks) confirm the bound is conservative but the right shape:**
>
> | lobby | peak, 3 pellets | peak, 5 pellets |
> |---|---|---|
> | shipped archetype weights | 35 | **62** |
> | all-Shelly (`fixed_enemy_types`) | 49 | **102** |
> | all-Grom | 47 | 47 |
>
> 192 covers the 135 bound with 57 slots spare and the measured 102 with 90. 256 was not needed;
> per D14 that keeps the cost at ~7.7 ms/step rather than ~15.
>
> A separate guard covers the *other* silent failure: `MAX_PROJ_PER_ENTITY` sizes `spawn_volley`'s
> `(N,E,K)` grid, so a kind with `proj_count > K` fires only part of its volley — and
> `peak_projectile_demand` does not catch that, because buffer capacity and volley width are
> different questions. `test_volley_width_constant_covers_every_kinds_proj_count` asserts it
> directly.

---

### Step B3 — Proportional zone damage

**Files:** `brawl_sim/core/zone.py`, `brawl_sim/config.py`, `configs/default.yaml`,
`brawl_sim/core/{observation,obs_schema}.py`, `tests/test_zone.py`

**D15: nothing survives more than ~5 seconds in the zone.** That is a proportional requirement — it
has to hold for a 6000 HP Brock *and* a fully-cubed 34000 HP Buzz — so a flat `zone.dps` cannot
express it at all. Today's 1000 flat (+200/shrink) lets a base-HP Buzz stand in the zone for 10s and
a cubed one for **34s**, which is the opposite of a hard boundary.

This is also the divergence `configs/default.yaml`'s own comment already flags: real Showdown poison
deals a percentage of max HP per second, not a flat rate. So the fix is simultaneously a balance
requirement and an accuracy correction.

- Add `zone.max_hp_fraction_per_second` (**0.20** = 5s to die from full) and
  `zone.fraction_growth_per_step`, mirroring `regen.max_hp_fraction_per_second`'s naming exactly —
  that field is the established precedent for "a fraction of max HP per second" in this config.
- `zone.zone_damage` computes `frac * state.ent_max_hp * dt` per entity instead of broadcasting one
  `(N,)` scalar across `E`. `ent_max_hp` is already cube-scaled, so cube stacking no longer buys
  zone tankiness.
- **Rename rather than reinterpret** `zone.dps` → the new key, exactly as
  `regen.per_second` → `regen.max_hp_fraction_per_second` was handled: a config still carrying
  `dps: 1000.0` must not be silently read as "10000% of max HP per second". `config.validate` should
  reject the old key and any fraction outside `[0, 1]`.
- `zone.current_dps` currently returns `(N,)` for `obs["zone"]["dps"]`. Keep the shape, but return
  the **hero's** effective HP/s (`frac * ent_max_hp[:, 0]`) — that is both more meaningful to the
  agent than a rate abstracted away from any particular body, and unchanged in dtype/shape, so the
  observation schema needs only a description edit.

**Acceptance:** an entity of any max HP standing in the zone dies in `1/frac` seconds ±1 tick — check
Brock (6000), Buzz (10000), and a cube-stacked 34000 HP entity, all of which must take the same
number of ticks. Escalation still applies per shrink step. `zone_enabled: false` is still all-zero.

> **✅ DONE.** Measured against the shipped config: 6000 / 8000 / 10000 / 34000 HP bodies all die in
> **exactly 100 ticks = 5.00 s**, and by shrink step 5 the rate is 0.40/s → 2.50 s. Under the old
> flat `dps: 1000` those same bodies lasted 6 s / 8 s / 10 s / **34 s**.
>
> Split the rate helper in two rather than overloading one name: `current_fraction()` is the
> per-second fraction (what `zone_damage` uses), `current_dps()` is that fraction × the **hero's**
> max HP, which keeps `obs["zone"]["dps"]` at `(N,)` and makes it mean "how fast does the zone kill
> *me*" instead of a rate detached from any body.
>
> **The rename's failure mode is silent, so it needed its own guard.** `regen.per_second` →
> `regen.max_hp_fraction_per_second` got caught by a range check, because the stale value (400.0)
> was absurd as a fraction. That does not work here: a config still on `zone.dps` simply has the new
> key resolve to `_dget`'s default of **0**, which is an ordinary-looking number that produces a
> completely inert zone — surfacing much later as "episodes never resolve", the exact symptom this
> step exists to fix. `config.validate` now rejects `zone_enabled and fraction <= 0` outright.
>
> That guard immediately earned itself: it failed `test_config.py::test_validate_passes_on_default`,
> whose inline YAML fixture was still carrying `dps: 1000.0`. A real stale config, caught at
> construction rather than as mysterious behavior later.
>
> `test_zone_kills_any_body_in_the_same_time_regardless_of_max_hp` asserts the D15 property
> directly, and `_fresh_state` now fills `ent_max_hp` — `allocate` zero-inits it, which was harmless
> for a flat rate but would have made every proportional-damage assertion compare 0 to 0.

---

## PHASE 2 — The four new mechanics

Each step adds exactly one mechanic, and each is gated so its config field at 0 restores prior
behavior bit-for-bit. C1 must land before C2 (C2's sweep clock is the timer C1 gives meaning to);
C3 and C4 are independent of both.

### Step C1 — Firing pauses the reload

**Files:** `brawl_sim/core/hero.py`, `configs/brawlers.yaml`, `tests/test_hero.py`,
`tests/test_action_repeat.py`

Closes **Gap 1**, and under D7 it is now the smallest step in the plan rather than the largest.

**The whole code change is one line.** In [`hero.tick_timers`](brawl_sim/core/hero.py#L71):

```python
# before
state.ent_ammo.copy_(torch.clamp(state.ent_ammo + cfg.dt / reload_seconds, max=max_ammo))
# after
gain = torch.where(state.ent_attack_cd <= 0, cfg.dt / reload_seconds, torch.zeros_like(reload_seconds))
state.ent_ammo.copy_(torch.clamp(state.ent_ammo + gain, max=max_ammo))
```

Everything else D7 asks for **already works**: `ent_attack_cd` is already set to `attack_cooldown` on
every attack (`_attack_phase` for ranged/melee, `start_dash` for dashes), already decremented here,
and already gates re-attacking in all three of `_attack_phase.can_attack`, `hero.action_mask.fire_ok`,
and `policy.fire_gate`. No new state field, no new param, no `env.py` change.

Then apply §3's `attack_cooldown` column: 0.25 floor everywhere, Mortis 0.35, Buzz 1.00.

**Ordering note:** `tick_timers` runs in phase 2 and `_attack_phase` in phase 6, so an entity that
fires on tick *t* has `ent_attack_cd == 0` when phase 2 runs on tick *t* and banks one tick (0.05s) of
reload before its own shot stops it. Identical in shape to the regen leakage `combat.apply_regen`
already documents and accepts, and worth one comment rather than a reordering.

**Update `hero.tick_timers`' docstring** — it currently asserts "reload continues during a dash",
which this step makes false (a dash sets `attack_cooldown` too).

**Acceptance:** an entity that fires at t=0 has *exactly* the same `ent_ammo` at t=`attack_cooldown`
as it did at t=0⁺, and resumes accruing at `1/reload_seconds` immediately after. Buzz's ammo is frozen
for a full 1.0s; Brock's for 0.25s. Zero-to-full from idle (no attack) is unchanged at
`max_ammo × reload_seconds`. Sustained fire rate is `attack_cooldown + reload_seconds` per shot,
matching §1.4's table.

> **✅ DONE**, and it really was one line of logic plus five config values — D7's collapse of the
> `attack_anim_seconds` / `ent_attack_anim_t` subsystem held up completely. No new state field, no
> new param, no `env.py` change.
>
> Cooldowns: Brock / Grom / Shelly **0.25** (the floor), Mortis **0.35** (must outlast
> `dash_duration` 0.30, or the dash masks it entirely and the reload pause never fires), Buzz
> **1.00** (the sweep *is* the cooldown).
>
> Realised sustained DPS, matching §1.4's predictions to the digit:
>
> | | cooldown | reload | cycle | dmg/shot | sustained | full-clip burst |
> |---|---|---|---|---|---|---|
> | Shelly | 0.25 | 1.75 | 2.00 | 3000 | **1500** | 9000 over 0.50s |
> | Buzz | 1.00 | 1.00 | 2.00 | 2520 | **1260** | 7560 over 2.00s |
> | Brock | 0.25 | 2.10 | 2.35 | 2320 | **987** | 6960 over 0.50s |
> | Grom | 0.25 | 2.30 | 2.55 | 2080 | **816** | 6240 over 0.50s |
> | Mortis | 0.35 | 2.25 | 2.60 | 2000 | **769** | 6000 over 0.70s |
>
> Note how much burst now dominates: a full Shelly clip is 9000 damage in half a second, enough to
> delete any brawler in the roster from full — but across a 60° spread, so only point-blank. That
> is the intended shape (D16: fights are decided by burst, not by sustained trading).
>
> **`peak_projectile_demand` did NOT rise, which is worth recording** — the cooldown drop to 0.25
> raises `floor(lifetime/cooldown)+1` for everyone, but Shelly was already capped by `max_ammo: 3`,
> so the roster worst case stays at 135 and the 192 budget is untouched. The check earning its keep:
> it would have caught the opposite outcome automatically.
>
> `test_shipped_cooldowns_all_exceed_one_decision_window` was rewritten, and one of its three
> assertions **deleted rather than re-valued**: `cooldown < reload_seconds`, whose stated rationale
> was "this costs no sustained DPS because ammo regen is still the binding constraint". D7 makes
> that false by design. It was also numerically false for Buzz (1.00 vs 1.00). Keeping it re-valued
> would have pinned a philosophy the design deliberately moved away from.

### Step C2 — Swept multi-hitscan melee (Buzz)

**Files:** `brawl_sim/core/{combat,movement}.py`, `brawl_sim/env.py`, `brawl_sim/bots/melee.py`,
`brawl_sim/config.py`, `configs/brawlers.yaml`, `tests/test_combat.py`, `tests/test_melee.py`

Closes **Gap 3**. Generalizes "melee = one instantaneous cone" to "melee = `hitscan_count` cones
spread across `attack_cooldown`, sweeping `hitscan_sweep_rad`". `hitscan_count ≤ 1` is exactly
today's behavior, so every non-Buzz kind is unaffected by construction.

**Also no new state fields** — the sweep clock is `ent_attack_cd` and the sweep anchor is
`ent_facing`:

- **Clock.** With `elapsed = attack_cooldown - ent_attack_cd` and
  `interval = attack_cooldown / hitscan_count`, sub-swing `k` fires on the tick where
  `floor(elapsed/interval) > floor((elapsed - dt)/interval)`, with `k = floor(elapsed/interval)`.
  Same stateless pattern as Step C3's hazard schedule: no counter to keep consistent, and no swing
  dropped or doubled if `dt` doesn't divide `interval`. At `cooldown 1.0, count 5` this fires at
  t = 0, 0.2, 0.4, 0.6, 0.8 s.
- **Anchor (D6).** Suppress [`apply_movement`'s facing update](brawl_sim/core/movement.py#L64) while
  `ent_attack_cd > 0 AND hitscan_count > 1`, so `ent_facing` stays pinned at whatever it was when the
  sweep started. Buzz walks freely but cannot re-aim. **Gate on `hitscan_count > 1`, not on the
  cooldown alone** — otherwise every brawler in the game would freeze its facing for 0.25s after each
  shot, which is a much larger behavior change than this step is scoped for.
- **Angles.** Sub-swing `k`'s cone center is
  `ent_facing - sweep/2 + k * sweep/(count-1)`. **Clockwise = increasing angle**, since y increases
  downward (CONVENTIONS.md §"Coordinates").
- `melee_hitscan` takes the cone center as an argument instead of reading `state.ent_facing` directly.
- `bots/melee.py`'s fire gate widens to the *sweep's* reachable arc (`sweep/2 + arc/2`), not the
  single-cone arc — otherwise Buzz never starts a sweep whose later swings would have connected.

**Cost:** none. `melee_hitscan` already runs every tick over `(N,E,E)`; this changes which entities
have `fire=True` and where the cone points.

**Acceptance:** a stationary target at the sweep's center takes exactly 3 hits (2520) — the D9
arithmetic, verified empirically. A target at the sweep's edge takes 1–2. Sub-swings land at
t = 0, 0.2, 0.4, 0.6, 0.8 s ±1 tick. Buzz's `ent_facing` is constant for the whole sweep while his
position changes. With `hitscan_count: 1`, output is identical to Step C1's.

> **✅ DONE**, in a new `core/melee_sweep.py`, and the "no new state fields" claim held — the clock
> is `ent_attack_cd` (which C1 gave meaning to) and the anchor is `ent_facing` (which movement now
> stops updating mid-sweep).
>
> Measured: 5 sub-swings at t = 0.00 / 0.20 / 0.40 / 0.60 / 0.80 s, spanning **70.02°**, clockwise.
> Sweeping a target across every bearing from −50° to +50°, the maximum that ever connect is
> **exactly 3** (2520 damage) — D9's arithmetic confirmed empirically rather than trusted.
>
> **The sweep costs nothing in the attack phase.** Swept and single cones resolve in ONE
> `melee_hitscan` call with a per-entity `cone_dir`, not two calls: that function is a dense
> `(N,E,E)` cone + LOS march and the most expensive thing in the phase, and the two populations are
> disjoint per entity so a second call would have doubled it to buy nothing. `_attack_phase` in fact
> came *down* slightly (34.8 → 33.2 ms/step).
>
> Three subtleties worth recording:
>   - **A swept kind is masked out of the trigger-time cone** (`& ~swept`). Sub-swing 0 already
>     fires on that tick, so letting the ordinary path through as well would double the first swing.
>   - **The facing freeze is gated on the KIND, not on `ent_attack_cd > 0`.** Since C1 every brawler
>     has a cooldown running after every shot, so the looser gate would have pinned all five
>     brawlers' facing for 0.25 s after every attack — silently changing dash directions and every
>     melee cone in the game. `test_a_ranged_kind_still_turns_freely_while_on_cooldown` guards it.
>   - **`bots/melee.py`'s fire gate widened to `sweep/2 + arc/2`.** Gating on `arc/2` alone would
>     make Buzz refuse to start a sweep whose 2nd–5th sub-swings would have landed squarely.
>
> The schedule is deliberately stateless (a `floor()` crossing on elapsed time, the same trick C3's
> hazard ticks use), and `test_schedule_never_drops_or_doubles_a_swing_when_dt_does_not_divide_the_interval`
> forces a 0.30 s cooldown / 0.06 s interval against a 0.05 s tick to prove it does not drift.

### Step C3 — Projectile classes and lingering hazards (Brock)

**Files:** `brawl_sim/constants.py`, `brawl_sim/core/{state,projectiles}.py`,
`brawl_sim/core/{observation,obs_schema,obs_select}.py`, `configs/{brawlers,agent_obs}.yaml`,
`brawl_sim/render/viewer.py`, `tests/test_projectiles.py`, `tests/test_artillery.py`

Closes **Gap 2** via D8. Two sub-changes; do them in this order so the first is provably inert.

**C3a — `prj_lobbed` (bool) → `prj_class` (enum).** Pure refactor, no behavior change.
`ARTILLERY = 1` reproduces `lobbed=True` exactly and `PROJECTILE = 0` reproduces `lobbed=False`; every
`~state.prj_lobbed` becomes `prj_class == PROJECTILE`, and `is_lobbed` in `spawn_volley` becomes a
class assignment. `obs["projectiles"]["lobbed"]` becomes `class` + `class_onehot` (D13).

**C3b — the `HAZARD` class.** In `step_projectiles`:

- **Spawn.** When a `PROJECTILE`-class projectile whose owner's kind has `on_hit_area_radius > 0`
  dies by unit *or* wall hit, seed a `HAZARD` at the hit point. This is the same shape as
  [`_spawn_splits`](brawl_sim/core/projectiles.py#L232) and reuses `alloc_slots` verbatim — already
  documented as algorithm-generic, and already reused by `combat.py` for pickups. Like splits, it
  runs **after** `prj_alive` is updated so the hazard can claim the slot its own bullet just freed.
- **Stats come from the owner's kind**, gathered via `prj_owner` + `ent_kind` — exactly how
  `_spawn_splits` resolves shard stats today, and for the same reason: `ent_kind` outlives the
  entity's death, so a hazard whose Brock died still ticks correctly.
- **Physics.** `vel = 0`; skipped by every collision branch (`wall_kill`, `unit_kill`, `box_kill`) —
  the same gate `ARTILLERY` already uses. D5: no LOS check, it is a ground area.
- **Schedule — stateless, derived from `prj_age`.** No new per-projectile fields:
  ```
  ticks_done = floor((age - dt) / interval)
  ticks_now  = floor( age       / interval)
  fires this tick  <=>  ticks_now > ticks_done
  expires          <=>  ticks_now >= on_hit_area_ticks
  ```
  At `interval: 2.0, ticks: 2` this fires at t=2.0s and t=4.0s and then expires — **D3(a) falls out
  for free**, with no counter to keep consistent and no drift if `dt` doesn't divide `interval`.
- **Damage** reuses the detonation AoE test already in the file
  ([`projectiles.py:396`](brawl_sim/core/projectiles.py#L396) — `ent_dist <= prj_aoe`, then
  `scatter_add_` by owner), gated on class and the tick boundary. D4: mask out the owner's own column;
  stacking is automatic since two hazards are two slots.

**Cost:** the `(N,P,E)` AoE distance test already runs every tick for detonations — hazards ride it
for free. The only new cost is slot *occupancy*: see the Step B2 note below.

**⚠ This changes Step B2's answer.** A bullet occupies a slot for ~1.5s; a Brock sphere occupies one
for **4s**. Nine Brocks landing a hit every ~2.1s hold ~2 spheres each ≈ 18 slots of standing
occupancy, on top of in-flight pellets. Step B2's peak-live-projectile measurement must be re-run
*after* this step, not before it.

**Acceptance:** C3a is byte-identical to Step C2 on a fixed-seed 500-tick rollout. C3b: a Brock rocket
that connects deals 2320 on impact, then exactly 696 at t=+2.0s and 696 at t=+4.0s to anything inside
0.75 tiles, then frees its slot. Two overlapping spheres deal 1392/tick. The owner takes nothing. A
hazard is not stopped by, and does not stop, anything walking through it. `on_hit_area_radius: 0`
everywhere reproduces C3a exactly.

> **✅ DONE.** Measured: **2320 on impact, 696 at +2.05 s, 696 at +4.05 s, total 3712**, owner takes
> 0. The 0.05 s lag is one tick of float accumulation in `prj_age` (by 4 s it trails the exact value
> by a hair, so the `floor()` crossing lands one tick late) — harmless, and the test tolerance says
> so explicitly rather than hiding it.
>
> **The hazard tick costs essentially nothing.** "Everyone within `prj_aoe` of this point takes
> `prj_damage`" is *exactly* the query the artillery detonation already performs, so a firing hazard
> is folded into that same `(N,P,E)` computation as `blast = detonate | haz_fires`. The added cost is
> two boolean ops — no second distance pass, no new buffer, no new tick phase, which is what D8's
> unified-class design bought over the separate `haz_*` buffer originally proposed.
>
> **A real ordering bug caught while writing it:** `died_projectile` reads `prj_class`, but both
> spawners write `prj_class` into the very slots those deaths just freed — so computing it after
> `_spawn_splits` would have read the NEW occupant's class and mis-attributed the death. It is now
> captured before either spawner runs, with a comment saying why.
>
> **Budget re-measured after the change, as promised.** A sphere holds a slot for 4 s versus a
> rocket's 1.5, so `peak_projectile_demand` gained a hazard-residency term — but the answer is still
> **135**, because Brock is already capped by `max_ammo: 3`. Measured peaks: shipped 68, all-Shelly
> 99, all-Brock **35**, all-Grom 48. The 192 budget stands untouched.
>
> The viewer draws hazards as translucent circles at their true `prj_aoe` rather than scatter dots —
> a dot would be actively misleading, since what matters when watching a rollout is the *ground*
> being denied, and a 0.75-tile sphere as a 15-point marker reads as a stray bullet sitting still.
>
> D13 landed too: `projectiles.class_onehot` is in `agent_obs.yaml`. Without it a puddle and a bullet
> are near-identical rows to the policy — a hazard has zero velocity, but so does a bullet at its
> moment of closest approach, and both carry a damage figure and an `aoe`.

### Step C4 — Grom's timed splits

**Files:** `brawl_sim/core/projectiles.py`, `configs/brawlers.yaml`, `tests/test_artillery.py`,
`tests/test_projectiles.py`

Smallest of the four. Today a shard's lifetime is `split_distance` tiles at `proj_speed` tiles/s
(1.2 / 6.0 = 0.2s). CHARACTER_DETAILS specifies **0.5s**, so express it directly: add `split_seconds`
and derive `shard_speed = split_distance / split_seconds` in `_spawn_splits`, rather than making the
config author solve for a speed. `split_damage_fraction: 1.0` (splits deal the full 2080).

**Re-check `_spawn_splits`' rim geometry** ([`projectiles.py:273`](brawl_sim/core/projectiles.py#L273)):
its comment proves shards can never touch anything already caught by the blast, which keeps the
"full damage OR half damage, never both" partition. With `split_damage_fraction: 1.0` that partition
becomes "2080 either way" — the geometry still matters (no double-dipping) but the comment's framing
needs updating.

**Acceptance:** a shell fired at t=0 lands at t=1.25s ±1 tick; its four shards expire at t=1.75s ±1
tick. Damage is 2080 on the landing tile and 2080 along the arms, and never both to the same victim.

> **✅ DONE.** Measured exactly: shell detonates at **1.25 s**, arms finish **0.50 s** later at
> 1.75 s. Shard speed is now derived — `split_distance / split_seconds` = 1.2 / 0.5 = **2.4 tiles/s**,
> down from `proj_speed` 6.0.
>
> **That 2.5× slowdown is the real content of this step, not the timing.** At 6.0 tiles/s a shard
> crossed its 1.2-tile arm in 4 ticks — effectively instant, nothing to react to. At 2.4 the cross
> takes 10 ticks to form, so it can be seen and stepped out of. Combined with B1's
> `split_damage_fraction: 1.0`, that is the whole shape of Grom: a big, slow, readable threat.
>
> **The rim geometry became MORE load-bearing, and its comment was rewritten to say so.** It used to
> justify itself as preventing a 3200 + 1600 overlap; with full-damage shards the same overlap would
> now be 2080 + 2080 = **4160 from one shell**, two thirds of a brawler's health on a single tile.
> The comment previously described an invariant B1 had already invalidated.
>
> **The budget formula's two special cases were merged into one `dwell` concept** rather than
> growing a third branch: a hazard's lingering ticks and a shard's flight are both "how long a shot
> keeps occupying slots after its own projectile ends", combined with `max()` (no kind does both).
> `residency = flight_life + dwell`. Demand stays **135** and the 192 budget holds — Grom is
> `max_ammo`-capped at 3 volleys — but a future ammo or cooldown change now gets caught.
>
> One more stale-derivation test: `test_detonation_splits_into_four_axis_aligned_half_damage_shards`
> read shard speed from `proj_speed`, silently asserting the pre-C4 rule. It now derives the speed
> the same way the code does, and a new sibling pins the **duration** — the thing CHARACTER_DETAILS
> actually specifies.

---

## PHASE 3 — Hero abilities

### Step D1 — Long dash charge

**Files:** `brawl_sim/core/{state,hero}.py`, `brawl_sim/env.py`,
`brawl_sim/core/{observation,obs_schema,obs_select}.py`, `configs/{brawlers,agent_obs}.yaml`,
`tests/test_hero.py`

New `(N,E)` f32 stopwatch `ent_attack_idle_t`, counting **up**, reset **only** by attacking. Do *not*
reuse `ent_out_of_combat_t` — §1.2's last note explains why (it is also reset by taking damage, so a
Mortis under fire would never charge his long dash).

`long_dash_ready = ent_attack_idle_t >= long_dash_seconds`. In `hero.start_dash`, the pre-clip
distance becomes `dash_distance * where(long_dash_ready, long_dash_multiplier, 1.0)`. Everything
downstream (the terrain `march` clip, the `safe_hit_t` back-off, `dash_speed = clipped/duration`)
already handles a longer distance correctly — **`dash_duration` stays 0.30s**, so a long dash is
genuinely faster, which is what it is in the real game.

Obs: `hero.long_dash_ready` (bool) and `hero.long_dash_frac` (0→1 ramp — the continuous version is
what a policy can actually plan against). Add both to `agent_obs.yaml`'s `self` group.

**Acceptance:** after 4.5s without attacking, the next dash covers 5.34 tiles instead of 2.67 and the
flag clears on use. Attacking resets the stopwatch; taking damage does not. Wall clipping still leaves
`unit_radius` of clearance at the doubled distance (the existing regression test for
[`hero.py:118`](brawl_sim/core/hero.py#L118) must be re-run at 2× range).

> **✅ DONE — and re-running the wall clip at 2× range was the right call: it exposed TWO real
> pre-existing bugs, both of which could permanently trap the hero.**
>
> The ability itself works as specified: charge reaches 1.00 at 4.5 s, the dash covers exactly
> **2.00×** the distance, and the charge is spent on use. `ent_attack_idle_t` is a new stopwatch
> reset only by attacking — reusing `ent_out_of_combat_t` would have meant a Mortis under fire never
> charges it, since that field is also reset by taking damage.
>
> **Bug 1 — every dash in the game travelled 1.167× its configured distance.** `dash_duration: 0.30`
> is not representable in float32; after six subtractions of `dt` it leaves **1.19e-8 s**, which is
> still `> 0`, so `is_dashing` stayed True for a seventh tick. Present since Step 11 and affecting
> the ordinary 2.67-tile dash too (it actually covered 3.11). Fixed by advancing only the time that
> remains: `step_seconds = clamp(ent_dash_t, max=dt)`.
>
> **Bug 2 — a dash could land the hero permanently stuck.** `terrain.circle_blocked` probes all 8
> compass points at a candidate destination, so an entity whose circle merely OVERLAPS a wall is
> blocked in *every* direction, including away from it — and `resolve_move` then rejects every
> escape step identically, forever. `start_dash` backs a clipped dash off to prevent exactly this,
> but it marched only as far as `dash_distance`, so a wall between the last 0.5-tile sample and the
> landing point was never detected and no back-off happened. Fixed by probing
> `dash_distance + los_step_tiles + radius` (with `minimum(safe_hit_t, dash_distance)`, so a wall
> found beyond the landing point can only ever shorten the dash), plus a 1-millitile clearance —
> backing off *exactly* the geometric amount leaves the hitbox edge on the wall face, and
> `floor(30.0)` is the wall tile, so touching still reads as blocked.
>
> Bug 1 made Bug 2 reachable: the overshoot silently spent the very clearance margin the back-off
> computes. D1 did not introduce either, but doubling the dash distance made them reproducible.
>
> Verified by sweeping 800 approach positions at 0.01-tile resolution across both dash lengths —
> **0 stuck cases**, against 29/200 before the fix. Kept as a regression test, along with one
> pinning that a dash travels exactly its clipped distance.
>
> Obs carries both `hero.long_dash_ready` (the discontinuity that changes what a dash does) and
> `hero.long_dash_frac` (the ramp a policy can plan against); a flag alone gives no gradient, a
> fraction alone hides the moment the ability arrives.

### Step D2 — The Super

**Files:** `brawl_sim/core/{state,hero,projectiles,combat}.py`, `brawl_sim/env.py`,
`brawl_sim/config.py`, `brawl_sim/wrappers/{sb3_vecenv,gym_single}.py`,
`brawl_sim/core/{observation,obs_schema,obs_select}.py`, `configs/*.yaml`, `scripts/play_manual.py`,
`tests/test_hero.py`, `tests/test_projectiles.py`

Closes **Gaps 4 and 5**. The largest step; keep it last.

**Charge.** New `(N,E)` i32 `ent_super_charge`, incremented in `_bookkeeping` from this tick's
`dmg_by_total` — one per (attacker, victim) pair that dealt damage to a **living player**, never a
box. Caps at `super_charge_hits`.

**D2 — hero only, for now.** Build every piece of this kind-agnostically (the charge counter is
`(N,E)`, `super_*` are per-kind params, the projectile spawn reads the owner's kind), but configure
`super_charge_hits` only for `hero_mortis`; every bot resolves it to 0 and never charges. Giving a bot
a super later is then a `configs/brawlers.yaml` block plus one `fire`-source change in
`bots/combat_rules.py` — **not** a rewrite of this step. The one piece that is genuinely hero-shaped
is the action-space wiring below, and that is correct: bots don't have an action space.

**Action space — D1, settled: widen the attack dim.** `cfg.action_nvec` becomes
`(n_move_bins + 1, 3)`. `hero.action_mask` returns a 3-wide attack mask `[no-fire, attack, super]`;
`decode_action` returns `(move_dir, fire, super_fire)`. [`_held`](brawl_sim/env.py#L373)'s
`held[..., 1] = 0` already zeroes the attack column and keeps working **unchanged** — the super is a
distinct *value* in that column, not a new column, so "one decision is at most one attack attempt"
holds for supers automatically. `override`'s `(N,E,2)` shape and its `-1` sentinel in column 0 are
likewise untouched. This is the entire reason widening beats adding a dimension.

**The projectile.** `Proj.SUPER_BOLT`, spawned by a new branch in `spawn_volley`:
- ignores walls — skip the `wall_kill` branch where `prj_pierce`
- ignores unit death — skip `unit_kill`, so it keeps flying
- hits each victim once — new `(N,P,E)` bool `prj_hits`, exactly the pattern `ent_dash_hits` already
  uses for the dash capsule (see [`hero.advance_dash`](brawl_sim/core/hero.py#L159))
- `dist_left = super_range` (10 tiles), `radius = dash_radius` (0.70), `damage = super_damage` (1800)

**Lifesteal.** `apply_damage` has no heal path. Add `combat.apply_heal(state, heal, params)` clamped
to `ent_max_hp`, called from `_projectile_phase` with `heal = super_heal × (players hit this tick by
a super owned by this entity)`. Boxes never heal.

**Obs:** `hero.super_charge`, `hero.super_charge_frac`, `hero.super_ready`; add the fraction and the
flag to `agent_obs.yaml`'s `self` group.

**Acceptance:** the super is masked until 5 hits land on players, and box hits never charge it. It
passes through a wall and through the first enemy it hits, damaging both, and damages neither twice.
Mortis heals exactly 1800 per player hit, clamped at max HP. `MaskablePPO` trains for 10k steps
without an action-space assertion. `scripts/play_manual.py` can fire it.

> **✅ DONE.** Verified end to end: the bolt pierces the first enemy, then a wall, then hits a second
> enemy behind both — 1800 each, 3600 lifesteal, owner untouched, charge caps at 5.
>
> **Nothing is hero-special-cased**, which is the point given that bot supers are planned.
> `ent_super_charge` is `(N,E)`, every `super_*` stat is per-kind, `spawn_supers` takes an `(N,E)`
> fire mask, and `BotIntent` already carries a `super_fire` field (all-False today) plumbed through
> to `_attack_phase`. Giving a bot a super is a `brawlers.yaml` block plus a decision in its combat
> rule — no plumbing change. `test_a_kind_without_a_super_never_charges_or_becomes_ready` pins that
> the gate is `super_charge_hits > 0` rather than "is this entity slot 0".
>
> **`attacked = fire | super_fire`.** A super has to break concealment, reset out-of-combat regen,
> AND spend the long-dash charge. Missing any one would make it a free way to shoot from a bush,
> out-heal a fight, or keep a long dash banked while attacking — all three are covered by one
> integration test.
>
> **Charge comes from the attacker × victim damage matrix, not a "did I deal damage" flag.** That is
> what excludes boxes (box damage never enters `dmg_by`), so the hero cannot farm his super off
> crates, and it makes a multi-target hit count once PER VICTIM — which is what "5 hits against
> other players" means for Buzz's sweep or Shelly's spread. Both alternatives look identical in
> every other test, so `test_super_charge_comes_from_player_hits_not_box_hits` checks it directly.
>
> **Supers ignore boxes entirely.** Damaging them would need a second `(N,P,B)` hit-memory buffer to
> stop re-hitting the same crate every tick, and CHARACTER_DETAILS describes the super purely in
> terms of players.
>
> `step_projectiles` returns a 4-tuple now (`heal_ent` added) rather than mutating `ent_hp` in
> place. That churned ~25 call sites, but the module's contract is "compute, the caller commits",
> and healing has to respect the same never-touch-a-dead-entity rule `apply_damage` centralises —
> an undocumented mutation inside a function named for reporting is exactly the kind of thing that
> bites six months later.
>
> `prj_hits` `(N,P,E)` bool is the largest new buffer in the overhaul: **7.9 MB at 4096 envs**
> against a measured 2.2 GB total. Cleared on DEATH rather than spawn — a slot is only ever
> reallocated after it dies, so that is sufficient, and it is a single masked AND instead of a
> gather/scatter that could let an invalid claim clobber a live projectile's memory.

---

## PHASE 4 — Generalize and verify

### Step E1 — Collapse the archetype modules into a fire-rule table

**Files:** `brawl_sim/bots/policy.py`, new `brawl_sim/bots/combat_rules.py`, delete
`bots/{sniper,artillery,melee,rifle}.py`, `configs/brawlers.yaml`, `tests/test_bot_dispatch.py`,
consolidate `tests/test_{sniper,artillery,melee,rifle}.py`

This is goal (2) — and it is deliberately last, because it is a pure refactor and refactoring against
a moving target is how mechanics get lost.

After Phases 1–3, the four `combat()` functions differ in exactly four data-expressible ways:

| | needs LOS | fire fraction of range | cone check | aim model |
|---|---|---|---|---|
| sniper | yes | 1.0 | no | angular lead |
| artillery | **no** | 1.0 | no | **positional**, timed lob |
| melee | yes | 1.0 | **yes** | none |
| rifle | yes | **0.9** | no | angular lead + **lateral-mover hold** |

Replace all four modules with one `combat()` reading four new per-kind params: `fire_needs_los`
(bool), `fire_range_fraction`, `fire_needs_cone` (bool), `aim_model` (enum: `NONE` / `ANGULAR` /
`POSITIONAL`), plus the existing `lateral_hold_speed` / `lateral_hold_range_fraction` as ordinary
numeric fields (0 = disabled). `_archetype_modules()` and the `torch.where`-by-kind selection chain
disappear entirely.

**Bonus:** this should also be a speedup. Today every archetype's `combat()` runs over all `(N,E)`
entities and 3 of 4 results are discarded ([`policy.py:416`](brawl_sim/bots/policy.py#L416)) — a
documented 4× waste on the fire/aim half of the bot phase. One data-driven pass computes each
entity's answer once. **Measured in E2, not here:** the machine was 63% busy with an unrelated
process when E1 landed, and §1.3's caveats are the record of what measuring under contention cost
this project twice already.

**Acceptance:** for 1000 randomized states, the table-driven `combat()` returns `(fire, aim_dir,
aim_point)` identical to the four-module dispatcher (modulo RNG draw order — seed the comparison
carefully). A **fifth** brawler can be added as a `configs/brawlers.yaml` block with no Python change;
ship one as a test fixture to prove it. `_bot_phase` cost drops.

> **✅ DONE.** The four modules are deleted. `bots/combat_rules.py` is one `combat()` reading five
> per-kind fields; `policy._archetype_modules` and the four-way `torch.where` chain are gone.
>
> **Equivalence was proven before anything was deleted**, over 5 seeds x 64 envs x 10 entities
> (~2,500 bot rows per seed, with walls, boxes, dead entities, empty clips and live cooldowns):
> with the noise parameters zeroed, the new rule returns `(fire, aim_dir, aim_point)` **bit-identical**
> to the four-module dispatcher for all four kinds. With noise LIVE, `fire` is bit-identical for
> all four (it consumes no randomness at all) and so are sniper/artillery/melee aim.
>
> **One deliberate difference: the RNG stream.** The old dispatcher drew noise three times per tick
> (sniper's angular, artillery's positional, rifle's angular) because each module drew its own; the
> new rule draws twice — one `(N,E)` angular tensor shared by every LEAD kind, one `(N,E,2)`
> positional tensor for every LOB kind. Rifle therefore reads the draw sniper used to get. Nothing
> correlates that did not before (two entities read different cells of the same tensor), and the
> distribution is unchanged — measured over 2,308 rifle samples, the applied angular offset is
> N(-0.0009, 0.0798) against a configured 0.08. What does change is that a fixed seed produces a
> different, statistically identical rollout than it did pre-E1. D11 makes that a non-issue.
>
> **Two deviations from the plan above.**
>
> 1. **`fire_needs_cone` is NOT a field.** It is exactly `attack_arc_rad > 0`, which is already how
>    `config.cone_ray_tiles` and `combat.melee_hitscan` identify a cone swinger. A second field
>    saying the same thing is a second field that can disagree with the first — and the disagreeing
>    case (`fire_needs_cone: true, attack_arc_rad: 0`) is a bot that can never fire, since
>    `point_in_cone` with a zero half-angle is never satisfiable. Four fields, not five.
>
> 2. **`desired_range_fraction` was added to the step.** `policy.RANGE_FRACTION_BY_KIND` was a
>    Python tuple indexed by `Kind` carrying `assert len(...) == N_KINDS` — a sixth brawler would
>    have tripped it on import. Leaving it behind would have meant E1 missed its own goal by one
>    line, so it moved into the same per-kind table.
>
> **Where the fifth axis went.** `aim_model` is an enum name string in YAML (like `proj_kind`), not
> a number, so `proj_kind`'s one-off resolution block became a general `PER_KIND_ENUM_FIELDS` table.
> `AimModel.DIRECT` is deliberately 0: a kind that forgets the field gets the model with NO lead —
> an aim that misses behind every moving target — rather than a perfect intercept solution.
>
> **Validation, because the failure modes moved from Python to YAML.** A rule that is data is a rule
> that can be silently under-specified, which is the "wrong for 300M steps and nobody notices" class.
> `validate()` now rejects: `fire_range_fraction` or `desired_range_fraction` outside [0, 1]; a
> lateral hold fraction paired with a 0 speed limit (the rule does nothing); and a kind with an
> `attack_range` but no `desired_range_fraction` (a KITE bot of that kind would hold range 0 and
> charge to point blank). That last check found a genuinely under-specified block in
> `tests/test_config.py`'s own inline brawlers fixture on its first run.
>
> **`tests/test_combat_rules.py` (21 tests) is deliberately kind-agnostic**: each test takes one kind,
> rewrites a single field, and asserts the behavior follows the FIELD rather than the `Kind`. The
> acceptance test the plan asked for is
> `test_a_kind_can_be_redefined_into_a_different_archetype_from_params_alone` — it turns BOT_MELEE
> (point-blank DIRECT cone, no LOS) into a long-range LOS-gated leading shooter with a range
> fraction, from params alone, and checks all four rule axes flip with it. Per-brawler behavior stays
> in `test_{sniper,artillery,melee,rifle}.py`, which now call the one rule; those four files were
> NOT consolidated, since what they cover is each brawler end to end, which still exists.

### Step E2 — Verify and re-baseline

**Files:** `configs/{default,train}.yaml`, `docs/`, `README.md`, `CHARACTER_DETAILS.md`

> **⚠ SCOPE CHANGE (user instruction, 2026-08-19).** *"I don't want game statistics to be
> rebalanced. The goal is for this agent to learn the brawl stars environment, so it must prioritize
> using real game statistics, even if that means some characters are better than others."*
>
> **`configs/brawlers.yaml` is therefore OUT of this step's scope**, and so is every per-brawler
> number in it — HP, damage, reload, cooldown, range, projectile speed. §1.4's flag on Buzz (D7's
> 1.0 s cooldown flips him from +34% to −33% sustained DPS) is now an OBSERVATION to record from the
> rollout, not a thing to fix: a five-hitscan sweep costing a full second of frozen reload is what
> the real character does, and Buzz being the weakest sustained-DPS bot in the roster is a fact
> about Brawl Stars, not a bug in the sim.
>
> What remains in scope is everything that is a SIM parameter rather than a game statistic: episode
> length, the observation/reward wiring, the benchmark baseline, and the docs. Where a value is
> currently NOT the real one, the fix is to make it real — not to tune it. `cubes` is the live
> example: `damage_bonus_per_cube` is already back at the real 0.10, and `hp_bonus_per_cube` is
> still 0.15 against a real +10%/cube, which is a question to settle rather than a knob to turn.

**Regen is fixed at the real rule by D12 and is not touched here.** The job is to MEASURE what the
finished roster actually does and to record it — fixing only things that are wrong about the sim,
not things that are merely lopsided about the game:

1. Instrument a 200-episode rollout: median time-to-kill, hero survival time, **fights abandoned
   before a kill**, time spent with nobody in anyone's range, and truncation rate.
2. **Verify Step B3's zone actually resolves disengagement** — it is the designed answer to it
   (D15/D16), so if fights still stall, the 0.20 fraction or the shrink schedule is what moves, not
   regen.
3. If truncation is still high, raise `sim.max_episode_steps` above 3000 — 150s may simply be short
   for a 10-player lobby at the new TTK. Max possible episode time is 180s. This is a SIM parameter,
   not a game statistic, so it is the correct lever.
   *(The zone itself is no longer an E2 question — D15 specified it and Step B3 implements it.)*
4. Settle `cubes.hp_bonus_per_cube` against the real +10%/cube (it is 0.15;
   `damage_bonus_per_cube` is already 0.10). A correctness question, not a balance one.
5. Re-run `scripts/benchmark.py` in full on an IDLE machine and record the new baseline against
   §1.3's table — including E1's saving, which was deliberately not measured at the time because
   the GPU was 63% busy with a foreign process (the A1/A2 retraction is why).
6. Update `CHARACTER_DETAILS.md`: move every implemented item out of its "unimplemented" sections.
7. Record per-brawler sustained and burst DPS as measured, so §1.4's post-D7 table stops being a
   calculation and becomes an observation.

**Acceptance:** median episode ends in a decision (kill or zone), not a truncation. Native throughput
at `n_envs=1024` is **≥ the 4,757 steps/sec baseline** in §1.3. The roster's measured numbers are
recorded, whatever they say — "character X is stronger than character Y" is a finding, not a failure.

> **✅ DONE.** Measured over **9,022 episodes** (512 envs × 1500 decisions, random hero, so every
> number describes the ENVIRONMENT and not a policy).
>
> | question | answer |
> |---|---|
> | do episodes resolve? | **100% terminated, 0% truncated** |
> | episode length | median **19.5 s**, p90 31 s, against a 150 s cap |
> | first death | median **5.0 s** |
> | deaths by cause | 76.2% combat, **23.8% zone** |
> | ticks with nobody engaged | **3.3%** |
>
> **§1.4's three open worries are all closed, and none needed a change.** Truncation is zero, so
> `sim.max_episode_steps: 3000` stays (it is ~5× longer than p90 even before a trained hero
> lengthens episodes). The zone is not decorative — it resolves nearly a quarter of all deaths,
> which is exactly the disengagement answer D15 designed it to be. And "nobody in anyone's range"
> is 3.3% of ticks, so the lobby is not drifting apart.
>
> **Measured per-brawler output.** §1.4's post-D7 table was arithmetic; this is observation. The
> units differ on purpose: §1.4 quoted damage per second of SUSTAINED FIRE, this is damage per
> second ALIVE, which includes all the time a bot has no target, is out of range, or is reloading.
>
> | | dmg/s alive | shots/s | dmg/shot | kills | deaths |
> |---|---|---|---|---|---|
> | Brock | **324** | 0.14 | 2375 | 13,068 | 9,298 |
> | Shelly | 263 | 0.21 | 1233 | 7,548 | 5,065 |
> | Grom | 262 | 0.12 | 2194 | 4,676 | 9,100 |
> | Buzz | 137 | 0.15 | 923 | 1,696 | **3,748** |
> | Mortis | 81 | 0.04 | 2044 | 600 | 9,012 |
>
> `dmg/shot` is the most informative column and it is not a config value — it is how much of each
> attack actually connects. Buzz lands **1.1 of his 5 hitscans** on average (923/840) and Shelly
> **2.05 of her 5 pellets** (1233/600), both close to what their geometry predicts.
>
> **Brock takes 47% of all kills** and Buzz survives more than twice as well as anyone else
> (3,748 deaths vs Brock's 9,298 on comparable population). **That is recorded, not fixed** — per
> the scope change above, an uneven roster is a fact about Brawl Stars to learn, not a bug.
>
> **Power cubes are now the real mechanic** (user-authorised): **+10% damage and a FLAT +400 max
> HP**, replacing a 0.15 FRACTION of base HP. The shape change matters more than the magnitude —
> as a fraction, cubes made the tankiest tankier (a 16-cube Buzz reached 34000 against Brock's
> 20400, a 1.67× spread); flat, everyone gains the same 6400 and the spread narrows to 1.32×.
> Late-episode HP roughly halves, and so does late-episode time-to-kill.
>
> `enemy_hp_mult` multiplies the TOTAL rather than just the base, matching `effective_damage`'s
> treatment of the cube damage bonus: at 16 cubes the flat bonus is larger than most base HPs, so
> leaving it outside the multiplier would let a cube-stacked enemy escape the one knob that says
> how tanky enemies are.
>
> **`config.check_removed_keys` is new**, because moving a mechanic's SHAPE into a renamed key is
> how a config silently resolves to 0 and disables something for a whole run. It rejects
> `cubes.hp_bonus_per_cube`, `zone.dps` and `regen.per_second` at `build_params` time, by name.
> It found a genuinely stale `regen.per_second: 400.0` in `tests/test_config.py`'s own inline
> fixture the first time it ran. Checked in `build_params` and NOT in `_resolve_all`, because
> `_resolve_all` also runs from `resample_params` — i.e. inside `step()` on every autoreset — and
> a Python dict walk has no business there.
>
> **`scripts/benchmark.py` was deliberately NOT re-run.** The user's instruction: *"many of your
> past speed measurements have been wildly inaccurate, they fail to account for 4000+ environments,
> training time, etc. I can manage resources on my own, but really want you to focus on anything
> that would risk runaway memory, storage, or other issues."* A synthetic steps/sec number at
> `n_envs=1024` is exactly the measurement that instruction deprioritises. The throughput figure
> recorded instead is the one from a **real training loop at the real `n_envs=4096`: 4,251 SB3
> timesteps/sec** during the rollout phase — same unit as §1.3's 4,757, but including the policy
> forward pass and the host transfer that the native benchmark excludes, and at 4× the batch. The
> two are not directly comparable and are not presented as if they were.
>
> **Resource audit** (the part that instruction DID ask for) is §6 below.

---

# 5. Files touched, by step

| file | A0 | A1 | A2 | A3 | B1 | B2 | B3 | C1 | C2 | C3 | C4 | D1 | D2 | E1 | E2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `constants.py` | | | | | ● | | | | | ● | | | ● | ● | |
| `config.py` | | | | | ● | ● | ● | | ● | ● | ● | ● | ● | ● | |
| `core/state.py` | | | | | | | | | | ● | | ● | ● | | |
| `core/terrain.py` | | ● | | | | | | | | | | | | | |
| `core/hero.py` | | | | | | | | ● | | | | ● | ● | | |
| `core/combat.py` | | ● | | | | | | | ● | | | | ● | | |
| `core/movement.py` | | | | | | | | | ● | | | | | | |
| `core/projectiles.py` | | | | | | ● | | | | ● | ● | | ● | | |
| `core/zone.py` | | | | | | | ● | | | | | | | | |
| `core/observation.py` | | | | | | | ● | | | ● | | ● | ● | | |
| `core/obs_schema.py` | | | | | ● | | ● | | | ● | | ● | ● | | |
| `core/obs_select.py` | | | | | | | | | | ● | | ● | ● | | |
| `bots/policy.py` | | ● | ● | | | | | | | | | | | ● | |
| `bots/perception.py` | | | ● | | | | | | | | | | | | |
| `bots/{sniper,artillery,melee,rifle}.py` | | | | | | | | | ● | | | | | ⊗ | |
| `bots/combat_rules.py` *(new)* | | | | | | | | | | | | | | ● | |
| `env.py` | | | | | | | | | ● | ● | | ● | ● | | |
| `wrappers/*` | | | | | | | | | | | | | ● | | |
| `render/*` | | | | | ● | | | | | ● | | | ● | | |
| `configs/brawlers.yaml` | | | | | ● | | | ● | ● | ● | ● | ● | ● | ● | |
| `configs/default.yaml` | | | ● | ● | ● | ● | ● | | | | | | ● | | ● |
| `configs/agent_obs.yaml` | | | | | | | | | | ● | | ● | ● | | |
| `CHARACTER_DETAILS.md` | | | | | ● | | | | | | | | | | ● |
| **tests** | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |

⊗ = deleted

**Note how much thinner C1 got.** After D7 it touches `hero.py` + `brawlers.yaml` + tests, and nothing
else — no `state.py`, no `env.py`, no obs. That is the single largest simplification your answers
produced.

**Test files by step** (the A0 deliverable, pre-filled from the audit).

> ⚠ **This map is a hint, not a gate. Run the FULL suite before calling any step done.** B1's row
> originally listed five files; the two failures it actually caused were in `test_observation.py`
> and `test_viewer.py`, neither of which was on it — one because a `kind_onehot` literal and one
> because a `_PROJ_COLOR` artist count both depended on the `Proj` enum's width from a distance.
> Anything derived from an enum, a config value, or a stat has a blast radius wider than the module
> that owns it. Targeted runs are for the fast inner loop only.

| step | test files |
|---|---|
| A1 | `test_terrain.py`, `test_combat.py` |
| A2 | `test_bot_dispatch.py`, `test_perception.py` |
| A3 | `test_obs_schema.py`, `test_configs_files.py` |
| B1 | `test_constants.py`, `test_config.py`, `test_configs_files.py`, `test_stats.py`, `test_obs_schema.py`, **`test_observation.py`**, **`test_viewer.py`**, **`test_projectiles.py`** |
| B2 | `test_projectiles.py`, `test_config.py` |
| B3 | `test_zone.py`, `test_config.py`, `test_configs_files.py`, `test_obs_schema.py` |
| C1 | `test_hero.py`, `test_action_repeat.py` |
| C2 | `test_combat.py`, `test_melee.py`, `test_movement.py` |
| C3 | `test_projectiles.py`, `test_obs_schema.py`, `test_obs_select.py`, `test_artillery.py` |
| C4 | `test_artillery.py`, `test_projectiles.py` |
| D1 | `test_hero.py`, `test_obs_schema.py` |
| D2 | `test_hero.py`, `test_projectiles.py`, `test_env.py`, `test_sb3_vecenv.py`, `test_gym_single.py`, `test_play_manual.py` |
| E1 | `test_bot_dispatch.py`, consolidate `test_{sniper,artillery,melee,rifle}.py` |
| E2 | `test_training.py`, `test_benchmark.py`, `test_integration.py` |

---

# 6. Resource audit (Step E2)

Measured at `configs/train.yaml` as shipped: `n_envs: 4096`, `ppo.n_steps: 128`,
`ppo.batch_size: 8192`, `total_timesteps: 300M`, eval on. **Every figure here is PROCESS-SCOPED**
— see §1.3's caveat on why `nvidia-smi` is the wrong instrument for this.

### VRAM — the binding knob is `batch_size`, not `n_envs`

| configuration | peak allocated | peak reserved | of 15.9 GiB |
|---|---|---|---|
| **batch 8192, eval on (shipped)** | 4.62 GB | **8.82 GB** | **55%** |
| batch 8192, eval off | 4.54 GB | 8.78 GB | 55% |
| batch 2048, eval off | 2.31 GB | 3.44 GB | 22% |

The simulator is not what fills the GPU. 4096 envs of `SimState` are **2.06 GB**, and the 1,200
eval envs `TierEvaluator` holds for the whole run add **~0.05 GB** at build time and ~0.08 GB at
the peak — they are dwarfed by the PPO update, which is where the peak lives: `batch_size` rows of
the `10×20×40` grid widened to float32 and held as CNN activations through backward. Dropping
`batch_size` 8192 → 2048 cuts reserved VRAM **2.6×** for identical total work (4× the minibatches
over the same 524,288 transitions). That is the knob to reach for if the GPU is ever shared.

### Host RAM — the SB3 rollout buffer, and where the ceiling actually is

Exact, from summing `arr.nbytes` over a real `MaskableDictRolloutBuffer`:

| array | dtype | GB at n_envs=4096 |
|---|---|---|
| `obs[grid]` | uint8 | **3.906** |
| `obs[projectiles]` | float32 | 0.398 |
| `obs[enemies]` | float32 | 0.299 |
| `obs[self]` + `zone` + `action_masks` + PPO scalars | float32 | 0.121 |
| **total** | | **4.72 GB steady** |

`RolloutBuffer.swap_and_flatten` does **not** share memory with the array it reshapes (verified
with `np.shares_memory`), so PPO's `get()` materialises a second copy of each obs group in turn:
**peak ≈ 4.72 + 3.91 = 8.63 GB** during each update. On a 31.1 GB machine that is comfortable, but
it scales linearly with `n_envs` and **83% of it is the egocentric grid** — so host RAM, not VRAM,
is what sets the practical ceiling on `n_envs`, at roughly 6000 on this box.

### Disk — settled, and the wiring is correct

One checkpoint is **5.3 MB** (1,360,149-parameter policy plus optimizer state).
`scripts/train.py:213` computes `save_freq = checkpoint_every_steps // n_envs`, which over 300M
steps produces **75 checkpoints = 0.39 GB**, plus 75 few-KB VecNormalize pickles and two more
model zips. The incident this guards against — a checkpoint every 200,000 steps — would be 1,500
files. No video or frame output is written during training; `progress.csv`/`log.txt`/TensorBoard
events cover ~572 rollouts and are negligible.

### No unbounded growth anywhere

Every per-episode accumulator in `training/callbacks.py` is a `deque(maxlen=...)`. The two plain
lists are `CurriculumCallback.history` (appends only on a stage transition, so at most a handful
over a run) and `TierEvalCallback.history` (one entry per eval, 60 over 300M steps). SB3's own
`ep_info_buffer` is `maxlen=100`. Nothing accumulates per step or per episode without a bound.

### Verified runnable

`python scripts/train.py --smoke` and the real config at `n_envs=4096` with eval enabled both
complete cleanly (exit 0), writing checkpoints, `final_model.zip` and `curriculum.json`. Rollout
throughput at 4096 envs is **4,251 SB3 timesteps/sec**.

---

# 7. Risks and deliberate non-goals

**Risk — the cooldown floor is exactly one agent decision.** D7's 0.25s minimum equals
`cfg.agent_dt` (`dt 0.05 × action_repeat 5`). That is deliberate — every brawler gets at least one
agent decision between its own shots — but it makes the existing strict `cooldown > agent_dt`
assertion false at the boundary, and it means *any* future reduction in `action_repeat` silently
changes what the floor means relative to the decision rate. If `action_repeat` ever moves, revisit the
floor with it.

**Risk — Buzz's 1.0s cooldown is the plan's least certain number.** It follows directly from D7
(the sweep is the cooldown) but costs him half his sustained DPS, flipping him from the roster's only
gainer to a 33% loser (§1.4). Explicitly scheduled for review in Step E2.

**Risk — reload-pause compounds with low ammo.** Sustained fire is now
`attack_cooldown + reload_seconds` per shot. For Buzz that is 2.0s between sweeps; for Grom 2.55s.
An entity that empties its 3-round clip is out of the fight for ~6-7s. Watch for bots that spend most
of an episode dry.

**Risk — Phase 2/3 change what the agent sees.** Steps C3/D1/D2 each widen the obs. Per D11 this
costs nothing in checkpoints, but it does mean **the agent cannot be meaningfully trained until D2
lands** — every intermediate obs shape is throwaway. Plan one real training run at the end of Phase 3,
not several.

**Risk — Step B2's projectile budget, compounded by D8.** Hazards live in the projectile buffer and a
Brock sphere holds a slot for **4 seconds** versus a bullet's ~1.5. Nine Brocks hold roughly 18 slots
of standing occupancy before a single pellet is counted. D14 permits raising `P`, but take the
smallest rung the measurement supports (`128 → 192 → 256`), and measure *after* Step C3. If the answer
really is 256, that is the entire Phase 0 saving spent — at which point the better mitigation is a
per-kind `MAX_PROJ_PER_ENTITY` in `alloc_slots` rather than one global constant, plus a separate cap
on hazard slots so a Brock-heavy lobby cannot starve everyone else's bullets.

**Not doing — Gadgets, Star Powers, Hypercharges.** Still out of scope (BRAWL_SIM_BUILD_PLAN.md
Appendix A item 1). The Super is in scope only because CHARACTER_DETAILS specifies one.

**Not doing — pathfinding.** Bots still steer, they do not path. Buzz's 2.67-tile range makes him
much more dependent on actually reaching a target than the old 3.0-tile El Primo was; if he proves
unable to close, the fix is a steering-weight change in `personality.py`, not A*.

**Not doing — `torch.compile`.** Still blocked on this machine (no MSVC `cl.exe`, no working Triton on
Windows + Blackwell). None of the above depends on it.
