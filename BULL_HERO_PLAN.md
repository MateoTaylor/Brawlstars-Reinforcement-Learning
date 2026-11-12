# Bull as the agent: plan

**Status:** Phases 1–5 BUILT 2026-10-23: the hero seam, the aimed volley, the status effects, the
charge super, and Stomper with the Berserker star power. Phase 6a and 6b DONE 2026-10-27: Bull's
spec, `configs/agent_obs_bull.yaml` (D6), and the CPU pre-training check, which passed (§5.6).
**The Bull run (6c) FINISHED 2026-10-28:** `runs/mortis_ppo-20261027-160804`, 400M, best = final.
It scored six-tier .825 and holdout .813 in the sim, against .424 and .391 for Mortis deploy6
(§5.6c). M1, the attack alone, needs `super_charge_hits: 0`, `gadget_cooldown: 0` and
`reload_boost_below_hp_frac: 0` in `hero_bull`. **7a and 7b BUILT 2026-10-28:** the loop takes
the hero from the run, and the shadow models Bull's kit (the volley, the charge at full length,
Berserker, the 20 s Stomper, `hero.slow_window`), so a Bull run can deploy. Phase 0 (f), the
reload, was measured the same day at 1.62 s, and you set `hero_bull.reload_seconds` to 1.6
(§5.0). Phase 0 (a, b, d, e) was measured 2026-11-01, and you set `hero_bull`'s range 5.2, pellet
speed 9.0 and push 2.5 (§5.0), then `bot_bull`'s range and pellet speed to match (D7). Later the
same day you set both blocks' walk to the measured 2.68 (D7 **DECIDED**), the stomp's wind-up to
0.30 s and its stun to 1.4 s, and kept the 2.67-tile radius (§5.0). A fresh 300M run on these
numbers is yours (§5.6d). **7d SETTLED 2026-11-01** on your real-match clip: the gate holds
through the lit hypercharge, so no gate change was needed, and a footage test pins it (§5.7d).
7c and 7e (live checks) are not started. Decisions marked **(open)** are
yours. The others are recommendations you can overturn.
**Mortis digest:** `48e1a2b59a31e97d` since `bot_bull` took the measured walk (2026-11-01, your
call), and `5b469682fe6833dc` after its measured range and pellet speed earlier that day. Both moved
it on purpose through every bot Bull. Before that it was
`247884ee3f0cf382`, from the 2026-10-24 attack-cooldown snap (§5.5, your call), and before that
`5df44a1488f7da5e`, from the 2026-10-23 zero-width cone fix (§5.3); Phases 4 and 5 left it
unchanged. Later phases compare against it. Bull's digest is `21c617decff340fb` since
`attack_miss` went to −0.10 (2026-11-03), and was `b8b6b69df80c322d` at −0.05; the miss
penalties move it through the reward alone (§5.8). With them zeroed it is `5d4250ff5c7ac420`, from both walks and the stomp's wind-up and stun. It was `285f0ed3e35482ee` after the bot's range and pellet speed,
and `82bc81256f947e00` after the hero's range, pellet speed and push (earlier the
same day), `367cb1d78c40591e` from his 1.6 s reload (2026-10-28), and `483861f81de74216` from the
snap before that. With the old walk, wind-up and stun put back through `run.env_overrides`, both digests
come back bit for bit, so nothing else moved. The 6c run trained at 2.0 s, 7.0 tiles, 6.4 tiles/s,
a 1-tile push, a 2.57 walk and a 0.25 s / 1.0 s Stomper, against bot Bulls at 7.0, 6.4 and 2.17.
**Scope:** train and deploy **Bull** as the hero instead of Mortis, across `brawl_sim`, training and
`brawl_deployment`. Mortis stays trainable and deployable, and his behaviour stays bit-identical.
**Template:** archive/EDGAR_HERO_PLAN.md (2026-09-14, never built) mapped the same seams for Edgar.
Its `file:line` references are stale; this plan re-checked the ones it reuses on 2026-10-23.

---

## 0. TL;DR

**Already built:** Bull's attack. `bot_bull` fires 5 × 880 pellets in a 0.5 rad fan (`BULL_SLUG`,
6.4 tiles/s). That machinery is per-kind, so it works from slot 0. The hero's attack path already
routes a kind with `dash_distance: 0` to `spawn_volley` ([env.py:650](brawl_sim/env.py:650)).

**New work, in rough order of size:**

1. **Status effects** (**built**, Phase 3). Slow, stun and airborne (untargetable and stunned) are
   per-entity timers with their gates. The charge applies airborne and the landing slow (Phase 4),
   and Stomper the slow or the stun (Phase 5). Knockback is a displacement, built with the charge.
   [BRAWL_SIM_DESIGN.md](BRAWL_SIM_DESIGN.md) §2's knockback divergence now says so.
2. **The super is a long charge** (**built**, Phase 4). It is 11 tiles at 6.7 tiles/s, so Bull is
   busy for about 1.8 s (7 decisions). It hits every enemy on the path for 2000, knocks them back
   and launches them, then slows everyone within 3 tiles of the landing. It rides Mortis's dash
   state (`hero.start_charge`, then `advance_dash`), and the first wall stops it (D3).
3. **Stomper** (**built**, Phase 5) is a self-centred pulse 0.30 s after the tap. It slows every
   other player within 2.67 tiles by half for 1 s, or stuns for 1.4 s one that is already slowed. It
   is a second gadget type next to Mortis's spinner (`gadget_type: STOMP`).
4. **The hero identity seam** (**built**, Phase 1). **Row 0 is "the hero"**, filled from the
   brawlers.yaml block the run names. That means no new `Kind`, no obs width change, the four "hero
   is Kind 0" spots stay correct, and the Mortis deploy6 run keeps loading (D1).
5. **Aiming the hero's shotgun** (**built**, Phase 2). Value 1 aims along the move bin, and value 4
   auto-aims like a bare tap.
6. **Star power: Berserker** (**built**, Phase 5; your pick). Below 60 % of his max HP he reloads
   twice as fast.
7. **Deployment.** The shadow's kit is Mortis-shaped (`ShadowParams` requires dash fields), the hero
   name is a constant (`loop.HERO_KIND`), and the **calibration gate anchors on the inert
   hypercharge disc. Your Bull has a hypercharge equipped**, so that anchor breaks (§5.7).

**Early signal:** milestone **M1** is a Bull with only his attack, no super and no gadget. It needs
only the seam and the aim (Phases 1 and 2), and it answers "can this agent play a slow-reloading
shotgun tank in this sim?" while the super is being built.

---

## 1. What the reference clip shows

**Clip:** `C:\Users\mateo\Videos\2026-10-23 11-49-15.mp4`. It is an OBS capture of BlueStacks at
2560×1440, 30 fps, 589 frames, recorded in Training Grounds.

**Method.** The Training Grounds camera does not move: the whole room is in view and never scrolls.
So screen positions convert to tiles through the shipped ground homography
(`brawl_vision/data/homography.json`, scaled 1126/1440 to its viewport), with no odometry needed.
Two checks support the scale:
- The top wall's block pitch reads 96.7 px, which matches the model at that row.
- The super's aim circle comes out round under the model (2.95 / 2.92 / 3.03 / 2.81 tiles on its
  four sides).

Bull's position comes from tracking his green HP bar, which is rigid with his body, frame by frame.
Scratch scripts are in the session scratchpad; none were added to the repo.

| Quantity | Reading | Frames | Confidence |
|---|---|---|---|
| Super aim held | The indicator is on screen from f45 to f221 and gone at f222 | f45–f222 | high |
| **Super wind-up** | Release (f222) to first motion (f228): **0.20 s**. He turns and lowers his head. | f222–f228 | ±1 frame |
| **Charge speed** | **6.71 tiles/s**, constant: every frame step is 0.21–0.24 tiles, with no ramp up or down | f228–f275 | high |
| **Charge length** | **11.02 tiles** in **1.63 s** (13.17 → 2.15) | f227–f276 | high, ±0.2 tiles |
| Where it ends | At the **centre of the aim circle** (2.14 against a landing at 2.15). The indicator's rectangle runs 1 tile past the end, which is its half-width. | f120 vs f276 | high |
| Path width on the indicator | 1.98 tiles wide, so **half-width 0.99** | f120 | medium: an indicator's width is not necessarily the hitbox |
| **Landing slow radius** | Aim circle **2.95**; landing pulse **3.0** (2.9–3.1) | f120, f280 | high |
| Landing pulse | Appears within 0.13 s of stopping; the crack decal fades by about f300 | f276–f300 | medium |
| **Stomper timing** | Button pressed f375–f376, released f377, stomp flash f384: **0.23–0.30 s** from tap to stomp | f375–f384 | medium |
| **Stomper radius** | Bright ring **2.4 tiles**, outer glow **3.1** | f388 | low: which one is the hitbox? |
| Stomper cooldown | **20 s**, as the card says. The button counts 20 → 16 over f384–f525. | f384–f525 | high |
| Attacks | Three shots (f434, f454, f476, 0.67–0.73 s apart). Each landed **one** 880 pellet on the top-left dummy (4000 → 3120 → 2240 → 1360). | f430–f510 | see Q9 |

**Not in this clip:**
- walking speed (Bull never walks);
- reload;
- attack range;
- any super hit on a dummy, so the 2000 damage, knockback and airborne time are your numbers, not
  measurements;
- every slow or stun duration and strength (dummies do not move);
- wall and water behaviour;
- how many hits charge the super.

One thing I could not identify: cyan swirls at Bull's feet over f0–f30 and f284–~f425. Is that a
skin effect?

**Card (your screenshots):**
- HP 10000, movement **FAST**.
- Attack: 5 × 880 at **NORMAL** range, reload **NORMAL**.
- Super: 2000 damage at **VERY LONG** range.
- Stomper: 20 s cooldown.
- Loadout: Stomper, a damage gear (fist), a star power (skull with "!!"), a health gear (heart), and
  a **hypercharge 1/1**.

---

## 2. Gap analysis

| Capability | Where | State for hero Bull |
|---|---|---|
| 5-pellet volley, cube scaling, super charge from hits | `projectiles.spawn_volley`, `stats.effective_damage`, `hero.add_super_charge` | ✅ per-kind |
| Hero ranged attack path | [env.py:533-534](brawl_sim/env.py:533) `is_other_attack = fire & ~(dash_distance > 0)` | ✅ |
| Hero volley aim | `env._attack_phase` overrides the hero's `aim_dir` row: move bin, or the auto-aim target | ✅ built (Phase 2) |
| Auto-aim target | `hero.auto_aim_target` reads `hero.weapon_reach`: the dash for Mortis, the volley for Bull | ✅ built (Phase 2) |
| Hero identity | `stats._HERO_KIND`, `spawn` slot 0 = zeros, curriculum column 0, [config.py:1203](brawl_sim/config.py:1203) | ✅ built (Phase 1, row swap) |
| Super as a charge | `super_type: CHARGE` fires `hero.start_charge`; `spawn_supers` launches only bolts | ✅ built (Phase 4) |
| Long busy window | Every gate reads `dash_t <= 0` | ✅ the charge rides the dash timer (D2) |
| Slow / stun / airborne | `core/status.py`: per-entity timers and their gates | ✅ built (Phase 3); the charge applies airborne and the slow (Phase 4) |
| Knockback | `hero.advance_dash`, on each new charge hit | ✅ built (Phase 4) |
| Stomper | `gadget_type: STOMP`: `hero.start_stomp`, landed by `env._stomp_phase` (`hero.advance_stomp`) | ✅ built (Phase 5) |
| Star power | Berserker, in `hero.tick_timers` (`reload_boost_*`) | ✅ built (Phase 5) |
| Obs | `configs/agent_obs_bull.yaml`: deploy6 minus `long_dash_*`, plus `hero.slow_window` and three critic-only status timers | ✅ built (Phase 6a, D6) |
| Shadow kit | `ShadowParams` loads any kit by name; `ShadowHero` models the dash, the volley, the charge, Berserker and `slow_window` | ✅ built (7b, 2026-10-28) |
| Deployment hero | `DeployLoop.hero_kind` from the run's `entities.hero`; `attack_reach_tiles` / `auto_aim_reach_tiles` call the sim's own reach functions; `SHADOW_KINDS` holds both heroes | ✅ built (7a + 7b, 2026-10-28) |
| Calibration gate | anchored on the hypercharge disc | ✅ the disc lights up when charged, and the gate holds through it on a real match (§5.7d, 2026-11-01) |
| Odometry envelope | `DASH_TILES_PER_SECOND = 17.8` | ✅ 6.7 tiles/s fits inside it |

---

## 3. Decisions

### D1: How the sim knows who the hero is: **DECIDED 2026-10-23, row swap (option A)**

| Option | For | Against |
|---|---|---|
| **A. Row swap (recommended).** `Kind` value 0 becomes "the hero". `entities.hero: bull` makes the per-kind loader read row 0 from `hero_bull` instead of `hero_mortis`. | <ul><li>`N_KINDS` stays 8, so `entities.kind_onehot` keeps its width. The deploy6 **critic** group carries that onehot, so the served Mortis run keeps loading.</li><li>`stats.is_hero`, `spawn`, the curriculum's column 0 and `validate`'s exclusion are already right.</li><li>Bot Bull (kind 7) keeps his own row, so tiers never touch the hero.</li></ul> | <ul><li>One hero per run. That is how runs work anyway.</li><li>`KIND_YAML_NAMES` becomes a function of config, which is read in about 8 places in `config.py`.</li><li>The viewer's sprite for kind 0 must come from config instead of the member name.</li><li>Rename `HERO_MORTIS` to `HERO`. The value stays 0 forever.</li></ul> |
| B. New `Kind.HERO_BULL = 8` (the Edgar plan's choice) | Both heroes can exist in one sim. | <ul><li>`kind_onehot` goes from 8 to 9, which **breaks the served deploy6 Mortis checkpoint** through its critic group.</li><li>Four silent Kind-0 hardcodes have to be fixed (adding-a-brawler checklist).</li><li>More tests to touch.</li></ul> |

The Edgar plan rejected "put the *bot* kind in slot 0", because hero and bot would then share a stat
row. Option A is different: the hero keeps row 0, and only the YAML block that fills it changes.

### D2: Where the charge lives in state: **the dash timer, in a super mode, BUILT 2026-10-23 (§5.4)**

Bull has no attack dash, so `ent_dash_t / dir / speed / hits` are free for his super. A per-entity
`ent_dash_super` (bool) marks a charge, and a per-kind `super_type` enum says which super a kind
has (`BOLT` = Mortis, `CHARGE` = Bull). When `super_type == CHARGE`, a super fire calls
`hero.start_charge` with the charge's numbers:

| Field | Value | Source |
|---|---|---|
| `super_range` | 11.0 | measured. This plan called it `super_charge_distance`; the charge reuses the bolt's field |
| `super_charge_speed` | 6.7 | measured |
| `super_windup_seconds` | 0.20 | measured |
| `super_radius` | 1.0 | the indicator's half-width (provisional) |
| `super_damage` | 2000 | the card |

Under `ent_dash_super`, `advance_dash`:
- deals `super_damage` instead of the weapon's damage;
- applies knockback and airborne on each new hit;
- stops at the first wall or water, as the dash does (D3);
- on its last tick, applies the landing slow pulse.

The wind-up is the first stretch of `dash_t`, counted down in its own `ent_dash_windup_t`, and he
stands still through it.

What this buys:
- every `dash_t <= 0` gate already masks the action column for the whole charge;
- `hero.dashing`, `hero.dash_t` and `hero.dash_dir` already mean "charging" in the obs;
- the shadow's dash branch can be reused.

`check_invariants` now bounds `dash_t` by the larger of `dash_duration` and the charge's duration.
It also checks that the flag and the wind-up exist only inside a running dash.

The alternative was separate `ent_charge_*` tensors, as the Edgar plan's D3 did for the jump. That
plan chose separate state because a jump must *not* deal capsule damage. A charge must, so reusing
the dash fits here.

### D3: Walls under the charge: **DECIDED 2026-10-23 (you), C: stop at the first wall**

The card says he "bulldozes through obstacles". In the real game the charge **destroys** walls and
bushes along its path.

| Option | For | Against |
|---|---|---|
| A. Pass through, destroy nothing. Snap the landing to the last point clear of walls and water (the Edgar plan's sampled snap), so he never ends inside a wall. | Cheap, and matches the shared `MapBank` design. | Cover he "broke" is still there for the rest of the match. |
| B. Per-env destructible walls | Faithful. | Needs a per-env (N,H,W) terrain overlay, which §3's vectorization rules forbid. Every LOS, march, nav field and obs grid would read through it. Large. |
| **C. Stop at the first wall, like Mortis's dash (your choice).** | Trivial. | Wrong: it removes the super's escape and engage value through cover. |

**Your call:** "Stop at the first wall". As with Mortis's dash, nothing is passed through or
destroyed, and water stops him too. The "Against" cell stands as an accepted divergence. Live, his
charge goes through the walls that stop the sim's (§6), and BRAWL_SIM_DESIGN.md §2 lists it.

### D4: Status effects: **generic per-entity timers, BUILT 2026-10-23 (§5.3)**

New SimState fields, all (N,E) and zeroed on reset. Under the "Per-kind data, generic code" rule
they cost a masked `torch.where` where unused.

| Field | Meaning | Gates |
|---|---|---|
| `ent_slow_t`, `ent_slow_frac` | Speed × (1 − frac) while `slow_t > 0` | `movement.apply_movement` |
| `ent_stun_t` | No move, no attack, no super, no gadget | movement, the attack phase's `can_*`, the mask |
| `ent_airborne_t` | Stunned, **and untargetable**: projectiles pass through, and melee, dash and charge skip it | the stun gates, plus `step_projectiles`' unit test, `melee_hitscan` and `advance_dash` |

**As built.** These rules were my calls while building; overturn any of them:
- **Durations are whole ticks.** The countdown snaps to 0 once less than half a tick is left.
  Plain float32 subtraction holds 0.25, 0.3, 1.5, 2.0, 2.5 and 3.0 s one tick too long. An effect
  of k ticks applied during tick t holds through tick t + k − 1 and is gone on tick t + k.
- **Airborne hides a unit from more than the table lists:**
  - blasts (artillery and Mortis's spinner);
  - the hero's auto-aim and his super tap;
  - the reward's in-reach flag.

  The gas still burns it.
- **Bots don't know.** Their intent is computed as usual and then gated. They still aim at an
  airborne target, and the shot flies through it.
- **What a stun leaves alone:**
  - It does not pause the reload or the cooldowns.
  - A dash already running finishes. No dasher can be stunned yet.
  - A melee sub-swing inside the stun is dropped, but later ones land on schedule.
- **A stunned unit does not turn,** and the slow touches walking only.
- **Mortis's spinner** (`gadget_target`) still picks an airborne enemy. Its blast then skips that
  enemy.

**Knockback** is a one-tick displacement. It is clipped by `terrain.body_travel`, so walls and water
stop it, and it is applied when the charge hits. Its direction is **DECIDED 2026-10-23 (you):**
1 tile straight away from Bull's centre at the moment of contact. A head-on hit goes along the
charge, and a glancing one goes off at an angle. Built in Phase 4 (§5.4).

**Stomper's rule, as built (Phase 5, §5.5):** a player with `slow_t > 0`, from any source, is
stunned for `stomp_stun_seconds` and keeps its slow. Any other is slowed for `stomp_slow_seconds`.
The check reads the slows from before the pulse, so a stomp can never stun with its own slow.
Being airborne counts as neither: the pulse skips an airborne player.

Only the hero applies these effects; no bot does. The machinery is per-entity, though, so a bot Bull
with a super later is data plus a fire decision (§1 #21).

### D5: The hero's shotgun aim (recommended: mirror Mortis's attack values)

- **Value 1** (aimed drag) fires along the move bin, or along facing when the bin is idle.
- **Value 4** (bare tap, under `action.auto_aim`) fires at the nearest alive enemy within
  `attack_range` plus both bodies; crates: Q8.
- Live, value 1 is already a drag along the bin and value 4 a tap (`buttons.py`), so nothing changes
  live.
- Build it as an `aim_dir[:, 0]` override in `_bot_phase` / `_attack_phase`. `spawn_volley` then
  fans the pellets around it. **Built 2026-10-23 in `_attack_phase` (§5.2).**

### D6: Observation spec: **a new `configs/agent_obs_bull.yaml`, trained from scratch, BUILT 2026-10-27 (§5.6)**

Start from deploy6, then:

- **Drop** `hero.long_dash_ready` and `hero.long_dash_frac`. They are always 0 for Bull.
- **Keep** `hero.dashing`, `hero.dash_t` and `hero.dash_dir`. Under D2 they mean charging.
  `hero.dash_t` is in seconds, which `normalize: true` passes through unscaled. A charge's timer
  therefore starts at 1.84, where Mortis's dash started at 0.3, and a decision reads at most 1.59
  (measured in Phase 6b, §5.6). That is still a sane input, so it needs no divisor.
- **Add** `hero.slow_window`: seconds left on the slow his last landing applied, 0 if none. This is
  the whole Stomper decision ("will this stomp stun?"). It is shadow-derivable live, because the
  shadow knows when it fired the super, and it passes the low-info rule.
- **Critic only** (privileged, zero-filled live): `entities.slowed_t`, `entities.stunned_t`,
  `entities.airborne_t`. A per-enemy "slowed" flag is not something CV can read, so the policy does
  not get it.

The rule in force applies: "a change to the obs or action width means a new spec file and a train
from scratch". A warm start from Mortis would not help anyway, because value 1 means a different
action for Bull.

### D7: Whose numbers **DECIDED 2026-11-01 (you): both blocks take the measurements**

The card says **FAST** (770/300 = **2.57 tiles/s**), and `bot_bull` ran at **2.17** (SLOW 650),
which you confirmed in August. Phase 0 (a) measured **2.68**, and **you set both blocks to it**
("follow the hero"). That changes the roster Mortis trains against (§6).

The card's range is NORMAL, and `bot_bull` used 7.0 tiles. Phase 0 (b) measured 5.2 off the aim
indicator and 9.0 tiles/s for the pellets. **You set both blocks to them** (2026-11-01). `bot_bull`
keeps its quoted 2.0 s reload, the one number the two blocks no longer share.

### D8: Where deployment learns the hero (recommended: from the run, as the Edgar plan's D6)

The run's env config carries `entities.hero`:
- `loop.HERO_KIND` reads it;
- `ShadowParams.load(kind)` loads that block;
- `brawl_deployment/data/kit_timing.json` becomes per-hero (`kit_timing_<hero>.json`), so
  `measure_reload --write` cannot overwrite Mortis's file.

`deployment.yaml` gets no hero key, so it cannot disagree with the checkpoint.

### D9: Reward for the gadget: **DECIDED 2026-10-23 (you), any slow or stun**

`reward.gadget_hit` (0.3) pays "a landed gadget that hurt a player". Stomper deals no damage, so the
term would never fire. The options were:
- **pay per stomp that slows or stuns at least one player (your choice);**
- pay stuns only;
- drop the term for Bull.

Built in Phase 5 (§5.5c): one payment per stomp, however many players it catches, as a spinner
landing pays once however many its blast hurts.

The super needs no new term. Its damage, kills and positioning already pay through the existing
rewards.

---

## 4. Questions for you

| # | Question | Why it matters |
|---|---|---|
| Q1 | Which star power: **Berserker** (reload faster below some HP) or **Tough Guy** (damage reduction below some HP)? Your icon reads like Berserker to me. Which numbers does Nulls Brawl show? | **Answered 2026-10-23:** Berserker. Below 60 % of his max HP he reloads twice as fast (§5.5b). The shadow's reload model now depends on HP (§5.7b). |
| Q2 | Super landing slow: how strong, how long, and is it a one-off pulse or a lingering area? | **Answered 2026-10-23:** one pulse at the landing, 40 % for 1.5 s. Both numbers are placeholders, since dummies cannot show them. **2026-11-01:** you kept both; the ~3.1 s icon in the 11-01 takes had a stun inside it (§5.0 d). |
| Q3 | Stomper: slow strength and duration, stun duration, and does it deal damage? | **Answered 2026-10-23:** the current wiki's numbers. A 2.67-tile radius, a 50 % slow for 1 s, or a 1 s stun on a player already slowed, and no damage (§5.5a). **2026-11-01:** you set the measured 1.4 s stun and 0.30 s wind-up, and kept the radius (§5.0 d, e, h). |
| Q4 | Knockback direction: away from the path, or pushed along with him? Does it stop at walls? | **Answered 2026-10-23:** 1 tile straight away from his centre at contact. Walls and water stop it (D4). |
| Q5 | Can Bull attack, or use Stomper, during the charge? Can he be hit during it? | **Answered 2026-10-23:** no. The charge holds the attack, the super and the gadget closed through its wind-up and rush (§5.5a). He can be hit throughout: no i-frames, Phase 4's call (§5.4). |
| Q6 | Is the charge always 11 tiles, or does a short drag shorten it? | **Answered 2026-10-28:** a short drag shortens it, so "assume it's always max length and drag off screen". The sim keeps 11 tiles; live, the super drags to the screen edge along the bin, and the shadow models every charge at full length (§5.7b). |
| Q7 | Gears: model the damage gear and health gear, or leave them out like Mortis's? | **Answered 2026-11-01:** leave them out. The sim models no gears, for either hero; the live Bull keeps his. |
| Q8 | Should the auto-aimed tap target crates when no enemy is in range (Mortis's does)? Built as Mortis's for now (§5.2). | D5 |
| Q9 | Were the three shots at f434–f476 tapped at the top-left dummy? Each landed one of five pellets at about 2.5 tiles. | **Settled 2026-11-01 by measurement:** the pellets' own fan opens 0.50 rad (Phase 0 b), so the 0.5 rad fan stands whichever way those shots were aimed. |
| Q10 | Can the hypercharge be unequipped in Nulls Brawl? | **Answered 2026-10-28:** no. It works the same for Mortis: it builds, then is used later. The gate fix is §5.7d. |

You decided D3 and D9 on 2026-10-23, and D7 on 2026-11-01.

---

## 5. Steps

Each step lists the files, the change, and a **done-when** check. After each phase, run
`graphify update .` (the project's CLAUDE.md asks for it after code changes).

### 5.0 Phase 0: ground truth (recordings; runs alongside Phases 1–2)

Training Grounds is **valid** for this kind of measurement, because its static camera makes screen
positions world positions. It stays invalid for taps, odometry and match-over. Record at 2560×1440,
the same as this clip.

- [x] **(a) Walk.** **Measured 2026-11-01** (`Videos/2026-11-01 09-57-42.mp4`, 30 fps, every
  distance through the corrected mapping below): three straight runs read 2.667-2.694 tiles/s, so
  **2.68** (the card's 2.57 is 4 % slow; Mortis runs 2.73). **You set 2.68 in `hero_bull` and
  `bot_bull`** (2026-11-01, D7).
- [x] **(b) Attack aim.** **Measured 2026-11-01:** the indicator reaches **5.2 tiles** (5.17-5.21
  left, down-left and right), against 7.0. The pellets' own fan opens 0.51 tiles per tile, so
  **0.50 rad stands** (the drawn cone is wider, ~0.71 rad); that settles Q9's worry directly. The
  pellets fly **~9 tiles/s** (8.8-9.0 on a sideways shot; the sim has 6.4) and vanish at 5.2-5.5.
  - **You set `attack_range` 5.2 and `proj_speed` 9.0 in `hero_bull`** (2026-11-01), then in
    `bot_bull` too (D7's weapon half). A kiting bot Bull now holds 3.1 tiles, not 4.2. The in-reach radius falls 5.61 → **5.42**, since the side
    pellet's tip now binds before it passes abeam, and auto-aim reach 7.30 → **5.50** (+0.40 to a
    centre).
  - A pellet's last tick sweeps its whole 0.45 tiles, so 12 ticks carry it **5.4 tiles**, inside
    the 5.2-5.5 at which the game's pellets vanish. Three still land out to 5.55 and the middle
    one to 6.1; the 5.42 flag errs short. At 6.4 tiles/s the overshoot was 0.04.
  - The shipped homography leans in this room: it reads 102 px/tile at x 400 and 96 at x 2100 on
    one row, where the real block seams are uniform. Measurements use a yaw-free row mapping scaled
    by the wall's 101 px/tile, which takes about 2 % off the model-only numbers (the 10-23 charge
    read 11.02 tiles that way and reads 10.78 corrected).
- [x] **(c) Super geometry.** **Answered by you 2026-10-28** (Q6): a short drag shortens the
  charge, and live play always drags to full length. No take needed.
- [x] **(d) Super into a dummy, then Stomper on it within about 1 s.** **Measured 2026-11-01**
  (charge 2: rush f2918-f2965, stomp pulse f2980):
  - The charge hit for 2000, as built. It pushed the dummy **2.55 tiles** (not 1.0), within 4° of
    straight away from his centre at contact (the D4 direction holds), over about 0.4 s. The dummy
    stopped about 0.2 tiles short of a wall, so the wall may have cut the push short. **You set
    `super_knockback_tiles` 2.5** (2026-11-01). The sim's push is still instant (the game's takes
    ~0.4 s, inside the 0.5 s airborne window, where the victim is untargetable either way).
  - The landing pulse slowed the dummy while it was still flying, as the sim's "airborne or not"
    does. The stomp then showed a stun sparkle for about 1.4 s (the sim had 1.0). **You set
    `stomp_stun_seconds` 1.4** (2026-11-01): 28 ticks.
  - The slow arrows stayed up about 3.1 s from the landing pulse, in both takes (sim 1.5). Both
    takes had a stun inside them, so this can't separate a 3 s slow from a 1.5 s slow paused by
    the stun. Settling it needs one landing next to a dummy with no stomp. **You kept 1.5 s**
    (2026-11-01), so no take is planned.
  - Path width: charge 1 passed a dummy 1.33 tiles off its line untouched, and charge 2 hit one
    0.94 off, so `super_radius` 1.0 stands. Length 10.78 tiles, speed 6.58 tiles/s, wind-up
    0.17-0.20 s, landing circle 2.95: all within 2 % of the built values.
- [x] **(e) Charge into a dummy while pressing the gadget** (Q5). **Measured 2026-11-01:** four
  gadget taps during charge 2's rush (f2937-f2964) were dropped, not queued. The first tap after
  he stopped (f2970, 5 frames later) fired. Q5 holds for the gadget; attack taps were not in this
  take.
  - Tap to pulse took 9, 9 and 10 frames (**0.30-0.33 s**) over three stomps (the sim had 0.25).
    **You set `stomp_windup_seconds` 0.30** (2026-11-01), so the pulse lands 6 ticks after the tap,
    and a stomp tapped after `hero.slow_window` reads `w` stuns the landing's victims iff
    `w` > 0.35 (BRAWL_SIM_DESIGN.md).
  - The cooldown measured 600-601 frames, so 20.0 s holds.
- [x] **(f) Reload.** **Measured 2026-10-28** from your clip (`Videos/2026-10-28 15-30-23.mp4`,
  Training Grounds, full HP so Berserker is off). `scripts/measure_reload.py` without `--write`
  read pip-to-pip reloads of **1.60 s and 1.63 s**, median 1.62. Only two samples, but they agree
  within 0.03 s. **You set `hero_bull.reload_seconds` to 1.6** the same day (D7 for the reload).
  `bot_bull` keeps 2.0, the one weapon key the two blocks no longer share (b).
  Mortis's measured 2.50 went into `brawlers.yaml` the same way, so the sim and the shadow share
  one number. Mortis's digest did not move; Bull's moved to `367cb1d78c40591e`. The 6c run
  trained at 2.0, so live it reloads 25 % faster than it learned, until a run trains at 1.6.
- [x] **(g) Super charge.** **Answered by you 2026-10-23:** "2 full volleys, or 9 projectile
  hits", so `super_charge_hits: 9`, counted per pellet (§5.4).
- [x] **(h) Stomper radius.** **You kept 2.67** (2026-11-01). **Measured 2026-11-01** (f2258-f2275): a dummy whose centre sat
  **3.19 tiles** from Bull's was missed.
  - The white flash reaches 3.0-3.1 tiles. The cyan ring grows from 1.8 to 2.7 over 0.25 s, so
  the earlier "2.4 ring" was a mid-growth reading.
  - The dummy's red ground disc has a 0.84-tile radius and overlapped the flash. That is why the
  stomp looked like a hit. The game tests centres, as the sim does.
  - So the radius is under 3.19, but 2.4, 2.67 and 3.1 all fit. One dummy at about 2.9 tiles
  would separate them, and 2.67 stays until then.
  - Training Grounds showed no use counter across three stomps.

**Done when:** every placeholder in `hero_bull` (5.1b) has a measured value, or one you specified.

### 5.1 Phase 1: the hero seam, row swap (Mortis bit-identical): **DONE 2026-10-23**

- [x] **a. [constants.py](brawl_sim/constants.py):** `HERO_MORTIS` is renamed to `HERO` (value 0)
  everywhere, tests and probes included. The `SuperType` and `GadgetType` enums moved to Phases 4
  and 5, where they are first read.
- [x] **b. [configs/brawlers.yaml](configs/brawlers.yaml): `hero_bull` block.**
  - It is `bot_bull`'s weapon, copied rather than shared, so a D7 change to the bot never moves
    the hero.
  - It has no bot-AI fields, and `super_charge_hits: 0` / `gadget_cooldown: 0`.
  - `move_speed` is 2.57 from the card. Range, reload and spread carry `PROVISIONAL (Phase 0 x)`.
- [x] **c. [config.py](brawl_sim/config.py):**
  - `entities.hero` (default `mortis`, in [default.yaml](configs/default.yaml)) and `EnvConfig.hero`.
  - `KIND_YAML_NAMES` became `BOT_YAML_NAMES` plus `kind_yaml_names(hero)` / `spec_kind_names(spec)`,
    read at all 9 sites.
  - `build_params` refuses a named hero with no block (`check_hero_block`).
  - `BrawlVecEnv` refuses a cfg and a spec that name different heroes. This is its docstring's
    override trap, for the one key both halves read.
- [x] **d. Render.** `viewer.entity_sprite_paths(cfg.hero)`: the hero's sprite is
  `hero_<name>.png`. No `hero_bull.png` exists yet, so Bull draws as the fallback circle. The PNGs
  are your gitignored art.
- [x] **e. Tests.** Only `test_configs_files.py` read `KIND_YAML_NAMES`. The other
  `"hero_mortis"` literals read Mortis's block on purpose, since he stays the default.
- [x] **f. Mortis digest.** A `train.yaml` rollout: 64 envs, seed 0, 100 random decisions
  (500 ticks), CPU. It is [scripts/probes/hero_seam_digest.py](scripts/probes/hero_seam_digest.py).
  Every later phase reruns it: `--out before.json` on the unchanged tree, then
  `--out after.json --compare before.json`.
  - Every step's obs, reward and done, plus all 87 SimState and 75 SimParams tensors at the end,
    match before and after. Two runs before the change also match each other, so the digest is
    deterministic.
  - With `entities.hero=bull`, the agent obs shapes are identical to Mortis's (the critic's
    `kind_onehot` included) and every obs value is finite.

**Done when:**
- [x] the suite is green: 2767 passed, 6 skipped;
- [x] the digest matches;
- [x] `entities.hero=bull` builds, steps and resets ([tests/test_hero_bull.py](tests/test_hero_bull.py)).

### 5.2 Phase 2: the shotgun with aim → milestone M1: **BUILT 2026-10-23** (M1 run not started)

- [x] **a. `hero.weapon_reach`, not a second targeting function.** One reach serves every caller.
  - A kind with a dash reaches `dash_distance + dash_radius`, with the long dash when charged.
  - A kind without one reaches `attack_range + proj_radius`. For Bull that is 7.3 tiles, or 7.7 to
    an enemy's centre.
  - It is a selection, not a blend, so Mortis's reach is the same float as before.
  - `auto_aim_target` reads it, so the targeting rule is unchanged: the nearest alive enemy or
    unbroken crate, with **no visibility check** (user decision 2026-09-26).
  - **Q8 is built as Mortis's** (crates included). It stays open for you to flip.
- [x] **b. The hero's `aim_dir` override (D5),** in `env._attack_phase`.
  - Value 1 aims along the move bin, or along facing when idle.
  - Value 4 aims at the auto-aim target, and fires like value 1 when there is none.
  - Firing does not turn the hero. Mortis's dash path is untouched.
- [x] **c. Tests (`tests/test_hero_bull.py`, 9 new):**
  - Value 1 goes down the bin, past an enemy on the facing; idle value 1 goes along the facing.
  - At 2 tiles and 60° off the bin, value 4 lands at least 4 pellets where value 1 misses.
  - Value 4's reach edge sits at 7.7 tiles (7.6 hits, 7.8 falls back to the bin); with nothing in
    reach and no move, it fires along the facing.
  - `weapon_reach`: Mortis 3.37 uncharged and longer when charged; Bull 7.30 either way.
  - `in_reach_radius`: Mortis 3.77, Bull 5.61. At 5.55 tiles a volley lands at least 3 pellets,
    and at 5.70 it lands at most 2. The in-reach flag reads 1 at 5.0 and 5.55, and 0 at 5.70 and 7.6.
  - The weapon is `bot_bull`'s. Every gun, clip and reload key is equal; `move_speed` is D7's.
  - **The bush test is dropped.** Auto-aim deliberately ignores visibility. The reward's in-reach
    flag still needs an enemy the hero can see.
  - Each test was checked against a deliberately broken build: dropping the aim override, ignoring
    the auto target, using the dash reach for Bull, or keeping Mortis's in-reach radius each fails
    the tests meant to catch it.
- [x] **The `attack_in_reach` reward term reads `hero.in_reach_radius`.** **DECIDED 2026-10-23
  (user): Bull's term pays only where most of his volley can land.**
  - Mortis keeps his 3.77-tile uncharged dash, bit for bit.
  - Bull's radius is the farthest an enemy aimed at dead centre still has a strict majority of the
    pellets touching it. For 5 pellets that is the middle three: 0 and ±0.125 rad, which gives
    0.70 / sin 0.125 = **5.61 tiles** to the enemy's centre.
  - The radius is derived from `proj_count`, `proj_spread_rad`, `attack_range` and the radii, so a
    Q9 spread change moves it.
  - Measured in the sim with an auto-aimed volley:

    | Enemy distance (tiles) | Pellets landed |
    |---|---|
    | 2.0 | 5 of 5 |
    | 2.9 | 3 of 5 |
    | 5.55 | 3 of 5 |
    | 5.70 | 1 of 5 |
    | 7.6 | 1 of 5 |
    | 7.8 | 0 of 5 |

  - Auto-aim (value 4) still reaches the full 7.7. Only the reward is shorter.
  - `scripts/audit_attack_cadence.py`'s sim recorder reads the same function, so the term still
    pays for exactly what the audit measures. Its `--telemetry` side still reads the deployment's
    dash reach (Phase 7).
- [x] **d. Smoke:**
  ```bash
  python scripts/train.py --smoke --set run.env_overrides.entities.hero=bull
  ```
  Ran 2026-10-23 on CPU in 58 s. It walked the curriculum to elite and evaluated every tier
  (8.4 shots per episode). The env it rebuilds from the run's train.yaml is Bull's: 5 pellets,
  dash 0.

**Done when:**
- [x] the suite is green: 2782 passed, 6 skipped (after the in-reach radius decision);
- [x] the Mortis digest still matches (`d53b1c4140510e7d`, every per-step and tensor hash),
  before and after the in-reach radius change;
- [x] the smoke run completes with `entities.hero=bull`.

**M1:** a Bull with no super and no gadget can train on deploy6 unchanged, because Mortis-only fields
read 0. If the GPU is free, a short run answers whether he is learnable at all. Since Phase 5 that
means `super_charge_hits: 0`, `gadget_cooldown: 0` and `reload_boost_below_hp_frac: 0` in
`hero_bull`; `validate()` accepts a STOMP block with no cooldown.

### 5.3 Phase 3: status effects (D4): **BUILT 2026-10-23**

- [x] **a. [state.py](brawl_sim/core/state.py):** `ent_slow_t`, `ent_slow_frac`, `ent_stun_t` and
  `ent_airborne_t`, all (N,E) f32.
  - `zero_` resets them with every other field.
  - `check_invariants` rejects a negative timer, a fraction outside [0, 1], and a fraction with no
    slow running.
- [x] **b. Countdown.** `hero.tick_timers` ends with `status.tick`, in the new
  [core/status.py](brawl_sim/core/status.py). It snaps each timer to 0 once less than half a tick
  is left (D4, "As built") and clears the fraction on its timer's last tick.
- [x] **c. Gates.** `status.disabled` means stunned or airborne.
  - **Movement:** `status.walk_speed` scales the walk. A disabled unit neither walks nor turns.
  - **Attack phase:** `can_attack`, `can_super`, the gadget and melee sub-swings, from every source.
    Overrides are included; they have no mask in front of them.
  - **The hero's mask:** attack, super, gadget and auto are closed. The move half stays open.
  - **Bots:** their intent is gated like anyone's.
- [x] **d. Untargetable while airborne.** These all skip it:
  - `step_projectiles`' unit hit and its blasts;
  - `melee_hitscan` and `advance_dash`;
  - `auto_aim_target`, `super_aim_target` and the in-reach flag.

  Shots fly through it.
- [x] **e. Tests: [tests/test_status_effects.py](tests/test_status_effects.py), 55 cases.**
  - **The countdown:**
    - Every duration from 0.25 to 3.0 s lasts its whole ticks and not one more.
    - `status.tick` runs each timer on its own and clears the slow's fraction with it.
  - **Movement:**
    - A 0.25 s stun releases a bot on the 5th tick, and a 0.5 s airborne on the 10th.
    - A stopped bot neither walks nor turns.
    - Under a 0.4 slow a bot walks 0.288 tiles in 4 ticks, against 0.48 when free, and still
      turns.
    - Mortis's dash lands on the same float under a slow.
  - **Attacking:**
    - A stopped bot cannot fire, even on an override.
    - A stopped hero's mask is `[T, F, F, F]`.
    - His attack, super and gadget do nothing through `env.step`.
    - `_attack_phase`, called directly, refuses all three.
  - **Edgar:** a 0.5 s stun after hit 1 drops hit 2, and a 0.1 s stun does not.
  - **Airborne:**
    - Bull's pellets pass bot 1, and three of them land on bot 2.
    - The spinner's blast skips it but hits the bot beside it.
    - Edgar's cone skips it.
    - Mortis's dash skips it. A dash still inside the unit when it lands hits it then.
    - Auto-aim, the super tap and the in-reach flag skip it.
  - **State:**
    - `check_invariants` catches each broken field.
    - An episode that ends mid-effect starts the next one clear.

**Done when:**
- [x] the Mortis digest matches (`d53b1c4140510e7d`), and so does Bull's (`cb6ee179322bcbaf`). Every
  per-step hash and every tensor both digests share is equal. The four new tensors are listed, not
  compared: the probe's compare now separates tensors added since the baseline;
- [x] the suite is green: 2837 passed, 6 skipped, which is the 2782 before plus the 55 new cases. In
  the full run, the classifier's 30 ms timing test measured 36.7 ms while two probes and the graph
  rebuild shared the CPU; run alone, it passes;
- [x] every gate is load-bearing. 24 mutations each drop one gate, the snap, the fraction's clear,
  an invariant or the reset, and each one fails a test in the new file.

**Found while testing, FIXED 2026-10-23 (it predates Bull).** `env._attack_phase` ran
`combat.melee_hitscan` for every attack that wasn't a dash, ranged ones too.
- **The bug:** a ranged kind's `attack_arc_rad` is 0, and `geo.point_in_cone` passes
  `diff <= half_angle`. A unit or crate exactly on the shooter's facing ray, in range and in sight,
  therefore took one extra full-damage hit on the trigger tick: +600 from a rifle, +880 from Bull.
  0.01 tiles off the ray, it took nothing.
- **How often, in a train.yaml rollout** (64 envs × 400 decisions, random hero actions):
  - Units almost never lined up: 0 free hits in 12,915 bot shots with Mortis, and 1 in 14,687 with
    Bull.
  - Crates did: 1,002 free crate hits with Mortis, spread over every ranged bot kind. They were
    about 7 % of the crate HP removed, and 6 % with Bull.
  - Why crates line up and units don't is untraced. A target that never moves is the likely reason.
  - Every deploy6 run trained with it.
- **Why Phase 2 missed it:** its volley test's `_damage_taken` counted from after the fire step.
- **The fix:** `melee_fire` also requires `attack_arc_rad > 0` (`swings_cone` in
  [env.py](brawl_sim/env.py)). That is the rule brawlers.yaml, `config.cone_ray_tiles` and
  `bots/combat_rules` already used for a cone swinger. `melee_hitscan`'s LOS march relies on it
  too, since the march covers only `cone_ray_tiles`, the longest cone reach. The docstrings of all
  three now say so.
- **Tests:**
  - Phase 2's volley tests read bot 1's HP before the fire step. The value-1 test then fails on the
    old code: the enemy 2 tiles down Bull's facing loses 880 while his volley flies +y. The volley
    at 5.55 and 5.70 tiles landed 4.00 and 2.00 pellets' worth on the old code, inside both bounds.
    It now lands 3.00 and 1.00, the middle-three geometry the 5.61-tile radius assumed.
  - [tests/test_combat.py](tests/test_combat.py)'s
    `test_a_ranged_shot_lands_nothing_in_the_attack_phase_even_dead_on_its_facing_ray` puts the hero
    and a crate on a ranged bot's facing ray and calls `_attack_phase` alone. It covers sniper,
    artillery, rifle, spike and bull. On the old code every case failed, with a free hit of 2320,
    2080, 600, 1080 and 880.
- **The digest moved, as intended:** Mortis `d53b1c4140510e7d` → `5df44a1488f7da5e`, and Bull
  `cb6ee179322bcbaf` → `7c781a0dd35248c6`.
  - For both heroes, step 3 is the first that differs, and 98 of the 101 step hashes differ.
  - 73 of the 91 state tensors moved with Mortis, and 69 with Bull.
  - 12 of the 75 params moved with both. They are resampled at every reset, so they move once an
    episode ends on a different step.
- **The crate economy** (the same rollout, one seed, before → after):

  | Hero | Crate HP removed | Crates broken | Cubes per live bot | Episodes ended |
  |---|---|---|---|---|
  | Mortis | 19.23M → 18.75M (−2.5 %) | 2,006 → 1,938 (−3.4 %) | 0.843 → 0.806 (−4.4 %) | 203 → 208 |
  | Bull | 17.80M → 17.65M (−0.8 %) | 1,922 → 1,895 (−1.4 %) | 1.004 → 1.012 (+0.8 %) | 141 → 140 |

  - The free hits' 7 % share overstated their effect. This note's earlier forecast, that crates
    would open about 7 % slower, was wrong.
  - Ordinary shots made up most of the lost damage, most likely because a bot keeps shooting a crate
    until it breaks.
  - With Bull the change is inside one seed's noise.

**Done when** (the fix):
- [x] the suite is green: 2842 passed, 6 skipped, which is the 2837 before plus the 5 new cases. In
  the full run, the classifier's 30 ms timing test read 32.6 ms again; run alone, it passes;
- [x] both tests fail on the old code and pass on the fixed one;
- [x] the probe's docstring names the new baselines.

### 5.4 Phase 4: the charge super (D2, D3): **BUILT 2026-10-23**

Your answers set four rules:
- **D3:** "Stop at the first wall".
- **D4:** the knockback goes away from Bull.
- **Q2:** one landing pulse, with placeholder numbers.
- **The meter:** "2 full volleys, or 9 projectile hits".

- [x] **a. Per-kind fields** ([config.py](brawl_sim/config.py); `hero_bull` in
  [brawlers.yaml](configs/brawlers.yaml)).
  - **The super type:** `super_type`, from the new `SuperType` in
    [constants.py](brawl_sim/constants.py). It is `BOLT` by default and `CHARGE` for Bull.
  - **Seven new fields:** `super_charge_speed`, `super_windup_seconds`, `super_knockback_tiles`,
    `super_airborne_seconds`, `super_slow_radius`, `super_slow_frac` and `super_slow_seconds`.
  - **Three of the bolt's fields, reused:** `super_range` (the 11 tiles), `super_radius` (on
    centres) and the flat `super_damage`.
  - **What `validate()` refuses:**
    - a CHARGE with a zero speed, range or radius;
    - a negative timing;
    - a `super_slow_frac` outside [0, 1];
    - a BOLT with a charge speed, which is a block that forgot `super_type: CHARGE`.
- [x] **b. State** ([state.py](brawl_sim/core/state.py)).
  - Two new fields, both (N,E) and zero outside a charge: `ent_dash_super` (bool) and
    `ent_dash_windup_t` (f32).
  - `check_invariants` bounds `dash_t` by the longer of the dash and the charge. It also keeps the
    flag and the wind-up inside a running dash.
- [x] **c. `env._attack_phase`.** A CHARGE kind's super fire goes to `hero.start_charge`, and every
  other super still launches a bolt. Both follow the same `super_dir`:
  - along the move bin;
  - or, when the bin is idle, tap-to-fire at the nearest enemy within 12.0 tiles of his centre
    (`super_range + super_radius`). A charge hits on centres, so the bolt's body term drops out.
- [x] **d. `hero.start_charge`** sets the dash state.
  - `dash_t` is the wind-up plus the rush, at the FIXED `super_charge_speed`.
  - The rush is clipped where his body first touches a wall or water, by `start_dash`'s own
    `terrain.body_travel` march (D3).
  - It is a separate function, so `start_dash` stays Mortis's, untouched.
- [x] **e. `advance_dash` under `ent_dash_super`:**
  - the wind-up stands still and hits nothing;
  - `super_radius` and the flat `super_damage`, once per player. Crates are passed untouched;
  - on each new hit, the 1-tile push away from his centre at contact (D4), clipped by walls and
    water, then 0.5 s airborne;
  - on the last tick, the slow pulse: 40 % for 1.5 s on everyone else alive within 3.0 tiles of
    the landing, airborne players included.
- [x] **f. The meter counts pellets.**
  - `step_projectiles` returns `charge_hits` (N,E,E): how many of an attacker's projectiles landed
    on each player, with the gadget's spinner left out.
  - The tick takes the larger of that and the 1 per pair that a melee or dash hit counts, so
    Mortis's dash-plus-bolt tick still counts 1.
  - The charge's own hits count nothing.
  - `super_charge_hits: 9`.
- [x] **g. Budgets.** `charge_ray_tiles` (the longest CHARGE `super_range`, 11) and
  `knockback_ray_tiles` (the longest push, 1) are new static march budgets in `SCALAR_FIELDS`.
  - The push gets its own budget: marched over 11 tiles, a 1-tile push would pay for 11.
  - `charge_ray_tiles` is also the Python switch for the whole charge. It is 0 in every spec
    without a CHARGE kind, so a Mortis run never executes charge code.
- [x] **h. Tests: [tests/test_charge_super.py](tests/test_charge_super.py), 18 cases.** They use
  literals, and the env-level ones read HP from before the fire step.
  - **The push:**
    - Head-on, glancing and dead-centre victims land exactly 1 tile from his centre at contact.
    - Nine tangent grazes go straight across his line.
    - A wall at y = 16 stops a push where the body touches it, on the push's own line, with no
      slide.
  - **Walls and crates:**
    - A charge into a wall column lands just short of it, on tick 24.
    - Crates are passed untouched.
  - **Timing, through `env.step`:**
    - He stands still for 4 ticks, then runs 0.335 tiles a tick and lands at x = 15 on tick 37.
    - A held move bin neither walks nor steers him.
    - Every attack column stays closed until he lands.
  - **Hits:**
    - Nothing is hit in the wind-up.
    - Each player is hit once, for a flat 2000, despite 6 cubes.
    - A hit player is airborne for exactly ticks 16–25.
  - **The landing:**
    - A player 2.9 tiles from it is slowed; one 3.1 tiles away is not.
    - The airborne player is slowed too, and Bull is not.
  - **The meter:**
    - One volley reads 5 and two read 9.
    - The charge's own 2000 adds nothing.
  - **Aim:** the idle super aims at an enemy 11.9 tiles away, and along the facing at 12.1.
  - **Config:**
    - A CHARGE spec must hand `advance_dash` the bank.
    - Mortis's budgets and super type are 0.
    - `validate` refuses a charge that cannot move, and a bolt with a charge speed.
  - **State:** `check_invariants` holds through charges in all four directions.
  - [tests/test_hero_bull.py](tests/test_hero_bull.py) pins `super_charge_hits: 9`.

**As built.** These rules were my calls while building; overturn any of them:
- **Crates are passed untouched,** as Mortis's bolt passes them. Whether the game's charge breaks
  crates is unknown; one Phase 0 take of a charge through a crate would settle it.
- **The charge's own hits charge nothing,** following your count in projectile hits. Mortis's bolt
  hits still charge his meter, as before.
- **A wall-cut charge ends sooner, at the same speed.** Against a wall 6.6 tiles away he lands on
  tick 24, not 37.
- **How the push moves the victim:**
  - The push is added to `ent_vel` too, so that stays this tick's displacement over dt.
  - A victim dead on his centre is pushed along the charge.
  - The capsule includes his start cap, as the dash's does. A player within 1.0 tile behind him
    is therefore hit on the first moving tick and pushed backward.
  - A player the hit kills is still pushed and slowed. Nothing reads a dead unit.
- **A tangent graze is pushed from its closest approach.** In float32, nearly half of exact
  tangents pass the capsule test and then find no crossing of the circle.
  - A probe counted 5,706 of 12,864 tangents on his axis.
  - It also counted 16,618 of 107,170 random near-tangents within 1e-6 of the reach.
  - Taking the tick's start as their contact would tilt the push up to 18.5°.
- **No i-frames.** Bull can be hit through the whole charge, as a dasher can.
- **The pulse also slows airborne players.** A slow is not a hit, and Phase 5's Stomper needs to
  read a slow on a player the charge just launched. The pulse skips Bull himself. A new slow
  replaces a running one only if it lasts longer.
- **Credit:** the 2000 goes through the normal `dmg_by` path, so kills and `last_hit_by` credit
  him as any hit does.
- **Still open:**
  - **Q6, answered 2026-10-28.** The charge always runs the full 11 tiles, and live play drags
    the super to full length (§5.7b).
  - Q5 (can he Stomp mid-charge?) was answered for Phase 5: no, the charge holds the gadget closed
    too (§5.5a).

**Done when:**
- [x] **The Mortis digest matches (`5df44a1488f7da5e`).**
  - All 101 step hashes are equal.
  - So are all 91 shared state tensors and all 75 shared params.
  - The 2 new state fields and the 8 new params are listed, not compared.
- [x] **Bull's digest moved, as expected:** `7c781a0dd35248c6` → `ec7997b246e9d0af`.
  - Step 5 is the first that differs, and 96 of the 101 step hashes differ.
  - 69 of the 91 shared state tensors moved, and 16 of the 75 params (resampled at resets).
  - The obs shapes are unchanged, and no obs value is non-finite.
- [x] **The suite is green:** 2860 passed, 6 skipped. That is the 2842 before plus the 18 new cases.
- [x] **Every rule is load-bearing.** 19 mutations each break one rule, and each one fails at least
  one test in the new file:

  | Mutation | Tests it fails |
  |---|---|
  | no wind-up | 7 |
  | push along the charge always | 2 |
  | push ignores walls | 1 |
  | pulse skips airborne | 2 |
  | pulse radius +0.25 | 1 |
  | charge hits crates | 1 |
  | charge hits charge the meter | 1 |
  | meter per victim, not per pellet | 1 |
  | the weapon's damage | 5 |
  | hits in the wind-up | 1 |
  | super aim keeps the body term | 1 |
  | a wall-cut charge slows to fill the time | 1 |
  | pulse slows the charger | 2 |
  | pulse every tick | 2 |
  | no airborne | 6 |
  | contact at the tick's end | 3 |
  | the charge ignores walls | 2 |
  | a bolt flies with the charge | 1 |
  | a graze's contact is the tick's start | 1 |

  The clean source passes all 18.

### 5.5 Phase 5: Stomper and the star power: **BUILT 2026-10-23**

Your answers set four rules:
- **Q1:** Berserker.
- **Q3:** the current wiki's Stomper: 2.67 tiles, a 50 % slow for 1 s or a 1 s stun on a player
  already slowed, and no damage.
- **D9:** `gadget_hit` pays any stomp that slows or stuns a player.
- **Q5:** the charge holds the gadget closed.

- [x] **a. `gadget_type: STOMP`.** The new `GadgetType` is in [constants.py](brawl_sim/constants.py).
  SPINNER is 0 and the default, so Mortis's block keeps his gadget without naming one.
  - **Per-kind fields** ([config.py](brawl_sim/config.py); `hero_bull` in
    [brawlers.yaml](configs/brawlers.yaml)):

    | Field | Bull | Source |
    |---|---|---|
    | `gadget_cooldown` | 20.0 | the card; starts charged, as Mortis's does |
    | `stomp_radius` | 2.67 | the wiki (the clip read a 2.4 ring inside a 3.1 glow) |
    | `stomp_windup_seconds` | 0.25 | the clip: tap to stomp 0.23–0.30 s (0.30 since 2026-11-01, §5.0 e) |
    | `stomp_slow_frac` | 0.5 | the wiki |
    | `stomp_slow_seconds` | 1.0 | the wiki |
    | `stomp_stun_seconds` | 1.0 | the wiki, on a player already slowed (1.4 since 2026-11-01, §5.0 d) |

  - **State:** `ent_stomp_t` (N,E) f32, the wind-up left on an armed stomp.
    - `zero_` clears it, so a reset disarms a pending stomp.
    - `check_invariants` bounds it by the kind's wind-up.
  - **A timer, not the HAZARD projectile** this step first proposed. A per-entity timer is one
    float. It takes no projectile slot, so a full buffer can never drop a stomp. It also needs no
    projectile rule for a pulse that follows its owner and spares him.
  - **Arming.** `env._attack_phase` spends the cooldown and reveals him, as for the spinner. Then
    `hero.start_stomp` sets `ent_stomp_t` to the wind-up instead of throwing.
  - **Landing.** A new tick phase, **5b**, runs `env._stomp_phase` (`hero.advance_stomp`) before the
    attacks.
    - It counts the wind-up down and lands the stomp 5 ticks after the tap, around where he stands
      then.
    - A stun it applies stops that same tick's attack and walk.
  - **The pulse** catches every other player alive within 2.67 tiles of his centre, on centres.
    - A player already slowed, by anything (the charge's landing included), is stunned 1.0 s and
      keeps its slow.
    - Any other is slowed 50 % for 1.0 s.
    - The slows are read from before the pulse.
    - An airborne player is skipped (D4).
    - No damage. Crates are not players.
  - **The Python switch** is `params.stomp_radius_tiles`: the widest STOMP radius in the spec, 0
    without one. A Mortis run skips phase 5b and the spinner/stomp split in Python.
  - **What `validate()` refuses:**
    - a STOMP with a zero radius or wind-up;
    - a STOMP with neither a slow nor a stun;
    - a negative stomp duration;
    - a `stomp_slow_frac` outside [0, 1];
    - a SPINNER with a stomp radius, which is a block that forgot `gadget_type: STOMP`.

    A STOMP with `gadget_cooldown: 0` (M1) passes.
  - **The charge holds the gadget closed (Q5).** `hero.gadget_ready` gains `~ent_dash_super`
    behind the charge's switch. So the mask, the obs `hero.gadget_ready` and the attack phase all
    read it. Mortis's dash still leaves his spinner open.
  - **Whole ticks for the gadget cooldown.** `ent_gadget_cd` now snaps to 0 as the status timers do
    (`status.count_down`). Under the plain clamp, float32 left 1.8e-5 s of 20.0 after its 400th
    tick, so the stomp would have come back on tick 401. Mortis's 18.0 reaches 0 on its 360th tick
    either way.
- [x] **b. Star power: Berserker** (`hero.tick_timers`).
  - **Fields:** `reload_boost_below_hp_frac` 0.6 and `reload_boost_mult` 2.0.
  - **The rule:** while his HP is strictly below 60 % of his current max HP (cubes' bonus
    included), each reload tick gains twice as much.
  - It reads this tick's HP before any of its damage.
  - It has no latch: healing back to 60 % ends it.
  - It still waits out `attack_cd`.
  - `validate()` refuses a threshold outside [0, 1], and a threshold whose multiplier is 0.
  - Tough Guy is not built.
- [x] **c. Reward (D9).** `_run_tick` ORs phase 5b's flag, "the hero's stomp slowed or stunned a
  player", into `gadget_hit`.
  - It pays once per stomp, however many players it catches.
  - At action_repeat 5 the stomp lands on the next decision's first sub-tick, so the term pays on
    the decision after the tap.
  - A stomp that catches nobody, or only airborne players, pays nothing.
  - `train.yaml`'s 0.3 is unchanged.
- [x] **d. Tests: [tests/test_stomper.py](tests/test_stomper.py), 45 cases.** They use literals and
  run through `env.step` where they can.
  - **The pulse:**
    - A fresh player is slowed by half from the tap's 5th tick through the 19 after it, and walks
      0.06 a tick instead of 0.12.
    - A player already slowed is stunned for exactly those 20 ticks and keeps its slow. Asked to
      fire, it fires nothing until tick 26.
    - The charge, then the stomp: the player the charge hit, once he has landed, is stunned, and
      has lost exactly 2000.
    - Airborne players are skipped, slowed or not.
    - It catches players at 2.6 tiles and on a 2.62 diagonal, and misses at 2.75 and on a 2.76
      diagonal.
    - The dead are skipped, and a stomper who dies in the wind-up lands nothing.
    - It deals no damage, to crates either.
  - **The tap:**
    - He walks and may attack through the wind-up, and the pulse lands where he stands.
    - The tap throws no spinner: a spy on `projectiles.spawn_gadget` sees no firing row.
    - A reset disarms a pending stomp and recharges the gadget.
    - The charge holds the gadget closed, in the mask and the obs alike, from its tap until he
      lands. A tap mid-charge is a no-op.
  - **Timing:**
    - The cooldown is 400 ticks to the tick.
    - At action_repeat 5 the stomp lands on the next decision's first tick. It re-arms on the
      81st observation (20.25 s), as `test_gadget.py` pins Mortis's 73.
  - **The reward:** `gadget_hit_tick` is 1 on the landing tick for one or two players caught, and 0
    for nobody or an airborne player alone.
  - **Berserker:**
    - It doubles the reload below 60 % of max HP, at 10000 and at 12000 (cubes).
    - It crosses 60 % both ways, with no latch.
    - It still waits out `attack_cd`, and it is Bull's alone.
  - **Config and state:**
    - `status.apply_stun` keeps the longer stun and leaves a slow running.
    - A Mortis spec has no stomp and skips phase 5b.
    - `validate` refuses 8 broken blocks and takes 3 good ones, M1's included.
    - `check_invariants` bounds the stomp timer.

**As built.** These rules were my calls while building; overturn any of them:
- **A per-entity timer** instead of the HAZARD projectile (a. above).
- **Phase 5b sits before the attacks.** A 5-tick wind-up then lands exactly 5 ticks after its
  tap, and a stun it lands stops that tick's attack.
- **He is free through the wind-up.** He walks, attacks and supers as usual, and the pulse lands
  around wherever he stands when it runs out. At action_repeat 5 it always lands before the next
  decision's super can fire.
- **The reach is centre to centre, ≤ 2.67 tiles,** as the charge's landing pulse is.
- **Who the pulse leaves alone:** himself, the dead, crates and airborne players (D4, §5.3).
- **"Already slowed" is read before the pulse,** so a stomp never stuns with its own slow.
- **A stun keeps a running slow.** Once the stun ends, the slow's remainder still applies.
- **Two stompers on one player in one tick** would take the largest of each number. No such tick
  happens today: only the hero stomps.
- **A dead stomper lands nothing.**
- **The tap reveals him** and breaks out-of-combat regen, the existing gadget rule.
- **The charge closes every gadget,** behind the charge's switch, so the obs `hero.gadget_ready`
  reads False mid-charge too.
- **Berserker** reads the current max HP (cubes included), with a strict `<`, no latch and no alive
  term.
- **`validate()`** accepts M1, a STOMP with cooldown 0, and refuses a SPINNER that has stomp
  numbers.
- **The gadget-cooldown snap,** above. It changes no Mortis tick.

**Done when:**
- [x] **The Mortis digest matches (`5df44a1488f7da5e`).**
  - All 101 step hashes are equal.
  - So are all 93 shared state tensors and all 83 shared params.
  - The new state field (`ent_stomp_t`) and the 8 new params are listed, not compared.
- [x] **Bull's digest moved, as expected:** `ec7997b246e9d0af` → `6ab2df8a244c9861`.
  - Step 0 is the first that differs, because the reset obs now shows the gadget charged. All 101
    step hashes differ.
  - 71 of the 93 shared state tensors moved. So did 13 of the 83 params: `gadget_cooldown` itself
    and 12 resampled at resets.
  - The obs shapes are unchanged, and no obs value is non-finite.
- [x] **Nothing else leaked into Bull.** I reran the same rollout with his new kit switched off
  (`hero_bull`'s `gadget_cooldown` and `reload_boost_below_hp_frac` zeroed through
  `run.env_overrides`). It reproduces Phase 4's `ec7997b246e9d0af` bit for bit.
- [x] **The suite is green:** 2905 passed, 6 skipped. That is the 2860 before plus the 45 new cases.
- [x] **Every rule is load-bearing.** 19 mutations each break one rule, and each one fails at least
  one test in the new file:

  | Mutation | Tests it fails |
  |---|---|
  | no wind-up | 16 |
  | pulse every tick | 11 |
  | fresh players stunned too | 1 |
  | "already slowed" read after the pulse's own slow | 8 |
  | airborne players caught | 3 |
  | the dead caught | 1 |
  | pulse radius +0.25 | 2 |
  | the pulse catches the stomper himself | 4 |
  | a stun clears the slow | 3 |
  | a stun replaces a longer one | 1 |
  | the charge leaves the gadget open | 1 |
  | `gadget_hit` ignores the stomp | 3 |
  | Berserker at exactly 60 % too | 2 |
  | Berserker always on | 4 |
  | Berserker reads the base max HP | 1 |
  | the tap throws a spinner too | 1 |
  | a dead stomper still lands | 1 |
  | no snap on the gadget cooldown | 1 |
  | the pulse after the attacks | 16 |

  The clean source passes all 45.
  - The first sweep ran over 44 tests, and "the tap throws a spinner too" survived it. Bull's
    spinner numbers are all 0, so the extra spinner changed nothing a test read; only the
    projectile slot it took showed.
  - The spy on `projectiles.spawn_gadget` was written for it.

**The attack-cooldown snap (your call 2026-10-23, built 2026-10-24).** `hero.tick_timers` now
counts `ent_attack_cd` down with `status.count_down`, as the status timers and the gadget cooldown
already did.
- **Why:** the plain float32 clamp left 7.45e-9 s of a 0.25 s cooldown after its 5th tick. So
  every kind with `attack_cooldown: 0.25` could not fire again until the 6th tick, and paused its
  reload a 6th tick too. Those kinds are Bull (hero and bot), sniper, artillery, rifle and spike.
- **The digests.** Both were taken fresh first, and both still matched Phase 5's.

  | | Mortis | Bull |
  |---|---|---|
  | Before | `5df44a1488f7da5e` | `6ab2df8a244c9861` |
  | After | `247884ee3f0cf382` | `483861f81de74216` |
  | Step hashes that differ | 99 of 101, from step 2 | 99 of 101, from step 2 |
  | State tensors moved | 77 of 94 | 72 of 94 (his dash tensors stay put) |
  | Params moved | 12 of 91 | 12 of 91 |

  - The 12 params are all resampled at resets, and episodes now end on other ticks.
  - The obs shapes are unchanged, and no obs value is non-finite.
- **Why step 2 (ticks 6 to 10) is the first to differ:**
  - **Mortis:** only `critic_enemies` differs there, in 51 of 64 envs. That is the privileged
    critic's view of the 0.25-s bots' cooldown and ammo. Their shots are the same through tick
    10, and Mortis's own obs is unchanged at step 2.
  - **Bull:** the digest's actions are random and unmasked. In the 26 envs where he fired on tick
    1, his reload resumes a tick sooner, so `self` differs in all 26. In 16 of them he also fires
    again on tick 6, which the residue used to block, so `projectiles` and `grid` differ there.
    `critic_enemies` differs in the same 51 envs as Mortis's.
- **Nothing else moved.** I patched the old plain clamp back over that one line of the new code.
  The same rollout then gives both old digests bit for bit.
- **Fire cadence.** One shooter at a time starts with a full clip and fires at an invulnerable
  target for 2400 ticks. Gaps are in ticks of 0.05 s:
  - **re-fire** is the gap between shots while the clip lasts, so the cooldown alone;
  - **sustained** is the mean gap once the clip is empty, so the cooldown plus the reload;
  - **raw** asks to fire every tick;
  - **in training** is the hero at action_repeat 5 behind the action mask, or the bot AI at
    decision periods 2, 3, 4, 6 and 8, the tiers' periods.

  | Shooter (`attack_cooldown`) | Raw re-fire | Raw sustained | Re-fire in training | Sustained in training |
  |---|---|---|---|---|
  | sniper, Brock (0.25) | 6 → 5 | 48 → 47 | 6, or 8 at p 4 and 8: same | 48.00 → 47.00 |
  | artillery, Grom (0.25) | 6 → 5 | 52 → 51 | same | 52.00–52.09 → 50.91–51.00 |
  | rifle, Shelly (0.25) | 6 → 5 | 41 → 40 | same | 40.98–41.04 → 39.95–40.00 |
  | spike (0.25) | 6 → 5 | 51 → 50 | same | 50.91–51.00 → 49.96–50.00 |
  | bot Bull (0.25) | 6 → 5 | 46 → 45 | same | 45.88–46.04 → 44.94–45.00 |
  | hero Bull (0.25) | 6 → 5 | 46 → 45 | 10: same | 45.92 → 45.00 |
  | hero Bull with Berserker | 6 → 5 | 26 → 25 | 10: same | 25.96 → 25.00 |
  | hero Mortis (0.35) | 7 | 57 | 10 | 57.05 |
  | Edgar (0.50) | 10 | 35 | not run | |
  | melee, Buzz (1.00) | 20 | 40 | not run | |

  The cone bots were left out of the AI runs because they park in contact with a motionless
  target and barely fire (the found-in-passing note below). Their raw rows cover them.
- **What changes in training:**
  - **Bursts do not.** A bot fires only on its decision ticks: every 2 ticks for elite, 3 for
    expert, 4 for hard and veteran, 6 for medium and 8 for easy. Its first decision after 5 ticks
    is the same tick as after 6, so its re-fire stays 6 or 8. Only a period of 1 or 5 would reach
    the 5th tick, and no tier ships one. The masked hero's re-fire stays 10.
  - **Sustained fire does, by one tick per shot.** Every 0.25-s kind pauses its reload a tick less.
    So once its clip is empty, it fires one tick sooner per shot at every tier, about 2 % more
    shots (2.0 to 2.5 % by kind).
  - **Bull under the mask** now takes exactly 45 ticks a shot once his clip is empty, the 5-tick
    cooldown plus 40 of reload. That is 9 decisions, so his shots no longer slip to the next
    boundary: 2.0 % more shots, and 3.8 % with Berserker (25 ticks, 5 decisions).
  - Mortis's 0.35, Edgar's 0.50 and Buzz's 1.00 come out the same either way.
- **The decision-boundary cap is unchanged.** A hero shot on a decision's first tick leaves 0.05 s
  at the next boundary, where the mask reads `attack_cd <= 0`. So both heroes re-fire at 0.50 s at
  best: 50 % of Bull's weapon rate and 70 % of Mortis's. BRAWL_SIM_DESIGN.md's cap paragraph now
  gives both.
- **Tests: 2 new, 5 cases.**
  - `test_every_shipped_attack_cooldown_blocks_exactly_its_whole_ticks` (tests/test_hero.py):
    0.25, 0.35, 0.50 and 1.00 s block the reload for exactly 5, 7, 10 and 20 ticks, and read 0 on
    the last.
  - `test_his_0_25_s_cooldown_refires_on_the_fifth_tick` (tests/test_hero_bull.py): at
    action_repeat 1, Bull asked to fire every tick fires on ticks 0, 5 and 10.
  - With the old clamp patched back, the 0.25 case and the Bull test fail, the other three pass.
    The failures are 7.45e-9 s left on tick 5, and shots on ticks 0 and 6.
  - The suite: 2910 passed, 6 skipped. The classifier's timing test failed once while the probes
    shared the CPU, and passed when rerun alone.
- **Live:** the shadow still clamps plainly. §5.7b lists it.

**Found in passing, deferred by you: cone bots swing only where they walk.** The cadence probe's AI
runs put one RUSH bot 2.5 tiles from the motionless hero on the tiny blank map, at decision
period 4.
- Buzz and Edgar walk in and park about 0.22 tiles from him, their facing frozen about 104° off
  his bearing. `combat_rules`' cone gate tests `ent_facing`, so it stays shut: each fired once in
  2400 ticks.
- The probe's invulnerability is not the cause. With the hero vulnerable, each still fired once in
  400 ticks.
- Against a hero walking east and west, 20 ticks each way, Buzz fired 11 times in 400 ticks and
  landed nothing. Edgar fired once.
- Bot Bull, a projectile kind, fires at his full rate from the same spot, because his shot follows
  his aim, not his facing.

**Measured in training rollouts, 2026-10-24, at your request.** The probe only reads, and nothing
in the sim changed.
- **How:** `scripts/probes/cone_bot_holds.py`, whose docstring has the commands.
  - It builds the env as configs/train.yaml does and sets one curriculum stage. Each of its 64
    envs plays one whole match (740 decisions), at stages 0–4 and seeds 0 and 1. The stages pool
    by their share of the 400M run.
  - The driver is the served deploy6 checkpoint, its actions sampled as in training. For Bull it
    is a stand-in through the D1 row swap, since no Bull policy exists. A uniform random policy
    at stage 0 stands for the start of a run.
  - Its hooks wrap four functions and the tick hook. Both seam digests are unchanged with them
    installed.
  - Damage is exact. The hooks read each attacker's entry in the matrices that
    `combat.dominant_attacker` receives. On every combat damage call, the hero's column summed to
    exactly what he was charged.
- **Terms:**
  - *Contact:* the bot's target is the hero, within its attack range.
  - *Held:* a decision tick with every other gate open (ammo, cooldown, reaction, line of sight,
    fire range, the lateral hold, the camper veto, not stunned), but the hero outside the cone.
    Held time counts each held decision as its decision period.
  - *Stall:* consecutive held decisions in unbroken contact, with no swing between.

| Hero, driver | Buzz contact, s per hero-minute | Buzz held, share of contact | Edgar contact, s per hero-minute | Edgar held, share of contact | Hero still, share of contact |
|---|---|---|---|---|---|
| Mortis, deploy6 | 1.15 | 30.6 % | 0.94 | 63.5 % | 5–6 % |
| Bull, deploy6 stand-in | 0.37 | 20.5 % | 0.23 | 50.6 % | 9 % |
| Mortis, random | 2.12 | 32.8 % | 1.76 | 74.0 % | 10 % |
| Bull, random | 0.51 | 37.6 % | 0.47 | 64.4 % | 9 % |

- **Per run** (one stage, one seed), Buzz's share ran 11–29 % with the Bull stand-in and 24–43 %
  with Mortis. Edgar's ran 42–58 % and 58–69 %. The random Bull's Buzz share was 48 % on one seed
  and 17 % on the other.
- **In time,** that is about 1 s of held swings per minute of Mortis's play (Buzz 0.35 s, Edgar
  0.60 s), and 0.2 s per minute of the Bull stand-in's. He spends a quarter to a third as long in
  their reach.
- **The holds are short.** Under the deploy6 driver, stalls have a median of 0.3–0.5 s, a p90 of
  1.1–2.3 s and a longest of 12.9 s (Buzz on Mortis). Stalls of 3 s or more: 114 in Mortis's
  1970 hero-minutes, 4 in the Bull stand-in's 1971.
- Under the same driver, held decisions faced a median 95–128° off the hero. The bot itself stood
  still in 10–25 % of them.

| Hero, driver | Buzz swings per contact-second | Buzz damage per contact-second, HP: all / hero still / moving | Edgar swings per contact-second | Edgar damage per contact-second, HP: all / hero still / moving |
|---|---|---|---|---|
| Mortis, deploy6 | 0.59 | 521 / 531 / 521 | 0.60 | 961 / 1263 / 943 |
| Bull, deploy6 stand-in | 0.92 | 938 / 801 / 951 | 0.93 | 1684 / 1011 / 1751 |
| Mortis, random | 0.54 | 378 / 400 / 376 | 0.44 | 602 / 564 / 606 |
| Bull, random | 0.64 | 571 / 792 / 549 | 0.60 | 859 / 869 / 858 |

**Why: a bot faces where it walks.** `movement.apply_movement` copies a bot's smoothed walking
direction into `ent_facing`, and the cone gate tests `ent_facing`. A projectile follows the aim
instead, so only the cone kinds hold.
- **Only CLOSE walks at the target, and not straight at it.** `steering.combine` adds a 0.30 unit
  strafe to a seek whose length is the distance (`policy.path_toward` under `bots.nav`), then
  normalizes once. So a steady CLOSE walk faces atan(0.30/d) off the target.
  - Edgar's half-cone is 0.25 rad (a 0.50 arc, no sweep), so the blend alone shuts it inside
    1.175 tiles, 59 % of his 2.00-tile reach. Walking in CLOSE, he held 98–99 % of ready
    decisions under 0.6 tiles, 67–73 % at 1.18–1.45 tiles and 31–41 % beyond. Beyond 1.18 tiles
    his facing sat within a median 5° of atan(0.30/d). Inside, it trailed by a median 4–13° more.
  - Buzz's half-cone is 0.936 rad (a 0.65 arc plus a 1.222 sweep), so the blend shuts it only
    inside 0.22 tiles. His holds come from his facing turning slower than the bearing: a median
    4–23° past atan(0.30/d) at 0.22–2 tiles, where he held 17–60 %.
- **The other modes walk elsewhere.** HOLD_RANGE strafes at its band, RETREAT flees, and
  HOLD_STILL keeps its last facing. Held share of ready decisions:

  | Mode | Buzz, Mortis / Bull | Edgar, Mortis / Bull |
  |---|---|---|
  | CLOSE | 45.8 / 21.6 % | 87.0 / 72.6 % |
  | HOLD_RANGE | 92.8 / 83.1 % | 97.3 / 95.1 % |
  | HOLD_STILL | 89.2 / 89.0 % | 96.1 / 96.1 % |
  | RETREAT | 97.4 / 93.9 % | 98.9 / 98.8 % |

  By personality, KITE holds most (87–98 %) and a RUSH Buzz least (26–51 %).
- **One reason the Bull stand-in sees fewer holds:** Mortis fights point-blank, where the cones
  are shut most. The stand-in meets Buzz mostly at the edge of Buzz's reach. 60 % of Buzz's ready
  decisions in a CLOSE walk against him came at 2–2.68 tiles, against 20 % for Mortis, and Buzz
  held only 5–17 % there.

**Not the motionless stalemate.** The hero stood still in only 5–10 % of the contact time, and a
still hero was not safe. Per contact-second, a still Mortis took about what a moving one did,
and more from Edgar (1263 against 943 HP). A still Bull stand-in took 801 HP from Buzz and 1011
from Edgar, against 951 and 1751 moving. This note's old guess, that a Bull standing still in
contact with a cone bot would be safe from it, came from the tiny map. It is wrong in training
rollouts.

**Against other bots the holds last longer.** With another bot as the target, Buzz held 57–61 %
of the contact time and Edgar 79–82 %, under every driver. The stalls reach a p90 of 7–10 s and a
longest of 78 s. The Mortis rollouts' 1970 hero-minutes hold 130 Buzz and 211 Edgar stalls of
10 s or more, and the Bull stand-in's hold 116 and 190. So a cone bot can stay in reach of
another bot for most of a minute without swinging, which shapes how fast the bot field thins out.

**By-product: the same policy wins twice as often playing Bull.** These are first matches (each
env's first, as TierEvaluator counts them), with the stages pooled as above.
- Bull: 60.3 % (383 of 640), mean place 2.29. By stage, 0 to 4: 75, 66, 58, 52 and 48 %.
- Mortis: 30.5 % (192 of 640), mean place 4.00. By stage: 50, 40, 27, 16 and 16 %.
- The policy never trained on Bull, so this is his kit against the same bots. It is not an eval:
  TierEvaluator's fixed tiers are.

**DEFERRED by you, 2026-10-27:** leave the cone bots' swing aiming as it is for now. Nothing
changed.

### 5.6 Phase 6: obs spec and training: **6a and 6b DONE 2026-10-27; 6c TRAINED by you 2026-10-28**

- [x] **a. `configs/agent_obs_bull.yaml`** (D6): **BUILT 2026-10-27.**
  - **The spec is deploy6 with D6's edits and nothing else**
    (`test_bull_differs_from_deploy6_by_exactly_d6`).
    - `self` is 26 floats: the long-dash pair is out, and `hero.slow_window` comes in after the
      gadget pair.
    - `critic_enemies` is 9×28, with `entities.slowed_t`, `stunned_t` and `airborne_t` after
      `entities.dashing`.
    - Widths: 274 actor floats under the run's auto-aim (271 without), and 258 critic floats.
    - Its header gives the run command.
  - **New state: `ent_slow_window_t`, the charger's own window.**
    - `hero.advance_dash` sets it to his `super_slow_seconds` (1.5) on every landing, whether it
      caught anyone or not.
    - `status.tick` counts it down with the other timers, so it equals each caught player's slow,
      tick for tick.
    - `check_invariants` bounds it by the row's own `super_slow_seconds`, so a bot's must stay 0.
    - The four obs fields are views of state, as `hero.dash_t` is.
  - **Schema and docs.** `obs_schema` declares the four rows (seconds, ≥ 0), and `hero.dash_t`'s
    description now names the charge. `docs/AGENT_OBS_BULL.md` is generated and kept in the repo,
    as deploy6's is; README.md and BRAWL_SIM_DESIGN.md §9 list it.
  - **13 new tests** across `test_configs_files.py`, `test_obs_select.py`, `test_obs_schema.py`,
    `test_status_effects.py` and `test_charge_super.py`. They include the stun rule on the tick
    grid: a stomp tapped after a 0.35 read stuns, and one after a 0.30 read slows.
  - **Results.** Suite: 2923 passed, 6 skipped. Digests unchanged: Mortis `247884ee3f0cf382`, Bull
    `483861f81de74216`. The new tensor is listed, not compared.
  - **Deployable since 7b (2026-10-28).** The shadow supplies `hero.slow_window` live, and the
    deploy-spec coverage tests name this file. The yaml's own header still says "NOT BUILT"; it
    is a spec file, so it was left as is.
- [x] **b. Pre-training check** on CPU, as on 10-07: **RUN 2026-10-27, passed.** Scratch scripts
  only; nothing was added to the repo.
  - **The smoke** (`train.py --smoke` with the run's two `--set`s) exited 0 in 0.9 min.
    - It ran all six eval tiers and saved checkpoints at 4096 and 8192.
    - The saved model reads Bull's inputs: `self` (26,), and `critic_enemies` (2, 28) at the
      smoke's 2 enemies.
    - Its actions are `[17 5]`, and its value net takes 334 = 104 CNN + 168 MLP + 62 critic
      floats.
  - **A rollout check:**
    - **Setup.** `configs/train.yaml` as the run builds it, plus `engine.debug_checks`, so every
      tick's invariants ran. 96 envs, 16 pinned to each tier. Half played random legal actions
      and half a scripted Bull. 900 decisions, then a 600-decision pass with every env scripted.
    - **Clean on every check:**
      - no invariant break;
      - no non-finite obs (raw or normalized), critic input or reward;
      - a legal move and attack on every decision;
      - every reward term finite.
    - **Capacity peaks:** projectiles 75/192, crates 44/48, pickups 11/64, cubes 28/99.
    - **Episodes:** 904 finished, none truncated (the longest ran 513 of 740 decisions). Random
      play won 2 of 304. The scripted Bull won 3–14 per tier.
    - **New fields:**
      - `hero.slow_window` stayed in [0, 1.5] on the 0.05 grid.
      - The critic timers stayed in [0, 1.5] / [0, 0.80] / [0, 0.50] on live rows, and 0 on dead
        ones.
      - The full obs' long-dash pair was 0 throughout.
      - `hero.dash_t` reads at most 1.59 at a decision's end: a full charge, counted down from
        its tap tick. D6's 1.84 is the charge's whole length.
  - **Gadget timing** at the shipped decision rate:
    - legal on the first obs of all 1,094 episode starts;
    - closed on every decision of every charge;
    - the pulse lands on the next decision's first tick: the stomp timer read 0.05 at the end of
      the tap's decision, for all 1,060 taps;
    - re-armed on the 80th obs after the tap's, so the next stomp goes on decision 81, 518 times.
      The 4 later re-arms were all held by a charge.
    - **the stun rule**, scored on every landing-slowed bot a pulse touched, in the all-scripted
      pass (a stomp 1–6 decisions after a landing that caught a bot, once that bot is within
      2.2 tiles):
      - all 16 bots whose window read 0.05–0.30 before the tap were slowed, and all 74 at
        0.35–1.45 were stunned: 0 violations;
      - every read from 0.05 to 0.50 occurred, so the edge is exactly between 0.30 and 0.35 (2
        slowed at 0.30, 9 stunned at 0.35);
      - 27 of the 74 stunned bots died later in the pulse's decision. Death leaves status timers
        running, so they were scored from their timers. No stun began on any other decision.
  - **The watch.py look:**
    - **The smoke's model.** watch.py found the smoke's own `train.yaml` and played its Bull. The
      first stomp went on tick 6 (charged at spawn) and pulsed on tick 11.
    - **The deploy6 stand-in.** The Mortis-trained policy playing Bull, through a scratch copy of
      its `train.yaml` with `entities.hero: bull`, charged and stomped once each in a 361-decision
      match. Its stomp slowed a bot, and the zone killed it mid-charge. It is no guide to what
      a trained Bull will do.
    - **Scripted Bull matches.** These were recorded through watch.py's own `build_watch_env` and
      per-tick frame capture, then drawn with `ReplayViewer`. Two sequences, frame by frame:
      - window read 0.95: charge, wind-up 0.20 s, the first rush tick launches the bot (airborne
        0.50). The wall stops the charge after 3.3 tiles (D3), and the landing slows the bot 40 %
        for 1.50, equal to his window. The stomp is tapped, and the pulse stuns the bot 1.00 with
        its slow at 0.65; the two count down together;
      - window read 0.10: a fresh 50 % slow for 1.00, no stun.
    - **`render/viewer.py` labels status effects** since 2026-10-27, your follow-up to 6b. The
      frames above predate it: they put scratch labels over the viewer.
      - Each living entity gets one line per running timer, under its ammo count, most disabling
        first: `AIR`, `STUN`, `slow N%`, and on a charger `slow window`. `e` toggles them, and a
        recording older than the timers plays with none.
      - Labels that would cover each other stack downward, top of the screen first. A charge
        carries the bots it hits to its landing, so without this the charger's window and their
        labels overprinted on exactly the landing, stomp and pulse frames.
      - Replayed through it with no scratch overlay: the scripted match above (the stun read
        `STUN 1.00 / slow 40% 0.65`, stacked under Bull's `slow window 0.65`), and the stand-in's
        stomp through the viewer's own npz CLI (`slow 50% 1.00`).
- [x] **c. The run: TRAINED by you, finished 2026-10-28.** It is `runs/mortis_ppo-20261027-160804`
  (400M, 4096 envs). It ran `train.yaml` with both overrides, which must go together (the spec's
  header says why):

  ```bash
  python scripts/train.py --set run.agent_obs=configs/agent_obs_bull.yaml --set run.env_overrides.entities.hero=bull
  ```

  - **Fixed eval tiers.** The best eval is the final one, at 400M.

    | | Bull | Mortis deploy6 (`mortis_ppo-20261012-082348`) |
    |---|---|---|
    | six-tier mean | **.825** | .424 |
    | holdout | **.813** | .391 |
    | easy | .97 | .87 |
    | medium | .95 | .66 |
    | hard | .92 | .53 |
    | veteran | .80 | .24 |
    | expert | .71 | .18 |
    | elite | .59 | .06 |

  - **The curve, six-tier / holdout:** .60/.58 at 60M, .68/.68 at 100M, .73/.73 at 200M,
    .79/.77 at 300M, .83/.81 at 400M. It was still rising at the end, about +.03 per 100M.
  - **Holdout gap:** about .01. Mortis's was .03.
  - **Explained variance:** .60 at the end, against Mortis's .70.
  - **Part of the margin is the kit, not the training.** The untrained deploy6 policy already won
    60% of first matches as Bull against 30% as Mortis. Several of Bull's sim numbers are also
    placeholders or knowingly off:
    - the landing slow (Q2);
    - his walk speed (D7; 2.68 since 2026-11-01);
    - the fan width (Q9; measured 0.50 since);
    - D3's wall stop (a live charge breaks walls).

    The live match is the test.
  - `watch.py` plays it next to other GPU work on CPU (`CUDA_VISIBLE_DEVICES=-1`). It won 3 of 3
    matches against hard bots on the 00:08 best model of 10-28.
- [ ] **d. The retrain on the measured numbers: yours.** Fresh, not a fine-tune (your call,
  2026-11-01), at 300M instead of 400M. 6c reached elite on the gates at 27M and was at .79 by
  300M. A 300M run at the 83M cap would force elite only at 332M, past its end, so the cap comes
  down to 60M (forced elite by 240M). The LR and clip schedules follow `total_timesteps`, so they
  decay over the 300M:

  ```bash
  python scripts/train.py --set run.agent_obs=configs/agent_obs_bull.yaml --set run.env_overrides.entities.hero=bull --set run.total_timesteps=300000000 --set curriculum.max_timesteps_at_stage=60000000
  ```

  Its evals face bot Bulls at 5.2 / 9.0 / 2.68, so they compare with each other, not with 6c's.

### 5.7 Phase 7: deployment

- [x] **a. The hero name from the run** (D8): **BUILT 2026-10-28.**
  - `DeployLoop.hero_kind = kind_yaml_names(policy.cfg.hero)[0]` replaces `loop.HERO_KIND`.
  - A hero outside `loop.SHADOW_KINDS` raises at startup rather than play on another hero's
    timers. 7b added `hero_bull` there.
  - `dash_reach_tiles` → `attack_reach_tiles(hero)`, and `auto_aim_reach_tiles(hero, long_dash=)`
    no longer restate the sim's formulas. They call `hero.in_reach_radius` and
    `hero.weapon_reach` on the sim's own params (`build_params` on `default.yaml` +
    `brawlers.yaml`, ~85 ms once at startup), so Bull's volley geometry has one copy.
  - Values: Mortis 3.77 / aim (3.77, 3.87), long (6.44, 6.54), the same as before. Bull 5.61 / aim
    (7.70, 7.80), and 5.42 / (5.90, 6.00) since the measured 5.2-tile range (2026-11-01).
  - Tests: [test_deployment_loop.py](tests/test_deployment_loop.py) pins both heroes' numbers as
    literals and the refusal of a hero the shadow does not model.
- [x] **b. `ShadowParams` / `ShadowHero` become kit-generic: BUILT 2026-10-28.**
  - **Params.** `ShadowParams` takes any `brawlers.yaml` block. The dash, long-dash, gadget,
    Berserker and super fields default to 0 / `BOLT`, as the sim resolves an absent key, and
    `charge_super` reads `super_type: CHARGE`. A kit without `attack_cooldown` is still refused.
  - **The volley.** A fire with `dash_distance: 0` spends one ammo and sets `attack_cd`, the
    sim's `is_other_attack` branch. No dash, and facing does not move.
  - **The charge.** `dash_t` = 0.20 s windup + 11 / 6.7 s rush = 1.84 s (37 ticks). The windup
    stands still. Attack, super and gadget are all masked until the landing (Q5). The landing
    starts `slow_window` at 1.5 s. **Always full length** (Q6, and live charges break walls). D3's
    wall stop is the sim's only, so water or the map edge is where the shadow's landing comes
    late, on the safe side.
  - **Berserker.** The reload gain doubles while HP < 60 % of max HP. Live there is no cube
    reader, so max HP is the most HP the loop has read this match (`set_hp`, every decision),
    which lags after a cube, also on the safe side.
  - **Snapped countdowns.** `attack_cd`, `gadget_cd` and `slow_window` use `status.count_down`,
    as the sim does. `invuln_t` keeps the plain clamp. Mortis's 0.35 / 18.0 come out the same
    either way, and his parity tests did not move.
  - **The live super drag** (Q6). `Buttons.press(..., full=True)` drags the super to the screen
    edge along the bearing, 2 px inside it. Not off-screen: the ADB backend clamps one axis at a
    time, which would bend the bearing. The loop passes the shadow's `charge_super`, so Mortis's
    drag is unchanged. An idle super stays a bare tap, and the shadow charges along `facing` for
    it, which is the game's pick only with no enemy in reach (docstring'd).
  - **Wiring.** `loop.SHADOW_KINDS` adds `hero_bull`. `_decide` hands the HP read to
    `shadow.set_hp`. `assemble._put_self` supplies `hero.slow_window`. `screen` is carried
    through `Controls.build`, `with_backend` and `replay_clip.build_loop`.
  - **Tests.**
    - Shadow vs sim field for field over both heroes, at action_repeat 1 and 5.
    - Bull's charge parity: two charges, along a bin and on the idle bin, with fire asked every
      decision.
    - Berserker parity across both HP crossings.
    - Gadget parity over both heroes (now 440 decisions at 20 Hz, past Bull's 20 s).
    - Bull's second Stomper at decision 81 at 4 Hz.
    - The assembler's deploy6 parity on Bull's spec, with a charge fired.
    - A Bull loop end to end: the edge drag, the mask, `slow_window` in the observation.
    - The edge drag over 19 bearings.
    - The deploy-spec coverage tests now name `agent_obs_bull.yaml`.
    - Halving Berserker's multiplier, or shortening the slow or the charge by 3 %, fails them.
- [ ] **c. `measure_reload`** writes `kit_timing_<hero>.json`.
- [x] **d. The hypercharge gate. SETTLED 2026-11-01: the gate holds and nothing changed.**
  - Footage: your real match (`Videos/2026-11-01 10-10-07.mp4`, copied to
    `tests/fixtures/vision/bluestacks-bull-hypercharge.mp4`). The hypercharge fills, lights at
    70 s, and stays lit through the 1st-place finish.
  - Every frame, through `MatchState` refined as the loop refines it (r 35.9 on the dull disc),
    the gate makes exactly one enter/exit pair:
    - **The 10-28 entry below was wrong twice.** The disc does have a meter, a magenta arc that
      fills around it: half at 30 s, ~85 % at 69 s. Each step lowers the dull score, 0.98 → 0.89
      → 0.80.
    - Lit, it holds 0.68 (median, 15 s), not the 0.17 seen in Training Grounds. There the disc
      sits on bright grass with no dark track around it, which is why that venue is invalid for
      the gate.
    - The frame it lights dips to **0.24 for 3 frames (100 ms)**. That is the only in-match dip,
      one sample short of `exit_samples` 4. Live samples come every 50 ms, so it reaches the gate
      at most 3 times too.
    - The results screen reads 0.000.
  - The policy never presses the hypercharge, so the lit look is the last one a deployed match
    shows. The "used" look never occurs live.
  - `test_gate_holds_while_the_hypercharge_fills_and_lights_on_real_footage`
    ([test_deployment_control.py](tests/test_deployment_control.py), `vision`, 18 s) pins the
    transitions, the dip against `exit_samples`, and the lit and dull margins. With
    `exit_samples` 3 it fails ("4 transitions").
  - **Open:**
    - Mortis's lit disc is unmeasured, but it is the same frame with his face.
    - A gate first refined on a lit disc fits r 43, where the dull disc scores under 0.45, so it
      would never open on the next match. That only happens when the loop starts mid-match, which
      deploy_run's docstring already forbids.
  - The 10-28 entry, kept for the record:
  The hypercharge cannot be unequipped (Q10), and **the disc is
  not inert for either hero.** There is no moving meter: the disc has two looks. It is dull purple
  while the hypercharge builds and after it is used, and bright magenta with a lightning bolt
  while it is ready. The gate needs one look all match. Measured 2026-10-28 (ring score vs radius,
  threshold 0.45; scripts in the session scratchpad):
  - **Mortis, 16 real matches** (`Videos/2026-10-13*`, `2026-10-14*`): the disc stayed dull in
    every match (8–80 s long), scored 0.989 at r 35.9, and the gate never exited. No live match
    has lasted long enough to light it, so the doc's "inert" guarantee has held by luck.
  - **Bull, two Training Grounds clips:** lit (10-23), the best radius is 46 at 0.74, and r 36
    gives 0.17. Just used (10-28), the best is r 48 at 0.59. Whichever look `refine()` fits,
    the other has little or no margin. Fit on the dull disc, 74 % of the lit frames fall under
    the threshold. Fit on the lit disc, the dull one bottoms out at 0.452.
  - So a Mortis match that lights the hypercharge probably stops the agent too: the lit disc
    scores 0.17 at Mortis's r 36. That is unmeasured on Mortis.
  - Fix options, none measurable without footage: score every look the disc has, each at its own
    fitted radius, and take the best; re-anchor on an element that never changes; or a
    2-of-N vote across the cluster's discs. Each needs **a real-match recording where the
    hypercharge lights and is used**, results screen included. Training Grounds is not valid for
    the gate (live-test venue rules). Re-measure with the operator naming what they see (the
    09-22 lesson).
- [ ] **e. Live checks:**
  - the entity detector does not box the hero Bull as `enemy`;
  - the ground-ring hero locator holds through a 6.7 tiles/s charge;
  - an idle super goes to the right disc as a bare tap;
  - **the edge drag gives a full-length charge in every direction** (7b). Down is the short one.
    On the shipped calibration the super button (1559.9, 901.1) sits 178 px above the bottom
    edge, so a straight-down drag gets 176 px, down-left and down-right 191, and right 357; up
    and left get 900 or more. Mortis's drag is 75. If the game's super stick saturates further out
    than 176 px, downward charges come up short and the shadow's landing is early;
  - the ammo canary stays quiet through Berserker: below 60 % the clip refills twice as fast, and
    a missed HP read or a cube (max HP the shadow has not seen yet) shows up there first. **On the
    real app this check does not apply (2026-11-04):** the account's Bull has Stomper and no
    Berserker, so the shipped `loadout.star_power: false` turns the shadow's boost off (design
    §5.2), and the canary there checks the base-rate reload at every HP. Nulls replays keep it on
    (`replay_clip.py --star-power on`).
  - **Found offline first (2026-11-02):** replaying the real-match clip through the deploy loop
    with the fresh 300M run, the hero's HP went unread whenever it showed `10400` (full HP with
    one cube). The `4`'s arm touches the `0` before it, the pair segmented as one blob too wide
    for a glyph, and the gap test rejected the row. That cost 14 s of skipped decisions at full HP.
    `hp_detection/locate.py` now cuts such a pair at its thinnest column. The replay went from
    67 `no hero hp` skips to 7 (popups), and from 240 decisions to 300.
  - **Found from the first live match (2026-11-02): the live super was masked all match.** The
    super was read off a bar under the hero's ammo pips (`hero_bars.read_super`). That bar is
    Mortis's long-dash meter, not his super, and Bull has none, so the reader read None on every
    frame and the shadow held the super illegal: legal on 0 of 155 decisions in the 2026-11-01
    telemetry, while the sim run uses it 2.77 times a match. The super is now read off the Super
    button (`brawl_deployment/perception/super_button.py`, BRAWL_DEPLOYMENT_DESIGN.md 6.4). On
    the real-match clip it reads ready from 21.9 s to the spend at 79.8 s. The old reader is
    removed. What it means for Mortis: every deploy before this fed his long-dash readiness into
    `hero.super_ready` and `super_charge_frac`.

### 5.8 Miss penalties and a 100M fine-tune: **BUILT 2026-11-02**; both fine-tunes **TRAINED** by you 2026-11-03

The live matches showed shots from out of range and stomps with nobody near. Sim matches of the
300M run show the same thing, so it is learned behaviour, not a deploy fault. Measured on 128
elite episodes (`final_model.zip`, stochastic): 19.5 volleys a match, 5.7 of them landing nothing
(29 %), and 2.22 stomps, 1.96 of them on nobody (88 %). The user's call: a small cost for each,
larger for the gadget. Neither changes the sim; both are reward terms (BRAWL_SIM_DESIGN.md §10).

- **`attack_miss` = −0.05** for a volley none of whose pellets landed on a player or a crate (the
  user's "or power box"). It is charged once per volley when its last pellet ends, up to 12 ticks
  after the shot. Pellets are tracked per volley, keyed by spawn tick (`state.hero_volley_hit`),
  because a 12-tick flight and a 5-tick cooldown put two volleys in the air at once.
- **`gadget_miss` = −0.15** for a stomp that landed on nobody (`env._stomp_phase`).
- `RewardConfig` defaults both to 0, and Mortis never pays them. Tests:
  `tests/test_miss_penalties.py`.
- **Digests.** Mortis is unchanged at 48e1a2b59a31e97d. Bull's moves to b8b6b69df80c322d through
  the reward alone, because the probe prices configs/train.yaml's rewards: no tensor moved, and
  with both weights zeroed the rollout gives the old 5d4250ff5c7ac420 bit for bit.

**The fine-tune** (yours to launch). It starts from the run's root `final_model.zip` with its
`final_vecnormalize.pkl`, both written 2026-11-02 06:08. It runs 100M steps at the run's own
floors, a constant LR of 5e-5 and clip of 0.1, on the run's own `train.yaml`, which carries the Bull
overrides that configs/train.yaml does not. `--curriculum-stage 4` keeps it at elite: the root
zip's `curriculum.json` lookup misses, and without the flag the run would restart at stage 0.

```bash
.venv/Scripts/python.exe scripts/train.py --config runs/mortis_ppo-20261101-133724/train.yaml --resume runs/mortis_ppo-20261101-133724/final_model.zip --restart-schedules --curriculum-stage 4 --set run.total_timesteps=100000000 --set learning_rate.initial=5e-5 --set clip_range.initial=0.1 --set reward.attack_miss=-0.05 --set reward.gadget_miss=-0.15
```

The eval runs at step 0 (`eval.at_start`), so the inherited policy's six-tier score, .727, is
re-read on the same seeds as the first point of the new curve.

**Result (read 2026-11-03): `runs/mortis_ppo-20261102-215433`, best = final at 400.5M.**
- Its own evals: six-tier .739 at step 0 (the inherited weights) to .771, holdout .698 to .709.
  The step-0 eval came out .012 above the 300M run's last eval of the same weights, and elite
  .067 above, so one eval's noise is about that size.
- Head-to-head, both final models on the same seeds, 600 episodes per tier, deterministic:

  | | six-tier | holdout | expert | elite | elite holdout |
  |---|---|---|---|---|---|
  | 300M final | .732 | .674 | .603 | .468 | .390 |
  | fine-tune final | **.775** | **.701** | **.675** | **.540** | **.443** |
  | gain (2 SE) | +.043 (.019) | +.026 (.020) | +.072 (.055) | +.072 (.058) | +.053 (.057) |

- Whiffs at elite, 256 episodes, deterministic (stochastic in brackets):
  - Stomps per match are unchanged at 2.46 (2.55); he still stomps about whenever it is charged.
  - Stomps that catch someone rose from 0.39 to 0.59 a match (0.27 to 0.45), a hit share of
    16 % to 24 % (11 % to 18 %).
  - Volleys that land nothing barely moved: 4.39 to 4.28 a match, a share of 22.4 % to
    21.3 % (32 % to 29 %). At −0.05 the far shots stay.
  - Damage dealt rose from 44.4k to 48.2k a match.
- What it cannot say: how much is the miss terms and how much is 100M more steps at the
  floors. The 300M run had flattened (.722 to .727 over its last 60M), and the stomp hit share
  is a direct effect of `gadget_miss`. There was no control run, so the split stays unmeasured.

**Second fine-tune, `attack_miss` −0.10 (prepared 2026-11-03, TRAINED by you 2026-11-03).** You asked to
raise `attack_miss` after the first fine-tune left the far shots in place. It is doubled, not
tripled, so the stomp's −0.15 stays the larger cost (your rule from 2026-11-02). At −0.05 a whiff
with an enemy in reach netted exactly 0 against `attack_in_reach`; at −0.10 every whiff costs.
It resumes the first fine-tune's final model. That run's saved `train.yaml` already holds the
100M length, the 5e-5 / 0.1 floors and `gadget_miss` −0.15, so only `attack_miss` is overridden:

```bash
.venv/Scripts/python.exe scripts/train.py --config runs/mortis_ppo-20261102-215433/train.yaml --resume runs/mortis_ppo-20261102-215433/final_model.zip --restart-schedules --curriculum-stage 4 --set reward.attack_miss=-0.1
```

Its baseline is the head-to-head above: six-tier .775, holdout .701, elite .540, and a 21.3 %
volley whiff share (deterministic, elite). Read its result head-to-head against
`runs/mortis_ppo-20261102-215433`'s final, not from the step-0 eval.

**Result (read 2026-11-03): `runs/mortis_ppo-20261103-092146`, final at 500.7M.**
- Its own evals were flat: six-tier .757 at step 0 to .767, holdout .727 to .696. Its
  `best_model.zip` (460M, .783) is one noisy eval's pick, not a better model.
- Head-to-head of the two fine-tunes' final models on the same seeds, 600 episodes per tier,
  deterministic. The first fine-tune's train-map row reproduced its earlier head-to-head exactly,
  so this eval is deterministic.

  | | six-tier | holdout | expert | elite | elite holdout |
  |---|---|---|---|---|---|
  | 1st fine-tune (−0.05) | .775 | .701 | .675 | .540 | .443 |
  | 2nd fine-tune (−0.10) | .776 | **.722** | .662 | .555 | .455 |
  | change (2 SE) | +.001 (.018) | +.021 (.020) | −.013 (.054) | +.015 (.057) | +.012 (.057) |

  On train maps nothing moved. Holdout rose on all six tiers, and the mean sits right at 2 SE.
- Whiffs at elite, 256 episodes, deterministic (stochastic in brackets):
  - Volleys that land nothing: 4.28 to 4.10 a match, a share of 21.3 % to 20.1 % (28.8 % to
    27.2 %). Across all three runs it reads 22.4 / 21.3 / 20.1 % (32.2 / 28.8 / 27.2 %).
  - Volleys per match barely moved, 20.0 to 20.4 (22.9 to 22.7). Shots with an enemy in reach
    rose from 9.12 to 9.54.
  - Stomps 2.46 to 2.56 a match. Catches 0.59 to 0.66 (0.45 to 0.51), so the miss share went
    from 75.6 % to 73.6 % (82.3 % to 79.8 %).
  - Damage 48.2k to 51.7k a match (46.1k to 47.1k).
- Reading: doubling the cost bought about one more point of whiff share and no win rate on train
  maps. The far shots that remain look like shots worth their price. A pellet pays 0.264
  (`damage_dealt` 3e-4 × 880), so a volley whose hit chance is above roughly 1 in 4 still pays for
  a −0.10 miss. Raising the cost further only trims the marginal shots. As before, there is no
  control run, so the extra 100M steps and the weight change are not separated.

---

## 6. Risks

- **A long blind window.** 7 decisions per charge in which the policy's moves do nothing. Same shape
  as the dash, only longer.
- **Walls stop the sim's charge (D3 C, your call).** Live, his charge goes through walls and
  destroys them, so the sim-trained policy undervalues a charge through cover. A live charge it aims
  at a wall also carries him past the point where the sim would have stopped him. The broken walls
  stay in the occupancy map and in known-map localization. That already happens with enemy Bulls
  today.
- **Fan width (Q9).** If the real spread is wider, bot Bull is also mis-specified.
- **Changing `bot_bull`'s speed (D7)** changes the roster the Mortis run trained against, so new
  Mortis evals stop being comparable to the old ones. That happened 2026-11-01 for his range,
  pellet speed and walk: evals from now on face a different bot Bull than deploy6's .424 and 6c's
  .825 did.
- **Hypercharge (5.7d).** Silent if missed: a wrong gate label went unseen for two weeks in
  September.
