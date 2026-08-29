## This file contains character conventions for all of the brawl stars characters being displayed

Stats below are sourced from [configs/brawlers.yaml](configs/brawlers.yaml), which grounds each sim
entity in a real Brawl Stars brawler's datamined stats (Power 11 / max level, Showdown-boosted
numbers). **Where a real number and a "feels better" number disagree, the real number wins** — the
agent is here to learn the Brawl Stars environment, so an accurate roster matters more than an even
one. Some characters are simply stronger than others, and that is a fact to learn rather than a bug
to tune out.

When each brawler fires and how it aims is no longer per-character Python. It is five per-kind
fields read by [brawl_sim/bots/combat_rules.py](brawl_sim/bots/combat_rules.py) — `fire_needs_los`,
`fire_range_fraction`, the lateral-hold pair, and `aim_model` — so no brawler needs a Python
module of its own. Movement is orthogonal and comes from the entity's `Person`, in
[brawl_sim/bots/personality.py](brawl_sim/bots/personality.py).

That covers *when* a brawler shoots, not *what its weapon does*. A brawler whose weapon is a new
mechanic still needs engine work: Edgar's lifesteal is a function in `core/combat.py`, and Spike's
six-arm ring turned a hardcoded four-direction constant in `core/projectiles.py` into a per-kind
`split_count`. Both were additive — no existing brawler's behaviour changed — but neither was
config alone.

**Everything in this file is implemented.** See "Still unimplemented" at the bottom for what is not.

---

### Global rules

- **Reload pauses while attacking.** For `attack_cooldown` seconds after an attack, an entity can
  neither attack again nor accrue ammo. One field, both effects — there is no separate "attack
  animation" concept. Sustained fire is therefore `attack_cooldown + reload_seconds` per shot, not
  `reload_seconds`. The 0.25s floor absorbs the real game's short attack frames plus the "+0.1s
  after" rule into a single number; Mortis (0.35s) and Buzz (1.00s) need genuinely longer.
- **Power cubes** give **+10% damage and a flat +400 max HP** each, the real game's numbers. Flat HP
  means cubes narrow the HP spread rather than widening it: every brawler gains the same amount.
  Capped at `max_cubes: 16`.
- **Out-of-combat regen**: after 3 seconds with no attacking and no damage taken, health regenerates
  at 13% of max HP per second. Any attack or hit resets the clock.
- **The zone** deals 20% of each entity's max HP per second, so nothing survives more than ~5s in
  it regardless of HP or cube count.

---

### 1. Mortis (hero) — `hero_mortis`

- Health: 8000
- Attack damage: 2000 per dash hit
- Attack type: melee dash (no ranged projectile; the "Attack" input drives `start_dash()` directly)
- Dash distance: 2.67 tiles · duration 0.30s · radius (hit width) 0.70 tiles
- Move speed: 2.73 tiles/s
- Ammo: 3 max, **reload 2.25s**
- Attack cooldown: 0.35s — must exceed `dash_duration` 0.30, since a dash *is* his attack
- **Long dash**: after **4.5 seconds without attacking**, the next dash reaches **2× as far**
  (5.34 tiles). Only the range doubles — hit width and damage are unchanged — so a charged dash
  covers twice the ground in the same 0.30s and is therefore twice as fast, which makes it an
  escape tool as much as an engage. The agent sees both `hero.long_dash_ready` (the flag) and
  `hero.long_dash_frac` (the 0→1 charge).
- **Super**: 10-tile range, aimed by the same logic as his normal attack. A 0.70-radius projectile
  travels in a straight line, **passes through walls**, pierces every player it touches for **1800
  damage**, and **heals Mortis 1800 per player hit**. Charged by landing **5 hits on other
  players** — box hits never charge it. Masked out of the action space until ready; the agent sees
  `hero.super_ready` and `hero.super_charge_frac`.

### 2. Brock — `bot_sniper` (was Nani)

- Health: 6000
- Attack damage: **2320** per rocket (a single rocket, not a collapsed 3-pellet volley)
- Attack range: 8 tiles
- Projectile: `SNIPER_BOLT`, **speed 5.33 tiles/s** — 8 tiles in ~1.5s, the *slowest* straight-line
  shot in the roster and the one the agent can most readily see coming and sidestep
- **Lingering sphere**: wherever the rocket dies (wall, unit, box, or range expiry) it leaves a
  **0.75-radius sphere** that **ticks twice, at 2s and 4s**, dealing **30% of the rocket's damage**
  (696 at the shipped 2320) to anything standing in it. It ignores walls, stacks with other
  spheres, and never damages its own owner. Expressed as a *fraction* so it scales with power
  cubes and difficulty tiers. This roughly doubles Brock's sustained output.
- Move speed: 2.40 tiles/s · Ammo: 3 max, **reload 2.1s** · Cooldown: 0.25s
- Fire rule: needs line-of-sight, fires at any range it can reach, LEAD aim (0.06 rad noise,
  0.9 lead) · prefers to hold 0.85× range when kiting

### 3. Grom — `bot_artillery` (was Dynamike)

- Health: 6000
- Attack damage: **2080** on the landing tile, **2080** from any one split shard — 2080 if you are
  standing where it lands, 2080 if an arm catches you, and never both (the shards start beyond the
  blast rim, so the two regions do not overlap)
- Attack range: 7.33 tiles
- Projectile: `ARTILLERY_SHELL`, a **timed lob** — it lands **1.25s** after firing regardless of
  distance, not at a constant speed. Closing on Grom therefore does *not* shorten your reaction
  window.
- AoE: 0.6 tiles (the landing tile only), then **4 shards along the world axes taking 0.5s** to fly
  their 1.2-tile arms — the cross reaches ~3.2 tiles and spans ~6.4 tip to tip, and takes 10 ticks
  to form, so it can be seen and stepped out of
- Move speed: 2.57 tiles/s · Ammo: 3 max, **reload 2.3s** · Cooldown: 0.25s (he can throw again
  while a shell is still airborne, which is exactly how Grom plays)
- Fire rule: **no line-of-sight requirement** — an arcing shell does not need a physical line, and
  this is the archetype's whole identity. LOB aim: positional noise (0.8 tiles) on the *landing
  point* rather than angular noise on the bearing.

### 4. Buzz — `bot_melee` (was El Primo)

- Health: 10000 — the tankiest in the roster
- Attack: **five hitscan cones fanned across 70° (1.222 rad)**, one every 0.2s over a full second,
  each dealing **840** damage. Consecutive cones overlap slightly, but at `attack_arc_rad: 0.65` at
  most **3** can connect — max 2520 for a full sweep. One ammo buys all five.
- Attack range: 2.67 tiles (the same as Mortis's dash)
- Move speed: 2.57 tiles/s · Ammo: 3 max, **reload 1.0s**
- **Attack cooldown 1.00s** — by far the largest in the roster, and the one place the "no attack, no
  reload" rule really bites. The sweep *is* the cooldown. This is what makes a five-hitscan attack a
  genuine commitment rather than free damage, and it is why Buzz has the lowest sustained DPS of the
  four bots despite the highest health. That trade is the real character; it is not a balance bug.
- He can move during the sweep but stays facing the direction he started it in.
- Fire rule: no LOS gate (the hitscan resolves its own per-victim ray), requires the target inside
  the swing arc, DIRECT aim (nothing in flight to lead)

### 5. Shelly — `bot_rifle` (was Bo)

- Health: 7800
- Attack: **5 pellets** in a radial shotgun spread of **34.4° total** (±17.2°), each dealing
  **600** damage — 3000 if every pellet lands, which only happens inside ~2 tiles
  - This used to read "60° total (30° either side)", which was wrong in both docs and in
    `brawlers.yaml`'s own comment: `proj_spread_rad: 0.6` is **0.6 radians**, and the number was
    misread as a degree figure. The value was hand-tuned, so the config was right and the prose was
    wrong. The field is the TOTAL tip-to-tip span, not a half-angle.
  - The fan stays *continuous* over its whole length — at 8 tiles the pellets are 1.18 tiles apart
    and each covers 1.40 — so the hold-fire rules below are about a target having time to move out
    of it during the ~1.25s flight, not about gaps opening up in it.
- Attack range: 8 tiles · Projectile: `RIFLE_ARROW`, **speed 6.4 tiles/s** (~1.25s to cross the
  full range), radius 0.25
- Move speed: 2.40 tiles/s · Ammo: 3 max, **reload 1.75s** · Cooldown: 0.25s
- Fire rule: the only brawler that uses every field. Needs line-of-sight; **holds fire past 0.9× of
  its range** (the fan has diverged too far to land); **holds fire on targets crossing faster than
  2 tiles/s beyond 0.6× range** (they slip between the pellets). Inside 0.6× the fan is tight
  enough to land on a mover anyway. LEAD aim, 0.08 rad noise, 0.7 lead.

### 6. Edgar — `bot_edgar`

Named for the brawler rather than a role: there is no label that separates him from Buzz (both are
point-blank multi-hit hitscan) without inventing a distinction the sim does not make.

- Health: 7400
- Attack: **two hitscan cones 0.25s apart**, each dealing **1080** — 2160 for a full combo. One ammo
  buys both.
- Unlike Buzz's sweep, the two cones fire at **the same angle**: whatever direction he was facing
  when he triggered the attack (`hitscan_sweep_rad: 0`). He can walk during the combo, but his
  facing is frozen for its duration, so the second hit cannot be steered onto a target that dodged
  the first. It is still genuinely two attacks — each re-tests the cone and its own line-of-sight
  0.25s apart, so a target who steps out between them costs Edgar half the combo.
- **Lifesteal: 35% of the damage he deals**, per hit. Landing one hit of two heals half as much
  (378, not 756). **Players only** — hitting a loot box heals nothing, and damage an invulnerable
  (dashing) target never took heals nothing either.
- Attack range: 2.00 tiles · arc 0.50 rad (~29°) — the shortest reach in the roster, a focused
  forward slash rather than Buzz's 70° fan
- Move speed: **2.73 tiles/s** — tied with Mortis for the fastest in the game, which is what lets a
  2-tile range function at all
- Ammo: 3 max, **reload 1.25s** · **Attack cooldown 0.50s** (the combo animation). Because reload is
  frozen during the cooldown, sustained rate is 0.50 + 1.25 = 1.75s per combo.
- Fire rule: no LOS gate (the hitscan resolves its own per-victim ray), requires the target inside
  the arc, DIRECT aim

### 7. Spike — `bot_spike`

Also named for the brawler: mechanically he is a shell that splits on landing, which is Grom's
description too, and the two play nothing alike.

- Health: 6000
- **The shell itself deals 0 damage.** All of it comes from the spikes.
- Attack range: 8 tiles · Projectile: `SPIKE_SHELL`, **constant speed 5.33 tiles/s** — 8 tiles in
  ~1.5s, exactly Brock's rocket speed. Unlike Grom's *timed* lob, a short throw lands proportionally
  sooner, so closing on Spike **does** shorten your reaction window.
- On landing it splits into **6 spikes in a hexagonal ring**, each flying **4 tiles in 0.6s**
  (6.67 tiles/s) and dealing **1080**. The ring is world-oriented, so two shells landing on the same
  tile from opposite corners of the map produce the same star.
- Geometry, and why he hurts: the star reaches **~5.4 tiles** and spans ~10.8 tip to tip — by far the
  largest area in the roster. Adjacent arms are 60° apart, so their sweeps overlap only within
  ~1.4 tiles of the landing point. Inside that it is solid ground cover; outside it is six separate
  lines with dodgeable gaps between them.
- What that means in practice: **a shell landing *next to* you hurts more than one landing *on*
  you.** A near miss catches you in 2–3 arms (2160–3240); further out, a single arm is 1080. Three
  is the structural maximum — the spikes spawn one hit-radius out from the landing point, so no
  target is ever inside more than three of them.
- Move speed: 2.40 tiles/s · Ammo: 3 max, **reload 2.25s** · Cooldown: 0.25s (he can throw again
  while a shell is airborne)
- Fire rule: **needs line-of-sight**, which is the one place he differs from Grom. Spike's cactus
  does not arc over walls in the real game, and the projectile class that gives "no damage in
  flight, resolves at a landing point" has no wall collision to offer — so the LOS gate is what
  keeps a Spike bot from lobbing through cover. It is a fire *rule*, not physics: a shell already in
  the air when its target ducks behind a wall still lands. The **spikes** are ordinary projectiles
  and walls stop them normally. LOB aim, 0.7 tiles of positional noise on the landing point.

### 8. Bull — `bot_bull`

The roster's second shotgun, and the first brawler added since the archetypes went data-driven that
needed **no engine change at all** — he reuses Shelly's pellet fan exactly and differs only in
numbers. Named for the brawler because "shotgun" is already Shelly's role.

- Health: **10000** — ties Buzz for the tankiest frame in the game
- Attack: **5 pellets** in a **28.6° total** spread (±14.3°), each dealing **880** — **4400** if the
  whole volley lands, the largest single burst in the roster
- Attack range: **7 tiles**, one tile shorter than Shelly
- Projectile: `BULL_SLUG`, **speed 6.4 tiles/s** — the same pellet speed as Shelly, so his shorter
  range also means a shorter time to target (~1.09s against her ~1.25s). Closing on Bull is worse
  than closing on Shelly.
- **Tighter fan than Shelly** (0.5 rad against her 0.6). At 7 tiles his outermost pellets are
  3.46 tiles apart — 0.87 between neighbours against the 1.40 each one covers — so his volley
  overlaps more heavily along its whole length. In practice: at 4 tiles and half a tile off-centre
  he still lands 3 pellets where Shelly is down to 2.
- Move speed: **2.17 tiles/s** (Speed=650 / 300, the **SLOW** tier) — one full step below Shelly's
  NORMAL 720, and **the slowest brawler in the roster**. This is the counterweight to everything
  above: the biggest burst in the game on the joint-tankiest frame, on a body that cannot close the
  distance to use it. Mortis and Edgar both move 2.73, 26% faster, so disengaging from Bull is
  always available.
- Ammo: 3 max, **reload 2.0s** (one ammo every 2 seconds) · Cooldown: 0.25s
- Fire rule: **identical to Shelly's** — needs line-of-sight, holds fire past 0.9× range and on
  targets crossing faster than 2 tiles/s beyond 0.6× range, LEAD aim with 0.08 rad noise and 0.7
  lead. Keeping her thresholds means the difference between the two bots is their *stats*, not a
  second set of hand-tuned heuristics.
- He does fight closer in, though: `desired_range_fraction` **0.6** against her 0.75, so a kiting
  Bull holds **4.2 tiles** where Shelly holds 6.0. A 10000 HP brawler should not be hanging back at
  the edge of a 7-tile shotgun.

---

### Still unimplemented

- **Gadgets, Star Powers and Hypercharges** — out of scope; see BRAWL_SIM_BUILD_PLAN.md Appendix A.
- **Supers for bots.** Only Mortis has one. Nothing about the mechanic is hero-specific — the charge
  counter, readiness test, projectile and lifesteal are all per-entity, and `super_charge_hits: 0`
  is what means "this kind has no super" — so giving a bot one is a `brawlers.yaml` block plus a
  fire decision in its combat rule.
- **Box HP** is a constant 3000. Real loot-box HP scales 4500–8500 with the lobby's average power
  level.

---

### Notes on the stat conversions (from configs/brawlers.yaml)

- HP / damage are Power 11 (max level, Showdown-boosted) = raw Power-1 CSV value × 2.
- `move_speed` = raw `Speed` / 300 (tiles/s).
- `attack_range` / `dash_distance` = raw `CastingRange` / 3 (tiles).
- `reload_seconds` = raw `RechargeTime` (ms) / 1000.
- `attack_cooldown` is a deliberate divergence from the real ~100–200ms `MsBetweenAttacks`: it is
  the no-attack, no-reload window described under Global rules, floored at 0.25s so a fired shot is
  actually dodgeable and the agent always gets at least one decision between its own shots.
- Not sourced from real data (hand-tuned for feel, not a CSV conversion): `dash_duration`,
  `dash_radius`, `proj_radius`, `aoe_radius`, `aim_noise_*`, `reaction_delay`,
  `lead_target_fraction`, `decision_period_ticks`, `super_proj_speed`, `melee_lifesteal_fraction`,
  and the `split_*` fields. Grom's `attack_range` is Dynamike's converted value — CHARACTER_DETAILS
  never specified one for Grom. Edgar's, Spike's and Bull's numbers were specified directly by the
  repo owner rather than converted from the CSVs, so their `split_count` / `split_distance` /
  `split_seconds` / lifesteal fraction / pellet damage are the spec, not a conversion. Bull's
  `move_speed` is an ordinary conversion like the rest (Speed=650 / 300, the SLOW tier).
- `proj_spread_rad` is in **radians** and is the **total** tip-to-tip span (`spawn_volley` lays the
  pellets from −spread/2 to +spread/2). Reading it as degrees is what produced the wrong "60°"
  figure for Shelly that stood in this file until Bull was added.
