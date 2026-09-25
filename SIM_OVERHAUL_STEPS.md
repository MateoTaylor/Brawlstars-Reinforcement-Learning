# Sim Overhaul -- Steps and Substeps

The execution checklist for [SIM_OVERHAUL_PLAN.md](SIM_OVERHAUL_PLAN.md). The plan holds the reasoning
(§0 decisions, §1 findings, §8 trade-offs, §9 revisits). This file holds the work: every plan step
split into substeps that one model can pick up, do, verify and tick, in order. Each step names the
plan section it comes from; go there when a choice looks odd, not to re-decide it.

## Conventions every substep assumes

- Python is `.venv/Scripts/python.exe` (bare `python` is not installed). Tests:
  `.venv/Scripts/python.exe -m pytest tests/<file>.py -q`. Never run the whole suite casually (~13 min).
- Run `graphify query "<topic>"` before grepping (a hook enforces it) and `graphify update .` after
  every substep that changes code.
- No git actions of any kind. The operator owns commits, branches and checkpoints.
- Hot path (`_run_tick` and everything it calls) is sync-free: no `.item()`, `.any()`, host loops,
  data-dependent shapes. Per-decision work may live in `env.step` outside `_run_tick`.
- Deploy parity: an observation field enters the deployable spec only if `brawl_deployment` can
  supply it (shadow proprioception or CV). Every new field below names its supplier.
- Settled and not to be touched: Mortis `attack_cooldown: 0.35`, the reload pause, aimed-drag attacks
  (no auto-aim for ordinary attacks), `sight_tiles: 14` as "in view".
- `(new)` after a path means the file does not exist yet. Line numbers drift; function names are the
  anchors.
- Tests that build their input from the same constant the code uses cannot fail. Pin numbers.
- The GPU may be held by a training run. Audits and verification run on CPU.

## Execution order

Steps in one wave are independent; a wave starts when its dependencies in earlier waves are done.

| wave | steps | notes |
|---|---|---|
| 1 | M1, B1, G1, A1, H1, M2 | all independent; M1/B1/G1 share `constants.py`, `config.py`, `brawlers.yaml`, so do them one after another, not concurrently |
| 2 | B2, B3, G2, A2, M3 | B2/B3 need B1; G2 needs G1; M3 needs M2 |
| 3 | G3, B4, M4 | G3 wires the gadget into the action space and needs G2 |
| 4 | G4, H2, (A3) | H2's attack one-hot is 4 wide only after G3; A3 is DEFERRED (operator, 2026-09-21), see A2.4 |
| 5 | I1, G5, H4, (A4) | I1 needs G4 + H2; the deployment mirrors need their sim halves green; A4 is deferred with A3 |
| 6 | H3, G6, I2 | H3 needs the deploy4 spec; G6 was a live measurement, DEFERRED until training completed (operator, 2026-09-21), DONE 2026-09-22; I2 needs B4 + M4 |
| 7 | I3, I4, I5 | docs, the suite, the run; I4.2's benchmark is REMOVED (operator, 2026-09-21) |

**Status 2026-09-21: waves 1 to 5 are complete** (each step carries a `DONE` note under its
heading with what deviated from the text and why). **The operator settled every open item on
2026-09-21; do not re-raise any of them:**

- **A2.4, A3 and A4 are deferred.** A1 found the sim policy itself holds fire (utilization 0.339,
  one fight decision in nine idling for the long dash). The operator accepts that as learned
  behaviour to tune later, so no live match is owed and no latch is built for this run. The lever
  taken instead is Step R2 (after A4): `reward.attack_in_reach: 0.05`, DONE.
- **M3.1's render review is approved** and the holdout pair `split_river` + `hollow_ring` is
  confirmed (M3, M4).
- **Pre-gadget `(17, 3)` checkpoints are retired.** Deployment's legacy acceptance is reverted
  (G3's amendment) and the operator deletes the old runs.
- **KITE bots hold no farther than their own fire reach** (B3's amendment, which closes B4's second
  review note).
- **`scripts/play_manual.py` gets no super key** and no further manual-play controls. Its module
  docstring records the decision; `g` (G3.3) stays as it is.
- **G6 and every deployment step wait until training is complete** (2026-09-21): G6.1's live
  trace, G6.2, and I5's dry run and deploy items. Training needs none of them, because the match
  gate reads the emulator's screen and the sim has no gate. G6 still lands before any live run
  of a checkpoint trained on this tree. **Lifted 2026-09-22**: the run is done (I5), and both
  G6.1's trace and G6.2 landed the same day. G6's own entry carries the numbers.
- **I4.2's benchmark is removed.** The operator watches `time/fps` during training instead;
  I4.2's note says what to read it against.
- **Attack, super and gadget stay one masked four-way choice.** The operator's rule, that
  choosing one of them rules out the others, holds by construction: `action[:, 1]` carries one
  value per decision, and
  `tests/test_hero.py::test_decode_action_yields_at_most_one_of_attack_super_gadget` pins it
  with all three legal at once. No priority order is needed, since the policy never picks two.

Wave 4 is complete: G4 and H2 are DONE (2026-09-21) and A3 stays deferred. Wave 5 is complete: I1,
G5 and H4 are DONE (2026-09-21), and A4 stays deferred with A3. Wave 6 is complete: H3 and I2 are
DONE (2026-09-21), and G6 is DONE (2026-09-22). Wave 7's checklist is DONE too: I4.1 on
2026-09-21, I4.2 removed, and I3 plus I5's run and cadence audit on 2026-09-22. What is left is the
live deployment work G6 unblocked -- I5's two unticked boxes and G6.2's trace-replay test -- and
nothing else in this file. Since I2 a bare `python scripts/train.py` trains deploy4.
Since H4, deployment serves `configs/agent_obs_deploy4.yaml`, and the two strict xfails that tracked
its refusal are gone (H4's note). Since G5 the deployed policy presses the gadget, which G6 found is
NOT the match gate's anchor and never was: all three button names in `control_calibration.json` sat
one disc off, so a commanded gadget pressed the Super and the gate sat on the Super too. The gate is
on `hypercharge` now, the one disc nothing presses. Since G3 the audit's `--run` mode refuses every
pre-gadget checkpoint, so A1's report (`runs/audit/cadence_mortis_deploy3_elite.md`) cannot be
regenerated; A1's table below is the record. No checkpoint on disk trains, evaluates, watches or
deploys against this tree any more (4-wide attack column, sixteen-map default). Two checklist file
names were wrong and are corrected in place: `tests/test_bots_policy.py` ->
`tests/test_bot_dispatch.py`, `tests/test_env_step.py` -> `tests/test_env.py`.

Cross-phase dependencies that are easy to miss:

- G3 before A3 and before H2 (attack column width 3 -> 4).
- G2's `charge_hit` return changes `step_projectiles`' tuple; anything else calling it (grep) updates in G2.4.
- I1 before H3 (the extractor test reads the spec).
- G6 before any live run with the gadget enabled, including the `control.backend: null` dry run's
  interpretation. DONE 2026-09-22.
- M3 before M4 (holdout maps must exist) and before I2.

---

# Phase A -- attack cadence (plan §2)

Know why the deployed agent attacks slowly before changing anything; then remove the sim-side 0.50 s
cap if the measurement says the cap is what binds.

Read first: plan §1.1 (the cadence arithmetic), `brawl_sim/env.py` `_run_decision` and `_held`,
`brawl_sim/core/hero.py` `action_mask`, `brawl_deployment/perception/shadow.py`,
`brawl_deployment/control/buttons.py`.

## Step A1 -- sim-side cadence audit

**DONE 2026-09-18.** Deviation: the recorder is called by the audit driver after each `predict`
(`CadenceRecorder.record(sim, action)`), not from `tick_hook` -- the hook gets no action, and the
statistics are per decision anyway. Report: `runs/audit/cadence_mortis_deploy3_elite.md`
(200 episodes, elite, deterministic `best_model.zip`, 8 envs, CPU, ~35k decisions, 542 fights):

| # | statistic | value |
|---|---|---|
| 1 | utilization P(attack \| legal & enemy in reach) | 0.339 (807 / 2382) |
| 2 | inter-attack interval mode | 10 ticks (141 of 454 intervals; long tail to 65) |
| 3 | phasing loss (ammo, 0 < cd <= 0.20 s) | 0.081 of fight decisions |
| 4 | long-dash waiting (legal, idle >= 4 s, no attack) | 0.114 of fight decisions |
| 5 | ammo at first attack of a fight (mean) | 2.41 (200 of 353 fights start at 3) |

Reading: the mode at 10 ticks is the structural cap showing up and the 8 % phasing tax is real,
but utilization 0.339 says the sim policy declines two of every three legal in-reach attacks, and
one fight decision in nine is spent idling for the long dash. The sim policy itself holds fire.
A3's gate clause "sim utilization high" is therefore NOT met by this audit; A2.4 decides with the
deploy numbers (if deploy intervals match the sim's, the conservation is learned, not a deployment
cadence bug). "In reach" is generous (any alive, visible enemy within dash + dash radius + unit
radius, any direction), so some of the 0.66 is legitimate spacing; the long-dash share is not.

Goal: five numbers that say whether the sim policy attacks whenever it can, and where the 0.50 s cap
shows up. Runs on CPU against the deployed checkpoint.

### A1.1 Recorder
- Files: `scripts/audit_attack_cadence.py` (new)
- Do: a `CadenceRecorder` usable as the env's `tick_hook` that, on decision boundaries only, appends
  per env: `can_attack`, `attack_cd`, `dash_t`, `ammo`, `long_dash_ready`, `attack_idle_t`, the
  attack column chosen, and `enemy_in_reach` = any enemy alive, revealed to the hero
  (`vis[:, 0, :]`), within `dash_distance + dash_radius + unit_radius` tiles. One host transfer per
  decision is fine here (audit, not the hot path).
- Verify: import the module and run it against a 4-env `BrawlVecEnv` for 20 decisions.
- Done when: the recorder holds one row per (decision, env) with all eight fields.

### A1.2 Statistics
- Files: `scripts/audit_attack_cadence.py`
- Do: `summarize(rows) -> dict` with (1) utilization `P(attack | legal & enemy_in_reach)`;
  (2) inter-attack interval histogram during fights (consecutive attacks with `enemy_in_reach`
  throughout), in ticks; (3) phasing loss = share of decisions with
  `enemy_in_reach & ammo >= 1 & 0 < attack_cd <= 0.20`; (4) long-dash waiting = share of fight
  decisions with `attack_idle_t >= 4.0 & enemy_in_reach & legal & no attack`; (5) ammo at the first
  attack of each fight. Write a markdown table plus one histogram text block per stat to
  `runs/audit/cadence_<run>.md`.
- Verify: A1.4's test.
- Done when: `summarize` returns the five keys with the documented meaning.

### A1.3 CLI
- Files: `scripts/audit_attack_cadence.py`
- Do: `--run runs/<name> --checkpoint best_model.zip --episodes 200 --tier elite --device cpu`.
  Build the env and policy the way `scripts/watch.py` does from the run's `train.yaml`; deterministic
  actions; fixed tier via the curriculum's fixed-tier path.
- Verify: `--episodes 2 --n-envs 8` completes on CPU and writes the report file.
- Done when: the report file exists and the CLI refuses a missing run dir with a clear message.

### A1.4 Test
- Files: `tests/test_audit_attack_cadence.py` (new)
- Do: feed a hand-scripted policy (attack whenever legal, move toward the nearest enemy) on a tiny
  env with one enemy in reach. Assert utilization == 1.0 and that the inter-attack histogram has
  mass only at 10 ticks (0.50 s). This is also the regression pin for the cap itself; A3 changes it.
- Verify: `.venv/Scripts/python.exe -m pytest tests/test_audit_attack_cadence.py -q`
- Done when: green, and the 10-tick pin is asserted explicitly (not derived from `attack_cooldown`).

### A1.5 Run it
- Needs: the deployed run dir `runs/mortis_deploy3_elite-20260913-015933`; CPU only.
- Do: run A1.3 for >= 200 episodes; paste the five numbers into the report's header.
- Done when: `runs/audit/cadence_mortis_deploy3_elite.md` holds the numbers.

## Step A2 -- deployment-side cadence telemetry

**DONE 2026-09-18 (A2.1-A2.3); A2.4 is the operator's.** `TickRow` gained **six** columns, not
four: `attack_legal` (bitmask, bit i = column i, so `& 0b10` = dash legal), `attack_cd_shadow`,
`enemy_in_reach`, `resync`, plus `attack_idle_t_shadow` (statistic 4 reads `attack_idle_t`, which
cannot be rebuilt from telemetry because the shadow advances on real elapsed time) and
`resync_error` (`Desync.error`, CV minus shadow in pips; `0.0` = no resync). All filled in `_decide`
before `policy.act` from the same `attack_mask()` call the policy gets; `resync`/`resync_error` are
set in `_read_own_bars` where `shadow.resync()` runs. New columns sit after `note`, so old CSV
column positions are unchanged; `TickRow.from_record` + `read_telemetry_csv` load pre-A2 files with
defaults. `enemy_in_reach` follows `tracked.enemies`, so an enemy counts one decision after it
first appears (track confirmation), and the reach is `dash_reach_tiles` = 2.67 + 0.70 + 0.40 =
**3.77 tiles** (the plan said ~3.9), read from `brawlers.yaml`/`default.yaml` because
`ShadowParams` carries neither `dash_radius` nor `unit_radius`. A2.3: `--telemetry <csv>` (exclusive
with `--run`) maps rows to `summarize`'s inputs (env 0; `step_count = 5 * decision slot`, slot
rebuilt from the `phase` column so a non-"playing" row starts a new episode; `ammo` = the shadow's
pre-canary clip; `can_attack = attack_legal & 0b10`), refuses a file without `attack_legal`, and
renders row 6 (resyncs per fight-minute + each error) into `runs/audit/cadence_<stem>.md`.
Rates come from the repo defaults (3, 5, 0.25 s), not a run's `env_overrides`; the header prints
them. Design-doc paragraph in section 7. Review fixes: `ammo_shadow` is now written first in
`_read_own_bars` (before the no-hero-box return) so a coasting-hero decision never carries the
`-1.0` sentinel next to a valid mask, and `telemetry_rows` refuses such a row; the audit CLI is
CWD-independent. Tests: `tests/test_deployment_loop.py` 68 (+10), `tests/test_audit_attack_cadence.py`
12 (+8). `tests/test_deployment_control.py::test_gate_separates_gameplay_from_menus_on_real_footage`
fails (ring-score floor 0.675 vs 0.7) -- it is a `vision`-marked real-footage test, pre-existing
and fed by nothing this step touched. Any telemetry for A2.4 must be recorded with the loop as it is
now (an older file can carry the sentinel and is refused by design).

Goal: the same five statistics computed from a live match's telemetry, on identical definitions.

### A2.1 TickRow fields
- Files: `brawl_deployment/loop.py` (`TickRow`)
- Do: add scalars `attack_legal: int` (bitmask of the shadow's legal tuple, bit i = column i),
  `attack_cd_shadow: float`, `enemy_in_reach: bool`, `resync: bool`.
- Verify: `tests/test_deployment_loop.py`
- Done when: rows serialize with the new columns (six shipped, see the DONE note); older telemetry
  files still load (missing columns default).

### A2.2 Fill them in `_decide`
- Files: `brawl_deployment/loop.py` (`_decide`)
- Do: encode `shadow.attack_mask()` into the bitmask; copy `shadow.attack_cd`; compute
  `enemy_in_reach` from `tracked.enemies` in tiles with A1.1's radius; set `resync` when this decision
  triggered `shadow.resync()`.
- Verify: `tests/test_deployment_loop.py` -- a decision with a legal attack and an enemy at 2 tiles
  yields `attack_legal & 0b10` and `enemy_in_reach`.
- Done when: the test passes.

### A2.3 Telemetry mode in the audit script
- Files: `scripts/audit_attack_cadence.py`
- Do: `--telemetry runs/deploy/<file>` computes A1.2's five statistics from TickRows, plus a sixth:
  resyncs per minute inside fights and the ammo error that tripped each.
- Verify: a unit test on a synthetic telemetry file with a known 0.50 s attack chain.
- Done when: the same report format is produced from telemetry.

### A2.4 Collect and decide
- Needs: live emulator, one real match (not Training Grounds).
- Do: play one match with A2.1/A2.2 in; run A2.3; put the sim and deploy tables side by side and
  read plan §2 Step A2's decision table. Record the verdict (cap binds / deploy-side / learned
  conservation) in `runs/audit/cadence_verdict.md`.
- Done when: the verdict file exists. A3 and A4 read it.
- **DEFERRED 2026-09-21 by the operator, with A3 and A4.** The sim policy's conservation (A1) is
  accepted as learned behaviour to tune later, so no verdict file is owed for this run. R2 is the
  lever taken instead.

## Step A3 -- the fire latch (sim)

Gate: A2.4's verdict says the structural cap binds (sim utilization high, deploy intervals ~= sim).
Depends on G3 (attack column 4-wide) being done first.

**DEFERRED 2026-09-21 (operator), see A2.4.** Kept as written for a later run; nothing in A3 is owed
now.

G3 is done (2026-09-21). The functions A3 edits changed shape: `hero.decode_action` returns
`(move_dir, fire, super_fire, gadget_fire)` and `env._attack_phase` takes `(..., gadget_fire, vis)`.
The latch gates `fire_ok` only; `gadget_ok` ignores `ready` by design, so the gadget stays legal
mid-dash and inside a latch window. `tests/test_gadget.py` pins that (mid-dash, empty clip,
`attack_cd` running); keep it green.

### A3.1 Config knob
- Files: `brawl_sim/config.py` (`EnvConfig`, `PER_ENV_FIELDS`), `configs/default.yaml`
- Do: `fire_latch: bool = True`; YAML key `hero.fire_latch: true` with a comment: "false reproduces
  the pre-2026-09 cadence (one attack per two decisions)".
- Verify: `tests/test_config.py`
- Done when: the key loads, and `false` is accepted.

### A3.2 State fields
- Files: `brawl_sim/core/state.py` (`_ENTITY_FIELDS`)
- Do: `ent_fire_latch (N,E) int64` (0 none, 1 attack, 2 super, 3 gadget), `ent_fire_latch_t (N,E)
  int64` (sub-ticks left). Only the hero row is ever written.
- Verify: `tests/test_state.py` (`check_invariants` includes the new fields, zero at reset).
- Done when: green.

### A3.3 Mask window
- Files: `brawl_sim/core/hero.py` (`action_mask`)
- Do: `action_mask(state, params, cfg)` computes `W = (cfg.action_repeat - 1) * cfg.dt` when
  `cfg.fire_latch` else `0.0`, and uses
  `ready = alive & (attack_cd <= W) & (dash_t <= W)`; `fire_ok = ready & (ammo >= 1)`;
  `super_ok = ready & super_ready`; gadget unchanged (S8).
- Verify: `tests/test_hero.py` -- with `fire_latch=False` the mask equals today's; with it on, a hero
  at `attack_cd = 0.15, dash_t = 0.05` is legal.
- Done when: both cases pinned.

### A3.4 Latch update and strict fire
- Files: `brawl_sim/env.py` (`_decode`, `_attack_phase`)
- Do, every sub-tick in `_decode`, verbatim from plan §2 Step A3:
  ```
  col = effective_action[:, 1]                 # _held zeroes this on sub-ticks 2..K
  set = col > 0
  latch   = where(set, col, where(latch_t > 0, latch, 0))
  latch_t = where(set, action_repeat, clamp(latch_t - 1, min=0))
  fire    = (latch == 1) & can_attack_now      # strict gates: cd<=0, dash_t<=0, ammo>=1
  super   = (latch == 2) & can_super_now
  gadget  = (latch == 3) & can_gadget_now
  ```
  `_attack_phase` zeroes `latch` and `latch_t` on rows that fired this tick. With `fire_latch` off,
  `latch_t` is forced to 1 so the press only ever fires on its own sub-tick (today's behaviour).
- Verify: A3.6's chain test.
- Done when: the chain test passes both ways.

### A3.5 Observation description
- Files: `brawl_sim/core/obs_schema.py`
- Do: `hero.can_attack`'s description becomes "legal to press this decision (cooldown and dash expire
  within the window when fire_latch is on)". `build_obs` must read the same mask function.
- Verify: `tests/test_obs_schema.py`; regenerate `docs/OBSERVATION.md` (I3 does the docs pass too).
- Done when: the observation's `can_attack` equals `fire_ok` from A3.3 in a test.

### A3.6 Tests
- Files: `tests/test_hero.py`, `tests/test_action_repeat.py`, the test that pins `_held` (grep
  `_held` in `tests/`), `tests/test_audit_attack_cadence.py`
- Do: (1) scripted hero attacking every decision with a target in reach dashes at ticks 0, 7, 15,
  22, 30 with the latch on and 0, 10, 20 with it off; (2) one attack per decision even with the bit
  held high; (3) a press whose cooldown does not expire inside its window never fires and the latch
  is 0 at the next boundary; (4) `test_action_repeat.py` invariance updated for the new timing;
  (5) A1.4's histogram pin becomes 7/8-tick alternation with the latch on.
- Verify: the four files.
- Done when: green, then `graphify update .`.

## Step A4 -- the fire latch (deployment mirror)

Gate: A3 done. Do not enable on a live run before A4.6.

**DEFERRED 2026-09-21 with A3.** Nothing in A4 is owed now.

### A4.1 Shadow latch
- Files: `brawl_deployment/perception/shadow.py`
- Do: `attack_mask()` uses `attack_cd <= W & dash_t <= W` with `W = latch_window_ticks * tick_dt`
  (a `ShadowParams` field, 0 disables); `act(move, attack)` arms a press instead of firing it;
  `_tick()` fires the modelled attack on the first sub-tick where the strict gate opens within
  `action_repeat` sub-ticks and drops it afterwards (log the drop).
- Verify: `tests/test_deployment_shadow.py` -- mirror A3.6's chain tick for tick.
- Done when: green.

### A4.2 Commit at lift
- Files: `brawl_deployment/perception/shadow.py`
- Do: `commit()` starts `attack_cd` and `dash_t` (and spends the ammo pip) at the moment the lift is
  emitted, not at the decision. `_tick` no longer starts them itself.
- Verify: shadow test -- timers are untouched until `commit()`.
- Done when: `hero.dashing`/`dash_t` in `observe()` describe the dash on screen (lift + 0).

### A4.3 Buttons defer the down
- Files: `brawl_deployment/control/buttons.py`
- Do: `press(action, bearing)` arms; `settle()` emits the down on the tick the shadow reports
  `fired`, then drag, then lift, and calls `shadow.commit()` on the lift tick. A press that never
  fires emits nothing.
- Verify: `tests/test_deployment_control.py` -- down deferred to the fire tick, lift two ticks later;
  a never-fired press emits no events.
- Done when: green.

### A4.4 Loop and config
- Files: `brawl_deployment/loop.py` (`_decide`, per-tick `settle`), `brawl_deployment/config.py`
  (`validate`), `configs/deployment.yaml` (`shadow.latch_window_ticks`, default 0 until A4.6)
- Do: wire the armed press through; `validate` rejects a window that exceeds `action_repeat`
  sub-ticks; telemetry logs dropped presses.
- Verify: `tests/test_deployment_loop.py` -- a latched-but-never-fired press sends nothing and logs.
- Done when: green.

### A4.5 Design doc
- Files: `BRAWL_DEPLOYMENT_DESIGN.md` §4.4, §6.3
- Do: append a dated paragraph each (latch mirror; commit-at-lift). Do not rewrite resolved items.
  Note the file stores `--` as an em-dash; anchor edits on headings.
- Done when: both paragraphs present.

### A4.6 Live measurement
- Needs: live emulator, real match.
- Do: with `latch_window_ticks` set to the latch window, a chain of six attack presses against a
  wall; count ammo drops through the ~1 s HUD lag over two clips. Expect six pips consumed, zero
  resyncs. If presses are dropped, keep the sim latch and leave `latch_window_ticks: 0` in
  `deployment.yaml` with a comment pointing at this measurement.
- Done when: the result and the chosen `latch_window_ticks` value are recorded in
  `runs/audit/cadence_verdict.md`.

## Step R2 -- the `attack_in_reach` shaping term (plan §9 R2)

**DONE 2026-09-21** (operator: "a small rwd for attacking when in range is reasonable"). One count
per decision, priced by `configs/train.yaml` `reward.attack_in_reach: 0.05`: the hero attacked
(column 1) or used its super (column 2) while an enemy it could see stood inside its uncharged dash
reach. The radius is the audit's own, `dash_distance + dash_radius + unit_radius` = 2.67 + 0.70 +
0.40 = 3.77 tiles, read off `_bot_phase`'s fair visibility (row 0), so the term pays for exactly
A1's utilization criterion. The gadget is not an attack for it, as in the audit. Wiring:
`env._attack_phase` computes the flag before `hero.start_dash` and returns it as a third value;
`_run_tick` returns six values; `_run_decision` sums the flag into an int32 count under the same
`live` gate as every other delta (0.1 s of action latency lands the attack on sub-tick 3 of 5, and
an attack that lands after a mid-decision truncation is not counted); `events.compute_info(...,
attacks_in_reach=)` publishes `info["attack_in_reach_tick"]` (zeros when absent, like `hp_healed`);
`ShapedReward` appends `attack_in_reach` to `TERM_NAMES`, last, so logged curves keep their order;
`RewardConfig.attack_in_reach` defaults to 0.0 so a hand-built config keeps its meaning. Sizing: a
twelfth of the 0.6 a landed 2000 hit pays, 1/60 of a kill, and about 3 over a whole 150 s episode of
in-reach attacks (some 55 at Mortis's 2.85 s sustained rate plus at most one super per five landed
hits) against 10 for a win. Blunt on purpose: a dash away from the enemy and a miss are paid too.
Tests: `tests/test_attack_in_reach.py`, 17, all literal (the 3.70-in / 3.84-out boundary; a super
counts, the gadget does not; hidden, dead and bot-only are False; one per decision with the hero
re-armed after every sub-tick; the latency pair; the price and `term_means`; train.yaml ships 0.05;
a real env pays 0.05 then 0; and 64 CUDA envs under `set_sync_debug_mode("error")`, 1920 of 1920).
`tests/test_events.py` +1; `tests/test_training.py`'s `_reward_inputs` carries the key. The number
to read at I5 is the sim audit's utilization on the new checkpoint against A1's 0.339.

---

# Phase B -- difficulty: accuracy + aggression (plan §3)

Bots scale on accuracy and aggression; HP and damage cap at 1.5x; at hard and above bots favor the
hero when it is in view.

Read first: plan §1.2, `brawl_sim/bots/perception.py` `select_target`,
`brawl_sim/bots/personality.py` `_select_mode` / `fire_allowed`, `brawl_sim/training/config.py`
`TIER_FIELDS`, `brawl_sim/training/curriculum.py` `_FLOAT_TARGETS`, `configs/train.yaml` curriculum.

## Step B1 -- two new per-kind params

**DONE 2026-09-18.** One deviation from the plan text: `aggression` is **not** required. Making a
missing key a validate error would fail every hand-built partial spec in the suite (`test_config.py`'s
inline `BRAWLERS_YAML`, `test_combat_rules`, `test_boxes`, `test_spike`, ...), so the file's own
convention won: a missing key resolves to 0 and **0 is neutral** (`hero_focus 0` = no preference,
`aggression 0` = read as 1.0 wherever it divides; see B3.1). `validate()` rejects only nonsense:
`aggression < 0` and `hero_focus` outside `[0, 1]`, both naming the kind. A separate test in
`tests/test_configs_files.py` pins that every shipped bot block authors both (read off the built
tensors, so a misspelled key fails too) and that the hero block authors neither. The tensors are
`(n_envs, n_kinds)`, like every other per-kind field, not `(n_kinds,)`.

### B1.1 Schema
- Files: `brawl_sim/config.py` (`PER_KIND_FIELDS`)
- Do: add `("hero_focus", "hero_focus", _F32)` and `("aggression", "aggression", _F32)` next to the
  existing bot-behaviour fields.
- Verify: `tests/test_config.py`
- Done when: `SimParams` has both tensors, shape `(n_envs, n_kinds)`.

### B1.2 Validation
- Files: `brawl_sim/config.py` (`validate`)
- Do: in the per-kind loop, after the `desired_range_fraction` block: reject `aggression < 0` and
  `hero_focus` outside `[0, 1]`, naming the kind. 0 is accepted for both (neutral).
- Verify: `tests/test_config.py` -- the inline spec (which authors neither) still validates;
  `aggression: -0.5` fails naming the kind; `hero_focus: 1.5` and `-0.1` fail; `1.0` / `2.5` and
  `0.0` / `0.0` pass.
- Done when: all cases pinned.

### B1.3 Authored values
- Files: `configs/brawlers.yaml`
- Do: `hero_focus: 0.5` and `aggression: 1.0` on each of the seven bot kinds, one comment naming the
  tier multiplier that scales them (B4), plus a glossary entry at the top of the file. The hero block
  gets nothing.
- Verify: `tests/test_configs_files.py` (every preset loads; `test_single_archetype_preset_pins_sniper`
  still passes; `test_every_shipped_bot_kind_authors_both_difficulty_axes`).
- Done when: green.

## Step B2 -- hero focus in targeting

**DONE 2026-09-18** (shipped together with B3; both edit `bots/policy.py`). `select_target(state,
vis, params, cfg)` is the plan formula verbatim with `_HERO_SLOT = 0` (the hero's entity slot along
the target axis) and a per-observer `gather_kind(params.hero_focus, ent_kind)` scale on column 0;
`prefer_hero` overrides stickiness. `focus = 0` multiplies by exactly 1.0, so the old
`where(current_valid, current, picked)` is reproduced bit-for-bit and the 35 pre-existing perception
tests pass unchanged. Callers updated: `policy.all_bot_intents`, `tests/bot_fixtures.build_targeting`
(shared scaffolding, outside the ownership list but a caller) and the nine calls in
`tests/test_perception.py`. Pins: 6 new in `tests/test_perception.py` (7 vs 4 at 0.5 -> hero; 9 ->
bot; sticky lock dropped the tick the hero appears at 7, kept at 9; 1.0 -> hero whenever visible
but not at 15 tiles; 0.0 -> lock survives a hero at 2 tiles; a concealed hero is never a candidate)
plus a dispatcher pin in `tests/test_bot_dispatch.py` (`tests/test_bots_policy.py` named below does
not exist). Note for B4: `select_target`'s scale is `1 - focus`, so the tier product must be clamped
to [0, 1] as B4.1 says -- a focus above 1 flips the hero's effective distance negative and argmin
always picks it. `BRAWL_SIM_BUILD_PLAN.md:910` still lists the three-argument signature
(historical doc, left alone).

### B2.1 `select_target`
- Files: `brawl_sim/bots/perception.py`
- Do: signature `select_target(state, vis, params, cfg)`. Verbatim from plan §3 Step B2:
  ```
  focus = stats.gather_kind(params.hero_focus, state.ent_kind).unsqueeze(-1)   # (N,E,1)
  scale = ones_like(dist); scale[:, :, HERO] = 1 - focus[..., 0]
  dist_eff = where(candidates, dist * scale, inf)
  nearest = argmin(dist_eff, -1); has_candidate = candidates.any(-1)
  picked = where(has_candidate, nearest, -1)
  prefer_hero = has_candidate & (nearest == HERO) & (focus[..., 0] > 0)
  target = where(prefer_hero, HERO, where(current_valid, current, picked))
  ```
  `focus = 0` must reproduce today's `where(current_valid, current, picked)` exactly.
- Verify: B2.3.
- Done when: existing `select_target` tests pass unchanged with `hero_focus = 0`.

### B2.2 Call sites
- Files: `brawl_sim/bots/policy.py` (`all_bot_intents`) and any other caller (grep `select_target`)
- Do: pass `params` through.
- Verify: `tests/test_bot_dispatch.py` (`tests/test_bots_policy.py` does not exist)
- Done when: green.

### B2.3 Tests
- Files: `tests/test_perception.py`
- Do: with `hero_focus = 0.5`: hero at 7 tiles vs bot at 4 picks the hero (3.5 < 4); hero at 9 keeps
  the bot (4.5 > 4); a sticky bot target is dropped for the hero the tick the hero becomes visible at
  a winning distance. With `1.0`: hero chosen whenever visible. With `0.0`: existing tests unchanged.
- Verify: the file.
- Done when: all four cases pinned.

## Step B3 -- aggression in the personality layer

**DONE 2026-09-18.** Helper is `core/stats.aggression_of(kind, params)` (the checklist's
`bots/stats.py` does not exist): gather + `where(a > 0, a, 1.0)`, the ONE place 0 is read as 1.0.
B3.1 `_select_mode(..., clearance, aggression, cfg)` with `hp_frac < clamp(0.35 / a, 0.05, 0.90)`
(`RETREAT_HP_FRACTION_MIN/MAX` public). B3.2 `policy.targeting`: `desired_range *= clamp(1 / a,
0.6, 1.4)` (`_HOLD_SCALE_MIN/MAX`). B3.3 `fire_allowed(state, tgt, aggression, cfg)`: silent iff
CAMPER and unseen and `a < CAMPER_FIRE_ON_SIGHT_AGGRESSION` (1.25, inclusive fires). **Deviation,
kept after review -- do not re-raise in B4:** the helper is called in three places (`targeting`,
`movement`, the `fire_allowed` site in `all_bot_intents`) rather than gathered once in `movement`
and passed down, because `targeting` runs before `movement` and `fire_allowed` after it; a single
gather would mean threading `aggression` through two public signatures with callers in three test
files, for two sync-free (N,E) gathers of saving. Note for B4: a tier that multiplies a 0 base stays
0 -> read as 1.0 (neutral); `brawlers.yaml`'s 1.0 bases (B1) are what make the multipliers bite.
Pins: 11 new in `tests/test_personality.py` (HUNTER 25% HP a=1.7 CLOSE / 1.0 RETREAT; 0 read as
1.0; a=0.6 retreats at 50%; both clamp edges; KITE shares the rule; RUSH unaffected; unseen camper
1.5 fires / 1.0 holds; 1.25 inclusive / 1.24 silent; veto unchanged for seen campers and trappers)
and a Brock hold-distance pin in `tests/test_bot_dispatch.py` (4.08 / 9.52 / 5.44 / 6.8 / 6.8 tiles
at a = 1.7 / 0.6 / 1.25 / 1.0 / 0). Counts: perception 41, personality 43, bot_dispatch 9,
archetype + steering + stats + config bundle 129, env 22.

**Amended 2026-09-21 (operator, on B4's second review note): a KITE bot holds no farther than its
own fire reach.** Two caps, not the one B4's note proposed. `policy.targeting` caps `desired_range`
at the new `Targeting.fire_reach`, `fire_range_fraction` (0 read as 1.0) x `attack_range`, and
`steering.maintain_range(..., max_dist=)` caps the SEEK edge at the same reach. The centre cap alone
was not enough: HOLD_RANGE's band is centre +- 1.5, and a kiter walking in stops at the band's FAR
edge (only strafe is left inside the band, and the orbit drifts outward), so a centre of 8.0 would
still have parked Brock near 9.5. Probe (open 60x60 map, a still hero, the bot starting 16 tiles
out, the last 5 s of 25): before, every kind at easy and medium spent 0% of that window inside its
reach, and so did the hard sniper (8.24 against 8.00), rifle (7.44 against 7.20), melee (2.77
against 2.67) and Edgar at every tier but elite (elite 21%); after, every kind at every tier is
96-100% inside, parked about 0.05 inside its reach (sniper 7.96, rifle 7.15, artillery 7.27, bull
6.25, melee 2.61, Edgar 1.94). The flee edge is unchanged, and so is every bot already inside its
reach. The Brock pin in `tests/test_bot_dispatch.py` moved 9.52 -> 8.0 (a = 0.6). Tests:
`tests/test_bot_dispatch.py` +1 (five kinds at three tiers, the hold is `min(scaled, reach)`),
`tests/test_steering.py` +1 (`max_dist` moves only the seek edge, scalar or tensor),
`tests/test_personality.py` +1 and one rewritten (Brock and the rifle close from just outside their
reach and orbit just inside it).

### B3.1 Retreat threshold
- Files: `brawl_sim/bots/personality.py` (`movement`, `_select_mode`)
- Do: gather `a = stats.gather_kind(params.aggression, state.ent_kind)` once in `movement` and read
  0 as neutral, `a = where(a > 0, a, 1.0)` -- B1 kept 0 as the value of a missing key so partial
  specs still validate, and every division below assumes this read has happened. Pass it to
  `_select_mode`; the HUNTER/KITE retreat test becomes
  `hp_frac < clamp(RETREAT_HP_FRACTION / a, 0.05, 0.90)`.
- Verify: `tests/test_personality.py` -- HUNTER at 25% HP: `a = 1.7` -> CLOSE, `a = 1.0` -> RETREAT.
- Done when: pinned; every existing case passes with `a = 1.0`.

### B3.2 KITE hold distance
- Files: `brawl_sim/bots/policy.py` (`targeting`, `desired_range`)
- Do: `desired_range = desired_range_fraction * attack_range * clamp(1 / a, 0.6, 1.4)`, with `a`
  gathered and 0-read-as-1.0 exactly as in B3.1 (a shared helper in `core/stats.py` is fine).
- Verify: `tests/test_bot_dispatch.py` -- a KITE Brock's hold distance at `a = 1.7` and `a = 0.6`.
- Done when: pinned.

### B3.3 CAMPER veto
- Files: `brawl_sim/bots/personality.py` (`fire_allowed`)
- Do: silent until seen **and** `a < 1.25`; at `a >= 1.25` a camper fires on sight.
- Verify: `tests/test_personality.py` -- unseen camper: `a = 1.5` fires, `a = 1.0` does not.
- Done when: pinned. Nothing else in the personality layer reads `aggression`.

## Step B4 -- the tier table

**DONE 2026-09-18.** B4.1: `TIER_FIELDS` and `DifficultyTier` gained `aggression` and `hero_focus`
(default 1.0). `aggression` joined `_FLOAT_TARGETS` as a plain, unclamped multiplier (B3 clamps each
of its reads). **Deviation:** `hero_focus` did NOT go into `_FLOAT_TARGETS`; the one-off
`lead_target` clamp became a two-entry `_UNIT_TARGETS = (lead_target, hero_focus)` loop that clamps
the product to [0, 1], so both fractions share one code path. Every write is still gated on
`reset_mask & bot_columns`; `torch.where`/`clamp` only. **Deviation:** `DifficultyTier` refuses
`aggression == 0`, because `aggression_of` reads a 0 product as the neutral 1.0 and such a tier would
silently play like `hard`; `hero_focus: 0.0` stays legal (easy uses it). B4.2: `configs/train.yaml`
tiers are the plan table exactly, six rows by nine columns (medium hp 1.0 -> 0.90; veteran 1.15/1.15;
expert 1.30/1.30; elite aim 0.20, reaction 0.30, hp 1.5, damage 1.5, aggression 1.7, hero_focus 1.7).
Stages, gates, window and the 75 M cap are untouched. The stale "elite hp 2.0 -> normalized 1.32"
comment was recomputed to 1.07 (Bull 10000 x 1.5 + 16 cubes x 400, over 20000), and the sentence
claiming easy/medium/hard eval columns compare with earlier runs was replaced: **no tier's eval
column is comparable with any run before 2026-09-18** (medium hp moved, and `hard` now includes
hero_focus 0.5). B4.3: `tests/test_training.py` pins the full 6 x 9 table as literals, the elite
row, S1's "no tier above 1.5x hp or damage", bot-columns-only on a stand-in params block whose hero
column holds non-zero bases (in the real env the hero's bases are 0 and cannot show a leak), the
clamp (0.7 x 1.7 -> exactly 1.0; aggression 3.4 not clamped), reset-rows-only for the two new
fields, the live dataclass defaults, and `TIER_FIELDS` == the applier's tables == `DifficultyTier`'s
fields. No other `TIER_FIELDS` reader needed a change. `scripts/train.py` has no `--dry-run`; the
config-only path is `load_train_config` -> `builder.build_run`, run on CPU: MaskablePPO builds and
elite gives 1.7 / 0.85 on all seven bot columns, 0 / 0 on the hero. **Not done, by design:** the
plan's "eval at the elite fixed tier shows the win-rate drop" -- no existing checkpoint loads
against this tree; the number moves to I5 as the new elite baseline. For I2: the tiers block, the
glossary above it and the two superseded bullets in the 2026-09-13 REBALANCED header are final.

**Review 2026-09-21 (second round).** No code change to B4. `tests/test_training.py` gained two
literal pins: the stage walk (`test_shipped_stage_walk_is_the_plan_walk`: five names, every weight,
gates 0.35 four times then None, window and min-at-stage 2000, cap 75 000 000, `demote_win_rate`
None; six curriculum mutants that survived before now fail), and the plan table's effective values
on a real env through `FixedTierHook.uniform` over two resets (aggression 0.6 / 0.8 / 1.0 / 1.2 /
1.4 / 1.7 and hero_focus 0.00 / 0.20 / 0.50 / 0.65 / 0.75 / 0.85 on every bot column, 0 on the
hero). **Resolved 2026-09-21 by B3's amendment** (the operator chose the cap; the paragraph is kept
as the finding): the plan-exact easy and medium aggression,
through B3's plan-exact `clamp(1/a, 0.6, 1.4)` hold scale, puts some KITE bots' hold beyond their
own fire range. Easy Brock holds 9.52 (8.02 to 11.02 with the 1.5 deadband) and fires at 8.00 or
less; easy rifle 8.40 vs 7.20; easy Spike 8.40 vs 8.00; medium Brock 8.50 vs 8.00; medium rifle
7.50 vs 7.20. Hard and above are all inside. These bots still fire at a target that closes on them
(the plan's "bad shot that runs away") but stand off without shooting in bot-vs-bot fights. `easy`
is in no training stage; `medium` is 30% of `hard_intro`. If unwanted, the change is B3's: in
`bots/policy.targeting` cap `desired_range` at `fire_range_fraction * attack_range` (read 0 as 1.0,
as `combat_rules` does; one more sync-free gather) and move the Brock pin in
`tests/test_bot_dispatch.py` from 9.52 to 8.0. The one-constant alternative is `_HOLD_SCALE_MAX`
1.4 -> 1.17, after which easy and medium hold the same. ("Hard and above are all inside" held for
the centre only; the probe in B3's amendment shows where the bots actually parked.)

### B4.1 Tier fields
- Files: `brawl_sim/training/config.py` (`TIER_FIELDS`), `brawl_sim/training/curriculum.py`
  (`_FLOAT_TARGETS`)
- Do: add `aggression` and `hero_focus` (default 1.0) as multipliers; clamp `hero_focus` to [0, 1]
  after scaling, the way `lead_target` is clamped.
- Verify: `tests/test_training.py::test_manager_applies_tier_multipliers_to_bot_columns` extended
  to both fields (bot columns only, hero column untouched).
- Done when: green.

### B4.2 The table
- Files: `configs/train.yaml` (curriculum tiers)
- Do: set the tiers to plan §3 Step B4's table. Summary: easy 3.0/2.2/0.25/2.0/0.90/0.75/0.70/
  ag 0.6/hf 0.0; medium 1.8/1.6/0.60/1.5/0.95/0.90/0.85/0.8/0.4; hard all 1.0; veteran
  0.6/0.6/1.0/1.0/1.0/1.15/1.15/1.2/1.3; expert 0.4/0.45/1.25/0.75/1.0/1.30/1.30/1.4/1.5; elite
  0.20/0.30/1.5/0.5/1.0/1.50/1.50/1.7/1.7 (order: aim_noise, reaction_delay, lead_target,
  decision_period, move_speed, hp, damage, aggression, hero_focus). Stages unchanged.
- Verify: B4.3.
- Done when: the file matches the table.

### B4.3 Pinned tests
- Files: `tests/test_training.py`
- Do: `test_shipped_tiers_are_listed_weakest_to_strongest` adds `aggression` and `hero_focus` to the
  monotone knob list; `test_shipped_elite_tier_is_the_operator_spec` pins the new elite row
  (hp 1.5, damage 1.5, aim 0.20, reaction 0.30, aggression 1.7, hero_focus 1.7).
- Verify: `.venv/Scripts/python.exe -m pytest tests/test_training.py -q`
- Done when: green, and a config-only build of `scripts/train.py` succeeds (its dry-run or the
  builder's config path).

---

# Phase G -- the gadget (plan §4)

A fourth attack-column value. An artillery spinner that flies up to 2 tiles toward the nearest
revealed enemy in 0.2 s and blasts 2000 in radius 1. 18 s cooldown, starts charged.

Read first: plan §0 S7-S12 and A-G1..A-G4, `brawl_sim/core/projectiles.py` (`spawn_supers`,
`step_projectiles`), `brawl_sim/env.py` `_attack_phase`, `brawl_sim/core/hero.py`,
`brawl_deployment/perception/shadow.py`, `brawl_deployment/control/buttons.py`,
`brawl_deployment/data/control_calibration.json` (`match_gate`).

## Step G1 -- constants, params, state

**DONE 2026-09-18.** As written, plus: `validate()` also rejects a negative `gadget_cooldown` and
deliberately allows `gadget_damage: 0` (a utility gadget); `core/state.check_invariants` bounds
`ent_gadget_cd` to `[0, gadget_cooldown]` the way it bounds `ent_dash_t`. Not touched, on purpose:
`render/ascii.py` draws every projectile with one glyph, and `render/assets/README.md` lists no
projectile kinds. Consequence to know: `configs/agent_obs.yaml` (full-info) selects
`projectiles.kind_onehot`, so its width went 7 -> 8 and checkpoints trained on that spec no longer
load; the lowinfo/deploy specs drop the field and are unaffected. The G3 fire path must set
`ent_gadget_cd = gadget_cooldown` and nothing else may write the field.

### G1.1 Projectile kind
- Files: `brawl_sim/constants.py`, `brawl_sim/core/obs_schema.py` (`projectiles.kind_onehot` width),
  `brawl_sim/render/viewer.py`, `brawl_sim/render/ascii.py`, `brawl_sim/render/assets/README.md`
  (if it lists kinds)
- Do: `Proj.GADGET_SPINNER = 7`; `N_PROJ_KINDS = 8`; `PROJ_CLASS_OF[GADGET_SPINNER] =
  ProjClass.ARTILLERY`; one-hot width 7 -> 8; a glyph and colour. Then grep
  `N_PROJ_KINDS|len(Proj)|== 7` in `tests/` for roster literals that pass by luck.
- Verify: `tests/test_constants.py` (`N_PROJ_KINDS == len(Proj)`), `tests/test_obs_schema.py`.
- Done when: green.

### G1.2 Per-kind params
- Files: `brawl_sim/config.py` (`PER_KIND_FIELDS`, `validate`)
- Do: `gadget_cooldown`, `gadget_range`, `gadget_flight_seconds`, `gadget_damage`, `gadget_radius`
  (all `_F32`; a kind that omits them resolves to 0 = no gadget). `validate`: if
  `gadget_cooldown > 0` then range, flight and radius must be > 0.
- Verify: `tests/test_config.py` -- a hero kind loads the five numbers; a bot kind resolves all to 0;
  `gadget_cooldown: 18` with `gadget_radius: 0` is rejected.
- Done when: pinned.

### G1.3 Authored values and the stats doc
- Files: `configs/brawlers.yaml` (hero block), `CHARACTER_DETAILS.md` §1
- Do: `gadget_cooldown: 18.0`, `gadget_range: 2.0`, `gadget_flight_seconds: 0.2`,
  `gadget_damage: 2000`, `gadget_radius: 1.0`. In the doc: the five numbers and the rules
  (nearest revealed enemy, facing when none; no self-damage; boxes take damage; no super charge;
  breaks concealment; does not reset the long-dash timer).
- Verify: `tests/test_configs_files.py`
- Done when: green and the doc paragraph exists.

### G1.4 State and timer
- Files: `brawl_sim/core/state.py` (`_ENTITY_FIELDS`), `brawl_sim/core/hero.py` (`tick_timers`)
- Do: `ent_gadget_cd (N,E) f32`; `zero_` leaves it 0 = ready ("starts charged"); `tick_timers`
  decrements it by `dt` every tick, clamped at 0, for every entity.
- Verify: `tests/test_state.py` (invariants), `tests/test_hero.py` (a cd of 18.0 reaches 0 after
  360 ticks and never goes negative).
- Done when: pinned.

## Step G2 -- the spinner mechanics (no action wiring yet)

**DONE 2026-09-18.** G2.1 `hero.gadget_target` is the plan formula except the wall clip:
`travel = where(hit, min(clamp(hit_t - los_step, 0), travel), travel)`, equal to the plan's
`|hit_pos - los_step*dir - pos_e|` whenever `hit_t >= los_step` and avoiding a bogus positive norm
when the endpoint sample itself is the hit (entity < 0.5 tiles from a wall); a target coincident
with the thrower falls back to the facing (`has & (nearest_dist > _EPS)`), so `dir` is a unit
vector on every row. G2.2 `projectiles.spawn_gadget(state, fire, origin, dir, travel, damage,
params, cfg)` with the listed slot values, plus **`_LANDING_OVERSHOOT = 1e-4`**
(`vel = dir * travel * 1.0001 / gadget_flight_seconds`): without it travels 0.3 and 0.8 fell an ulp
short in float32 and landed on tick 5 instead of 4; detonation is still on `prj_target`, in-flight
position is off by <= 2e-4 tiles, and `tests/test_projectiles.py` pins the resulting 7.50075. Do not
"fix" it back to the bare formula. G2.3 `no_self = is_hazard | (prj_kind == GADGET_SPINNER)`;
Grom's self-hit pinned at 2080. G2.4 `step_projectiles` returns a 5-tuple `(dmg_ent, dmg_by,
dmg_box, heal_ent, charge_hit)` via a parallel accumulator masked by kind; `_projectile_phase`
returns `(dmg_by, healed, charge_hit)` and `_bookkeeping(dmg_by_total, charge_hit)` takes the mask
built in `_run_tick` (two lines of `_run_tick` touched). Every caller updated: `env.py` plus the
unpacking lines of `tests/test_{artillery,bull,rifle,sniper,spike,super}.py`. G2.5
`stats.effective_gadget_damage` mirrors `effective_damage` (0 cubes 2000; 3/10/16 cubes 2600/4000/
5200; bots 0). The plan's "enemy behind a wall at 1.5 tiles takes 0" case is tested with reachable
geometry (hero x 10.7, wall column 11, enemy x 12.2): the literal point (hero 10.0, enemy 11.5) lies
INSIDE the wall tile and the 0.5-clipped landing is exactly 1.0 from it, which `<=` would catch. The
checklist's `tests/test_env_step.py` does not exist; the env pins are in `tests/test_env.py`
(gadget-only hit: bot HP delta 2000, super charge 0; dash hit: delta 2000, charge 1 -- pinned as HP
deltas because the tick clamps HP to the kind's effective max). Hot path checked sync-free.
Tests: `tests/test_gadget.py` (new, 16), `test_projectiles` +6, `test_stats` +4, `test_env` +2;
75 + 117 green across the twelve touched files. G3 must call `gadget_target` with the fair
`perception.visibility`, spawn via `spawn_gadget` with `effective_gadget_damage`, set
`ent_gadget_cd = gadget_cooldown`, and add `gadget_fire` to the reveal/out-of-combat reset but NOT
to `ent_attack_idle_t`.

### G2.1 Target selection
- Files: `brawl_sim/core/hero.py` (new `gadget_target(state, vis, params, bank, cfg)`)
- Do: returns `(dir (N,E,2), travel (N,E))`. Verbatim from plan §4 Step G2:
  ```
  revealed = vis & alive.unsqueeze(1) & not_self          # (N,E,E): what e can see
  dist = |pos_j - pos_e|; dist_eff = where(revealed, dist, inf)
  nearest = argmin(dist_eff, -1); has = revealed.any(-1)
  to = normalize(pos[nearest] - pos_e); dir = where(has, to, facing_vec)
  travel = where(has, minimum(gadget_range, dist[nearest]), gadget_range)
  hit, hit_pos, _ = terrain.march(bank.blocks_proj, map_id, pos_e, dir, travel, cfg)
  travel = where(hit, |hit_pos - los_step * dir - pos_e|, travel)
  ```
  `vis` is the fair `(N,E,E)` visibility; for the hero row it equals the grid's `enemy_revealed`.
- Verify: `tests/test_gadget.py` (new) -- revealed enemy at 1.5 tiles: travel 1.5 toward it; at 2.8:
  travel 2.0; nothing revealed: 2.0 along facing; hidden enemy at 1 and revealed at 2: aims at the
  revealed one; wall at 1 tile: travel clipped below 1.
- Done when: the five cases pass.

### G2.2 Spawn
- Files: `brawl_sim/core/projectiles.py` (new `spawn_gadget`)
- Do: sibling of `spawn_supers`: one slot per firing entity via `alloc_slots`, `_write_slots` with
  `target = origin + dir * travel`, `vel = dir * travel / gadget_flight_seconds`,
  `dist_left = travel + eps`, `aoe = gadget_radius`, `damage`, `kind = GADGET_SPINNER`,
  `cls = _ARTILLERY`, `pierce = false`.
- Verify: `tests/test_projectiles.py` -- one slot written with those values; a zero `travel` still
  writes a slot that `step_projectiles` detonates on its first tick.
- Done when: pinned.

### G2.3 Owner exclusion
- Files: `brawl_sim/core/projectiles.py` (`step_projectiles`)
- Do: `no_self = is_hazard | (prj_kind == GADGET_SPINNER)`;
  `owner_ok = ~no_self.unsqueeze(-1) | (prj_owner.unsqueeze(-1) != entity_idx)`. Grom's artillery
  self-hit stays as it is.
- Verify: `tests/test_projectiles.py` -- a spinner blast at the owner's feet deals 0 to the owner;
  a Grom shell at the owner's feet still hits the owner.
- Done when: both pinned.

### G2.4 Super-charge exclusion
- Files: `brawl_sim/core/projectiles.py` (`step_projectiles` return), `brawl_sim/env.py`
  (`_projectile_phase`, `_bookkeeping`), every other caller of `step_projectiles` (grep)
- Do: return one more `(N,E,E)` bool `charge_hit` = `dmg_by > 0` minus the gadget's contributions;
  `_bookkeeping` derives the projectile-phase `hits` from `charge_hit` instead of `dmg_by_total > 0`.
  Dash hits and volleys charge as today.
- Verify: `tests/test_projectiles.py`, `tests/test_env.py` -- `super_charge` unchanged after a
  gadget-only hit; changed after a dash hit.
- Done when: pinned.

### G2.5 Cube-scaled damage
- Files: `brawl_sim/core/stats.py`
- Do: `effective_gadget_damage(kind, cubes, params)` applying the same cube bonus as
  `effective_damage`, on `gadget_damage`.
- Verify: `tests/test_stats.py` (or where `effective_damage` is tested).
- Done when: 0 cubes -> 2000; the same multiplier as attacks at N cubes.

## Step G3 -- action space 3 -> 4 and the wiring

**DONE 2026-09-21.** G3.1: `action_nvec` is `(n_move_bins + 1, 4)`; `hero.action_mask` stacks
`[no_fire_ok, fire_ok, super_ok, gadget_ok]` (21 flat); `decode_action` returns a 4-tuple
`(move_dir, fire, super_fire, gadget_fire)`; `obs_schema` declares `action_mask.attack` as
`("N", 4)`. **Addition:** `hero.gadget_ready(state, params)` (`alive & gadget_cd <= 0 &
gadget_cooldown > 0`, not ANDed with `ready`) is the one predicate behind the mask, the
`_attack_phase` safety net and G4's `hero.gadget_ready` field. G3.2: `_bot_phase(hero_move_dir,
hero_fire, hero_super, hero_gadget)` returns `(move_dir, fire, super_fire, gadget_fire, aim_dir,
aim_point, vis)` and hands on the fair visibility it already built (no second pass: nothing between
phases 4 and 6 moves an entity or a reveal timer). `_attack_phase(..., gadget_fire, vis)` applies
the safety net, uses `offensive = attacked | gadget_fire` for the reveal and out-of-combat resets,
keeps `ent_attack_idle_t` on `attacked` (a throw does not spend the long dash), writes
`ent_gadget_cd = gadget_cooldown`, and spawns after `spawn_supers`, before the volley. Ammo,
`attack_cd` and `ent_shots_fired` are untouched; `_held` is unchanged. **Behaviour to know, all
pinned:** (1) at the shipped `action_repeat: 5` the gadget is legal again on the 73rd observation
after a throw, 365 ticks = **18.25 s**, not 18.0 (S5's decision-boundary effect); any decision-rate
comparison against the sim must expect that (G5.1, G4's `gadget_charge_frac`). (2) A full
projectile buffer spends the 18 s and the concealment and throws nothing, the same silent drop as a
super's charge. (3) An override on the hero's slot drives movement and the ordinary attack only;
super and gadget always come from the action, so an override that fires under an action of 3 gives
a dash AND a gadget on one tick (only tests override slot 0). (4) With `action_latency` longer than
the repeat, the mask stays legal until the delayed throw lands and a second request is a silent
no-op, as for attacks. G3.3: `sb3_vecenv.action_masks` changed in its docstring only;
`scripts/record_rollout.py`'s random policy draws the whole column; `scripts/play_manual.py` sends
3 on `g` and titles `gadget=READY` or the seconds left. **Deviation:** with `g` and space both held
the gadget takes the column only while it is legal (a `g` tap spans several ticks and would
otherwise eat the dash). Review added `connect_input`, which disconnects matplotlib's default key
handler from the game figure: `g` toggled the grid, `s` opened the save dialog, `l` raised
OverflowError on a log axis, `q` closed the figure. **Deviation:** `scripts/watch.py`
`_check_spaces` and the cadence audit's `--run` mode refuse a pre-gadget checkpoint with an "action
space mismatch" SystemExit instead of a mask-shape crash inside `predict`, so A1's report cannot be
regenerated against `mortis_deploy3_elite` (`--telemetry`, which A2.4 uses, loads no model). The
audit's `_attacked` is now `in (1, 2)`, so a throw is not scored as an attack. **Deviation, the
operator's to confirm:** `brawl_deployment/policy.py` sizes the mask from the checkpoint's own
nvec, holds the gadget column False until G5 (`_SHADOW_ATTACK_WIDTH = 3`), and `check_spaces`
accepts `cfg.action_nvec` OR the legacy `(n_move, 3)`, so the deployed checkpoint can still play
A2.4's live match. Checked on `runs/mortis_deploy3_elite-20260913-015933`: model nvec [17 3], mask
(1, 20), `act()` returns a `Decision`; `(17, 2)`, `(17, 5)`, `(16, 4)` and `(9, 3)` are refused.
Rejecting it is a one-line revert plus one test. Nothing else loads a pre-G3 checkpoint: not the
sim, training, watch or the audit. G3.4: `tests/test_gadget.py` 16 -> 40, every scenario through
`BrawlVecEnv.step` with literals (per-step HP loss `[0,0,0,2000,2000]`, 2.8 in and 3.5 out, two
enemies and a crate, the wall and its no-wall control, 18.0 then 17.95, 359 masked observations then
legal, reveal 1.0 with idle not reset, throws mid-dash and on an empty clip, the masked no-op, the
safety net, one per decision `[1,1,1,0,0]`, one slot for four ticks), plus two review blocks:
homing on an enemy off the facing, no homing on a bush-concealed enemy, cube-scaled damage through
the env, the three override cases, a restart charged after an episode ends, a three-env batch (each
env throws only its own), a throw on the move, the full buffer, the 73rd observation, and
`test_a_gadget_thrown_every_decision_is_sync_free_on_cuda` (64 envs, a throw every decision under
`set_sync_debug_mode("error")`). G3.3's verify clause is pinned in `tests/test_sb3_vecenv.py`
(MaskablePPO builds a `[17, 4]` head and a gadget pick reaches the sim), because
`tests/test_training.py` holds no width literal. Out-of-list files: width literals in
`tests/test_super.py`, `test_config.py`, `test_env.py`, `test_sb3_vecenv.py`,
`test_observation.py`; `tests/test_record_rollout.py` (+1, missing from the report);
`scripts/audit_attack_cadence.py` and its test; `brawl_deployment/policy.py` and its test;
`docs/OBSERVATION.md` (in sync with `dump_obs_schema.render()`, empty diff); `BRAWL_SIM_BUILD_PLAN.md`
Appendix D rewritten; `README.md` (the `g` key). **Left for later owners:**
`brawl_deployment/control/buttons.py`'s docstring (G5.2); `BRAWL_DEPLOYMENT_DESIGN.md` §4.3, guard 4,
the mask paragraph and the brawl-deployment skill file (G5.4, widened below); `scripts/smoke_test.py`
and `scripts/benchmark.py` still draw attack from {0, 1} (I4); `BRAWL_SIM_BUILD_PLAN.md:362` and
`bot_overhaul.md:1106` are historical step text that Appendix D supersedes. Both fix passes died on
the usage limit; their edits were reconstructed from the transcripts. The first review's one major
(no test proved the spinner homes on the nearest REVEALED enemy; two `vis` mutants survived) is
closed by the homing tests, and the second review found no majors.

**Amended 2026-09-21 (operator: old checkpoints are deprecated and will be deleted).** The legacy
acceptance above is reverted. `check_spaces` accepts `cfg.action_nvec` only, and a `(n_move, 3)`
model is refused with a message that names it a pre-gadget checkpoint and says to retrain; any
other mismatch is reported plainly. `DeployedPolicy` sizes its mask from `cfg.action_nvec` again,
with the gadget column still held False until G5. In `tests/test_deployment_policy.py` the "still
loads" test became `test_a_pre_gadget_checkpoint_is_refused`, and the end-to-end test skips a local
run whose load is refused as pre-gadget, by name ("delete it"), so a machine that still holds the
old runs stays green while any other load failure still fails. All four local deployable runs are
`[17 3]`, and `configs/deployment.yaml`'s `run.dir` names one of them, so deployment has no
loadable checkpoint until the next run trains. The stale `action_nvec = (n_move_bins + 1, 3)`
sentence in `brawl_deployment/control/buttons.py`'s docstring is corrected; G5.2 still owns the
gadget tap.

### G3.1 Action spec and mask
- Files: `brawl_sim/config.py` (`action_nvec = (n_move_bins + 1, 4)`), `brawl_sim/core/hero.py`
  (`action_mask`, `decode_action`), `brawl_sim/core/obs_schema.py` (`action_mask` width)
- Do: `gadget_ok = alive & (gadget_cd <= 0) & (gadget_cooldown > 0)` (not ANDed with `ready`);
  `attack = stack([no_fire_ok, fire_ok, super_ok, gadget_ok])`; `decode_action` yields
  `gadget = attack_col == 3`. Grep `n_move_bins + 1, 3`, `(N, 20)`, `[0, 1, 2]` in `tests/`,
  `brawl_sim/`, `brawl_deployment/` for width literals.
- Verify: `tests/test_hero.py` -- mask is `(N, 21)`; gadget legal while dashing; illegal at
  `gadget_cd > 0`.
- Done when: pinned.

### G3.2 `_attack_phase` wiring
- Files: `brawl_sim/env.py` (`_decode`, `_attack_phase`)
- Do: `gadget_fire` from decode; safety net `can_gadget = alive & (gadget_cd <= 0) &
  (gadget_cooldown > 0)`; on fire: `ent_gadget_cd = gadget_cooldown`, add `gadget_fire` to the
  `attacked` OR that breaks concealment and out-of-combat, but NOT to the `attack_idle_t` reset;
  damage from G2.5; `dir, travel` from G2.1; call `spawn_gadget`. `_held` is unchanged (it zeroes
  the whole column).
- Verify: G3.4.
- Done when: the scenario tests pass.

### G3.3 Consumers of the column
- Files: `brawl_sim/wrappers/sb3_vecenv.py` (`action_masks`), `scripts/play_manual.py` (a key),
  `scripts/record_rollout.py`, `scripts/watch.py` (if they decode the column)
- Do: widen; add a gadget key to manual play.
- Verify: `tests/test_play_manual.py`, `tests/test_training.py`'s MaskablePPO smoke test builds
  against the widened space.
- Done when: green.

### G3.4 Scenario tests
- Files: `tests/test_gadget.py`
- Do: enemy revealed at 1.5 tiles takes 2000 at tick 4, hero takes 0; enemy at 2.8 (0.8 from the
  landing point) takes 2000; enemy at 3.5 takes 0; two enemies in radius both take 2000 and a box
  inside takes 2000; enemy behind a wall at 1.5 takes 0; `ent_gadget_cd == 18.0` after firing and
  the mask refuses until it reaches 0; reveal timer set, `attack_idle_t` not reset; a gadget chosen
  while masked is a no-op; one gadget per decision; a gadget uses one projectile slot for 4 ticks.
- Verify: `.venv/Scripts/python.exe -m pytest tests/test_gadget.py -q`
- Done when: green, then `graphify update .`.

## Step G4 -- observation fields

**DONE 2026-09-21.** `core/hero.py` gains `gadget_charge_frac(state, params)` beside `gadget_ready`:
`1 - gadget_cd / gadget_cooldown` clamped to [0, 1], 0.0 for a kind without a gadget, and no `alive`
term, like `super_charge_frac`. `build_obs` emits `hero.gadget_ready` and `hero.gadget_charge_frac`
right after the long-dash pair, `OBS_SCHEMA` declares both rows in that order, and
`_HERO_DESCRIBE_FIELDS` gains `gadget_ready`. **Deviation:** the `hero.gadget_ready` field is G3's
`hero.gadget_ready(state, params)` predicate, as G3's addition said it would be, so it carries the
`alive` term that the formula below leaves out and always equals `action_mask.attack[:, 3]`. A dead
hero reads not-ready with its fraction intact. **What the policy sees at `action_repeat: 5`:** the
first observation after a throw reads 0.2 / 18 = 0.0111, not 0.0, because the throw lands on
sub-tick 1 and four more sub-ticks count down before the observation is built. It is the same
decision-boundary offset as G3's 73rd-observation behaviour, which G5.1's note already covers.
`docs/OBSERVATION.md` is regenerated with both rows (the file is gitignored, so `git diff` shows
nothing); `docs/AGENT_OBS.md` is unchanged, because no spec selects the fields until I1. Nothing in
deployment changes: `ObservationAssembler` fills only what its suppliers hand it and then selects by
spec, so G5.1's shadow must supply both fields before deploy4 selects them. Tests:
`tests/test_observation.py` +5 (charged at reset; 0.0 / 0.25 / 0.5 / 0.99722 / 1.0 for timers 18 /
13.5 / 9 / 0.05 / 0; the field equals the mask column across ready, cooling, dead and no-gadget,
with fractions 1.0 / 0.5 / 1.0 / 0.0; through `env.step` at repeat 1, 0.0 then 0.05 / 18 then ready
at 1.0; at repeat 5, a timer of 17.8 and 0.0111) and `tests/test_obs_schema.py` +4 (both rows'
shape, dtype, units and range; the hero rows in `build_obs` order with the gadget pair after the
long dash; `validate_obs` refuses either field missing; `describe_obs` prints `gadget_ready: T`).
Six mutants, all killed: the fraction inverted, the no-gadget zeroing dropped, the fraction gated on
`alive`, the field swapped for the super's flag, the field without `alive`, and the two schema rows
swapped. Full `-m "not vision"` suite: 2256 passed, 4 skipped (the four retired pre-gadget runs), 85
vision tests deselected, 20 min.

### G4.1 Fields
- Files: `brawl_sim/core/observation.py` (`build_obs` hero block), `brawl_sim/core/obs_schema.py`
  (`OBS_SCHEMA`, `_HERO_DESCRIBE_FIELDS`)
- Do: `hero.gadget_ready (N,) bool` = `gadget_cd <= 0 & gadget_cooldown > 0`;
  `hero.gadget_charge_frac (N,) fraction` = `1 - gadget_cd / gadget_cooldown` (1.0 when ready; 0
  for kinds without a gadget). Supplier at deployment: `ShadowHero.observe()` (G5.1).
- Verify: `tests/test_observation.py`, `tests/test_obs_schema.py` (round trip).
- Done when: green; `scripts/dump_obs_schema.py` regenerates `docs/OBSERVATION.md` with both rows.

## Step G5 -- deployment: shadow, buttons, policy

**DONE 2026-09-21.** The shadow owns the gadget as proprioception (plan S18):
`ShadowParams.gadget_cooldown` (Mortis 18.0) and a float32 `gadget_cd`, 0 at `reset()`, decremented
in phase 2 beside `attack_cd` and restarted by a throw in phase 6. `attack_mask()` returns four
legals; the fourth is `gadget_ready` alone (alive, a kind with a gadget, off cooldown), outside the
gate the attack and the super share, so it is legal mid-dash. `observe()` adds `gadget_ready` and
`gadget_charge_frac`, and `_put_self` maps both. `Buttons` takes a required `gadget` origin and taps
it: down on the decision tick, up on the next, never a move; its module docstring, which G3 left to
G5.2, describes the tap and points at G6. The policy's mask is the shadow's four legals verbatim
(`_SHADOW_ATTACK_WIDTH` is gone), and `act` refuses any other width by name. The loop builds the
gadget from `cal.button("gadget")`; a decision row carries attack 3 and bit 3 of `attack_legal`, so
a fresh match's first row reads `0b1011`. Deviations and choices:

1. **A throw is not an attack.** It spends no ammo and moves neither `attack_cd`, `attack_idle_t`
   nor `_last_attack_at`, so it never costs the long dash and opens no canary grace window, as in
   `env._attack_phase`. It shares the attack's one pending slot, so one decision is still one press.
2. **`resync()` also keeps a queued throw**, beyond G5.1's "leaves `gadget_cd` alone". The canary's
   evidence is ammo, and the loop taps only what `act` modelled, so the shadow's gadget can lag the
   game's but never lead it; dropping the queued throw would offer the policy, for a whole cooldown,
   a gadget the game has already spent. The skill file names this as the one exception to "reseed
   conservatively".
3. **`gadget_cooldown` is the one `ShadowParams` field with a default, 0.0.** The sim resolves an
   absent per-kind field to 0 (`config.PER_KIND_FIELDS`), which is its "no gadget": never legal,
   fraction 0.0, and `act` refuses the throw. Every other missing field still raises.
4. **`Buttons.gadget` is required**, so a `Buttons` that cannot press the gadget cannot be built by
   accident. `require_on_screen` only checks that the tap point is on the screen, since a tap needs
   no room around it, and `aim_point` raises for the gadget.
5. **Found and fixed: `scripts/deploy_run.py --dry-run` rebuilt `Buttons` positionally**, through
   `type(loop.controls.buttons)(...)`, which a search for `Buttons(` misses. With `gadget` required,
   every dry run would have crashed before its first tick. The rebuild moved to
   `Controls.with_backend(backend)` beside `Controls.build`, carrying every field by name, and the
   script calls it. Nothing offline reached either path before: one new test covers `with_backend`,
   and another runs the real `Controls.build` with only ADB faked, where `_shipped_buttons` only
   copies it.
6. **`match_state.py`'s second reason for the gadget anchor, that the policy never presses it, no
   longer holds.** Its module docstring and design §5 carry a dated note pointing at G6; the anchor
   and the JSON `_comment` stay G6.2's.
7. **G5.3's dry-run check needs the emulator window**, so it is covered offline by the unchanged
   `check_spaces` tests (`test_a_checkpoint_with_the_wrong_action_space_is_refused`,
   `test_a_pre_gadget_checkpoint_is_refused`). The real-checkpoint seam test now passes four legals
   and the gadget pair, but all four checkpoints on disk are pre-gadget, so it skips until a run
   trains on this tree.
8. **Deploy4 is still refused, now by H4 alone.** The assembler raises `KeyError('enemy_hist1')` at
   construction as before, so both strict xfails stand; the loop test's comment and reason now name
   only H4.
9. **The timing is the sim's to the bit**, G3's decision-rate figure included: the throw's own
   sub-tick writes 18.0, float32 needs 360 decrements to clear it, and at the deployed 4 Hz the
   earliest second throw is decision 73, 18.25 s after the first.

G5.4, each revision dated: `BRAWL_DEPLOYMENT_DESIGN.md`'s scope paragraph (gadget use left the
out-of-scope list), §4.3 (`attack ∈ {0, 1, 2, 3}`, nvec `(n_move_bins + 1, 4)`), §4.4 (its heading
marked, the "not emitted" sentence struck, masking four wide, and a closing gadget paragraph), §5
(the side effect revised), §6.3's table (a row for the gadget pair), guard 4 (holds as written
again) and the mask paragraph (`attack (4)`); the skill file's fire semantics, a gadget paragraph in
place of "never touch the gadget button", and its shadow section. Tests:
`tests/test_deployment_shadow.py` +16 (parity with the sim, closed-loop on the shadow's own mask at
1 and 5 sub-ticks per decision with throws mid-dash, exact on the pair and on every attack field a
throw must not move; charged at the gate; exactly 360 sub-ticks; decision 73 at 4 Hz; the first
sub-tick; not an attack; legal mid-dash; one pending slot both ways; the dead hero; a kind with no
gadget; resync keeping the timer and a queued throw; the `observe` coverage test over every deploy
spec), `tests/test_deployment_control.py` +6 (the bare tap, a tap and a press in flight, release, an
off-screen gadget, `with_backend`, the real `Controls.build`), `tests/test_deployment_assemble.py`
+1 (deploy4 minus `history` and the `enemy_hist` planes against `build_agent_obs`, throwing first so
a swapped pair shows) and `tests/test_deployment_loop.py` +1 (a gadget decision is a tap on the
gadget button, and the next row reads `0b0011`); the loop's fake policy asserts four legals, and the
policy, calibration and cadence-audit tests take four legals, a gadget origin and bit 3. Twenty-six
mutants, all killed: no phase-2 decrement, a throw counted as an attack, a throw restarting one
sub-tick short, resync dropping a queued throw or restarting the timer, a dash, attack-cooldown or
ammo term in the gadget mask, `gadget_ready` without `alive`, the fraction in float64, `reset`
leaving it uncharged, a kind without the field refused, `act` never queueing a throw, `observe`
swapping the pair, `_put_self` swapping the pair or reading the super, the tap on the super's
button, never lifted or dragged like a press, the on-screen check skipping the gadget, `aim_point`
accepting it, the policy holding the gadget column False or accepting three legals, `Controls.build`
wiring the super as the gadget, and `with_backend` dropping the gadget or keeping the live backend.
The parity test alone killed nine of the fourteen shadow mutants. Full `-m "not vision"` suite: 2313
passed, 4 skipped, 2 xfailed (deploy4's two), 85 deselected, in 21 min 1 s.

### G5.1 Shadow
- Files: `brawl_deployment/perception/shadow.py`, `brawl_deployment/perception/assemble.py`
  (`_put_self`)
- Do: `ShadowParams.gadget_cooldown`; `gadget_cd` timer, 0 at `reset()`; `_tick` decrements;
  `attack_mask()` returns four legals; `act(move, 3)` queues a gadget (no `attack_cd`, no ammo, no
  `_last_attack_at`); `observe()` adds `gadget_ready` and `gadget_charge_frac`; `resync()` leaves
  `gadget_cd` alone. `_put_self` maps the two new `hero.*` names.
- Verify: `tests/test_deployment_shadow.py` -- legal at reset, illegal for 18 s after `act(_, 3)`,
  legal at 18.0; attack legality unaffected; `tests/test_deployment_assemble.py`.
- Note (from G3): the shadow's own timer is right to read legal at 18.0, but a comparison against
  the SIM at decision rate sees 18.25 s (G3's note, behaviour 1). Do not report that as a parity
  failure.
- Done when: pinned.
- Amended by I1 (2026-09-21): deploy4 selects both fields, so `tests/test_deployment_loop.py`'s
  decision test runs deploy4 as a strict xfail (`NOT_SERVED_YET`) until G5.1 and H4.1 to H4.3 are
  all in. The step that lands last deletes the entry, and the strict XPASS says which step that is.
  A step that changes the exception the case raises restates it there.

### G5.2 Buttons
- Files: `brawl_deployment/control/buttons.py`, `brawl_deployment/match_state.py`
  (`Calibration.button("gadget")` exists; confirm)
- Do: `origin(3) = cal.button("gadget")`; action 3 is a tap: down on one tick, up on the next, no
  drag; `aim()` raises for 3; the off-screen radius check does not apply to a tap. One press per
  decision; a press in flight finishes first.
- Verify: `tests/test_deployment_control.py` -- action 3 emits down at the gadget centre then up and
  never a move.
- Done when: pinned.

### G5.3 Policy and loop
- Files: `brawl_deployment/policy.py`, `brawl_deployment/loop.py`
- Do: `_mask` is `n_move + 4`; `attack_legal` a 4-tuple; `Decision.attack` in 0..3; `check_spaces`
  compares the widened nvec; the "no-fire must be legal" guard unchanged; `TickRow.attack`
  docstring says 0..3.
- Verify: `tests/test_deployment_policy.py` -- a 4-wide mask reaches `predict`; the calibration
  script's dry run still refuses a spec/checkpoint mismatch.
- Starting point (from G3 and its 2026-09-21 amendment): the mask is already `n_move + 4` wide,
  sized from `cfg.action_nvec`, with the gadget column held False (`_SHADOW_ATTACK_WIDTH = 3`);
  `act()` still takes the shadow's 3-tuple; `check_spaces` accepts `cfg.action_nvec` only (the
  legacy `(n_move, 3)` is retired). `DeployedPolicy.from_run` passes `check_holdout=False` (M4's
  second review): keep the keyword.
- Done when: green.

### G5.4 Design doc
- Files: `BRAWL_DEPLOYMENT_DESIGN.md` §4.3, §4.4 and the policy guards;
  `.claude/skills/brawl-deployment/SKILL.md`
- Do: append a dated gadget paragraph; strike the "Gadgets are not emitted" sentence with a date,
  the way §4.4 was revised before.
- Also (G3's reviews): §4.3 (~:355-356) still says `attack ∈ {0, 1, 2}` and
  `action_nvec = (n_move_bins + 1, 3)`; guard 4 (~:1641) says any nvec other than `cfg.action_nvec`
  is refused, which holds as written again since the legacy acceptance was retired on 2026-09-21
  (a dated line saying so is enough); the mask paragraph (~:1648) says `[move (17), attack (3)]`;
  the skill file (:70) says `attack ∈ {0, 1, 2}`. Revise each with a date. The design doc stores
  `--` as an em-dash, so anchor any scripted patch on text without dashes.
- Done when: present.

## Step G6 -- the match gate anchor -- DONE 2026-09-22

Gate: G5 done. Must complete before any live run with the gadget enabled.

Deferred 2026-09-21 by the operator: G6.1's trace and G6.2 run after training is complete, with
I5's dry run and deploy items. Training does not depend on G6, which only gates live runs.
Unblocked 2026-09-22: the run finished, and both substeps landed the same day. Read G6.2's note
before anything above it. The trace found that the gate was not on the gadget at all and never
had been, because all three button names in the calibration were one disc off.

### G6.1 Probe job
- Files: `scripts/deploy_calibrate.py` (`--probe-gadget`)
- Do: in a real match (not Training Grounds), tap the gadget once and record all three anchors' ring
  scores every tick for 20 s to `runs/audit/gadget_anchor_trace.json`.
- Needs: live emulator.
- Done when: the trace file exists.
- Progress 2026-09-21: the probe is built; the trace waits on a live match. The command is
  `python scripts/deploy_calibrate.py --probe-gadget`, run alone, never with `--jobs` or
  `--write`, and the operator's steps are in the script's docstring. It waits for the gate,
  scores all three buttons every tick for 2 s, taps the gadget once through `Buttons.press` if
  the gate still reads in match, scores 20 s more at `loop.tick_hz`, writes the trace and prints
  KEEP, CHANGE or INCONCLUSIVE. Deviations: the 2 s before the tap are a baseline G6.1 did not
  ask for; `MatchState` gains a read-only `anchor` property, `(cx, cy, r)` after `refine`, where
  the plan lists match_state.py for G6.2 only; and each row records every button's mean disc
  colour, because a tap that never landed would otherwise read as KEEP. A gadget whose colour
  moves by at most `UNCHANGED_COLOUR` is INCONCLUSIVE, and so is a trace that does not cover the
  18 s recharge, `GADGET_RECHARGE_S`, which a test pins to brawlers.yaml. The 8.0 in
  `UNCHANGED_COLOUR` is a first guess, to revise against the first real trace. Ticks where every
  button is under the threshold at once are the controls leaving the screen, not the gadget, and
  are not counted against it. A CHANGE names the alternative anchor with the widest margin and
  says whether the 2-of-3 vote held. `runs/` is gitignored, so G6.2's replay test needs the trace
  copied under `tests/fixtures/`. Tests: `tests/test_deployment_calibration.py` +16. 21 mutants,
  all killed.
- **DONE 2026-09-22.** `runs/audit/gadget_anchor_trace.json`, 264 ticks at 12 Hz in a real
  match, tap at tick 24, verdict KEEP. It took two runs. The first returned INCONCLUSIVE, and the
  operator, watching the screen, reported that the gadget press had landed on the Super button.
  That was correct, and the cause was not the probe: every button name in the calibration had
  been one disc off since 2026-09-09 (see G6.2).
- Two defects in the probe itself surfaced from it and are fixed. `_verdict` read "did the tap
  land?" off the gate anchor's colour, which is only valid while the tapped button and the gate
  anchor are the same disc; they are now deliberately different, so `TAPPED_ANCHOR` is a separate
  constant and the probe reports on the button it pressed while gating on the button it watches.
  The live log line also named the gate as though it were the tap target.
- `UNCHANGED_COLOUR = 8.0` survived its first real trace, so the guess stands: the thrown gadget
  moved 224.3 colour levels and the two untouched discs moved 0.0. There is no ambiguity near the
  threshold to tune against.

### G6.2 Decide and implement
- Files: `brawl_deployment/data/control_calibration.json` (`match_gate`),
  `brawl_deployment/match_state.py` (only if voting)
- Do: if the gadget score never drops under 0.45 during the cooldown, keep the anchor and cite the
  trace in `_comment`. Otherwise either re-anchor on whichever of `super` / `attack` held the wider
  margin in the trace, or implement a 2-of-3 vote in `MatchState.update` (`_above`/`_below` per
  anchor, in-match while at least two agree). Rewrite the `_comment` that says "the one button the
  policy never presses" either way.
- Verify: `tests/test_deployment_capture.py` -- a `MatchState` test replays the trace and stays
  in-match throughout. Also rerun `tests/test_deployment_control.py::test_gate_separates_gameplay_from_menus_on_real_footage`,
  which the usual `-m "not vision"` run skips: it already failed its 0.7 gameplay floor at 0.675
  when A2 ran and still did on 2026-09-21 (threshold 0.45; its enter/exit checks pass), with no
  overhaul change feeding it. Any anchor change moves that margin, so re-measure it here.
- Done when: the trace-replay test passes and a `control.backend: null` dry run shows no gate exit
  at the first gadget decision.
- Amended by G5 (2026-09-21): the gadget is live in deployment, so any checkpoint trained on this
  tree can tap the anchor. `brawl_deployment/match_state.py`'s module docstring and design §5 carry
  a dated note that the policy presses it; settle both with the `_comment` once the trace decides.
  The dry run in "Done when" goes through `Controls.with_backend` since G5, gadget included, and
  `tests/test_deployment_control.py` covers that path.
- Amended by G6.1 (2026-09-21): a `control.backend: null` dry run sends no touch, so the game's
  gadget button never changes under it. It proves the loop's own path through a gadget decision,
  not that the anchor survives a real tap; that evidence is the trace and its replay test.
- **DONE 2026-09-22, and the verdict is CHANGE for a reason this step did not anticipate: the
  anchors were mislabelled, not mis-chosen.** Stored `attack` is the gadget (1675.9, 997.0), stored
  `gadget` is the Super (1559.9, 901.1), and stored `super` is a third disc nothing presses, now
  `hypercharge` after its icon. No coordinate moved. The rename is the whole fix, and the `_comment`
  claiming "the one button the policy never presses" was false about the disc it named. Two
  consequences beyond the gadget: every super the policy has emitted went to `hypercharge`, and the
  gate was anchored on the real Super, which the policy presses.
- Bakeoff on `tests/fixtures/vision/bluestacks-example-new.mp4`, 238 sampled frames, measured
  through `MatchState.update`'s own path at the stored centres and radii, partitioned by the 0.45
  threshold: `hypercharge` menu max 0.212 / play min 0.968 / median 0.985, `super` 0.083 / 0.675 /
  0.770, `gadget` 0.336 / 0.454 / 0.978. `hypercharge` is the anchor, 2.1x clear below and 2.2x
  above. `super` has the cleanest menus but the policy fires it. The gadget's in-match floor is
  0.454 against a 0.45 threshold with nobody pressing it, so it was never usable, press hazard or
  no. No vote was added: `MatchState` already had one, and `match_state.py` changed only in its
  docstring.
- **The Verify item paid off immediately.** The real-footage gate test,
  `test_gate_separates_gameplay_from_menus_on_real_footage`, had failed on `gameplay floor 0.675 too
  close to the threshold` since A2, with no overhaul change feeding it. 0.675 is the Super's own
  in-match floor on that clip: its face changes as it charges, so its ring score dips, and the gate
  was sitting on it. Moving the anchor to `hypercharge` fixes it at the root; the test now passes.
  Its bounds are restated against the threshold (floor > 1.7x, ceiling < 0.70x, ratio > 4.0) and its
  docstring carries the measured numbers and the reason they moved. This closes the known failure
  that was not to be re-diagnosed; it did not need diagnosing, it needed the anchor moved off an
  animating button.
- The live trace then confirmed the choice and showed the hazard was real on the other disc.
  `hypercharge` held 0.991 with a floor of 0.991 over the whole 22 s, zero ticks under threshold,
  zero colour movement, while the real gadget false-exited at +0.58 s and stayed under 0.45 for 60
  consecutive ticks, 5.0 s, bottoming at 0.164 at +18.4 s as the recharge sweep completed. Renaming
  the entries without also moving the gate would have built exactly the failure G6 was written to
  prevent.
- **Read the trace's two untouched columns carefully; "held" understates and possibly overstates
  them.** `hypercharge` and `super` are BIT-IDENTICAL across all 264 ticks -- exactly one distinct
  score and one distinct mean colour each, both equal to that anchor's own `refine_score` -- while
  the tapped gadget has 111 distinct scores and 9 distinct colours. `probe_gadget` builds a real
  `MatchState` per button and `record_gadget_trace` calls `update` and `_disc_colour` on every one
  every tick (there is no caching in `MatchState.update`), and `tick_ms` varies, so the loop did do
  the work on fresh grabs. Two readings fit and the trace cannot separate them: the capture is
  lossless and those HUD regions were pixel-for-pixel static for 22 s, which is the strongest form
  of "did not move" available; or something upstream served a stale ROI for the non-tapped anchors.
  What argues for the second is that `ring_score_at` samples AT radius r, on the disc edge, where
  moving world pixels are adjacent -- and on the fixture clip the same anchor's score does vary
  (play min 0.968, median 0.985). The frames were not retained, so this is not decidable after the
  fact.
- **This does not touch the anchor decision, and it does block the replay test.** The choice rests
  on the bakeoff and on the disc being inert, neither of which cites the trace. But a replay test
  over a column with one distinct value asserts nothing, so do not write one against this file.
  Settle it first. `probe_gadget` now records a blake2b digest of each anchor's ROI on every tick
  (`_roi_digest`, 2026-09-22, +1 test), so the next trace separates the two: identical digests mean
  identical pixels, and a digest that MOVES under a frozen score is a bug in the scorer. Re-run the
  probe and build the replay fixture from that trace, not this one.
- Files: `brawl_deployment/data/control_calibration.json` (three renames, `match_gate.anchor` ->
  `hypercharge`, both `_comment` blocks rewritten with the bakeoff and the live trace),
  `scripts/deploy_calibrate.py` (`TAPPED_ANCHOR`, the docstring protocol, the live log line),
  `configs/deployment.yaml` (two clearance comments, and the aim-radius ceiling is 179 px rather
  than 81, because the 81 px disc was never the Super), `brawl_deployment/match_state.py` and
  `brawl_deployment/control/buttons.py` (docstrings), design 5 and the new 5.1, and the
  deployment skill.
- Tests: `tests/test_deployment_control.py` +1,
  `test_the_gate_never_anchors_on_a_button_the_policy_can_press`, which is the standing guard. It
  cannot catch a fresh mislabelling, because no test can, but it catches what made this one harmful.
  `tests/test_deployment_calibration.py` is retargeted throughout: the probe fixtures now darken
  `hypercharge` to close the gate and darken the gadget to exercise the non-gate button path, which
  before 2026-09-22 could not be expressed at all, since a dark gadget WAS a closed gate.
- **Still open from "Done when", and neither blocks the anchor.** The trace-replay test is not
  written: `runs/` is gitignored and the trace has not been copied under `tests/fixtures/`. The
  `control.backend: null` dry run is in the I5 list at the end of this file. The live trace is
  stronger evidence than the replay test would have been, since it is the real button under a
  real throw, but the replay test is what keeps it from regressing.
- **The anchor's safety is conditional on the roster, confirmed by the operator 2026-09-22.** The
  third disc is the HYPERCHARGE button, and Mortis has no hypercharge, so it is drawn but inert: it
  never fills, sweeps or animates. That is stronger than the property the gate was chosen on. Give
  Mortis a hypercharge and it gains a charge meter that fills during a match, which is exactly what
  makes the Super unusable here, so re-measure the gate if the brawler or its unlocks change.

### G6.3 The live `--jobs super` check, and the readback defect it exposed

**DONE 2026-09-22.** The one check the rotation post-mortem said was still owed: no run had ever
discriminated a wrong super point from an uncharged Super, because `job_super` reports both as
"skipped". Run against a CHARGED Super, on the corrected labels.

- **The corrected mapping is confirmed live.** The press went to `(1559.9, 901.1)`, the disc this
  file has called `super` since the rename, and the operator watched the Super fire at exactly 90
  degrees right, which is the commanded `AIM_BEARING`. The names are right and the aimed drag lands.
- **The job printed FAIL anyway, and the job was wrong.** It slept a flat 0.8 s, took ONE reading
  and scored it. Deployment observables lag the input by about a second (design 6.8, and the ammo
  path's own `TROUGH_WINDOW_S`), so 0.8 s sat under the floor. Mortis's super is also a dash, so
  the hero detector can lose him mid-flight, and `ok = after is not None and not after.ready`
  scored a lost detection as a failed super.
- **Fixed by polling rather than guessing.** `SUPER_DRAIN_WINDOW_S = 3.0` at
  `SUPER_SAMPLE_S = 0.1`; PASS is the first not-ready frame and the latency it arrived at is
  printed, so the job now MEASURES the lag instead of assuming it. Unreadable frames are counted
  and skipped, never scored. A timeout still FAILS, and prints the charge trajectory so the next
  such failure reads itself.
- Files: `scripts/deploy_calibrate.py` (`job_super`, two new constants).
- Tests: new `tests/test_deploy_calibrate.py`, four cases on a fake clock and the real
  `SuperReading` -- a drain at 1.2 s passes (the old code fails it), unreadable frames decide
  nothing, a track that never drains fails with its trajectory, and an uncharged Super is skipped
  with no press. That last one pins the hole the rotation hid in as a deliberate property rather
  than an accident.
- **The ammo path had already learned this and the super path had not.** `verify_tap` samples a
  0.9 s window at 50 ms and looks for a trough; `job_super` slept once. One file, two theories of
  how fast the game answers. If a third verifier is ever added, poll.

---

# Phase H -- three frames of local history (plan §5)

No memory, but the last three decisions: own action, hp, ammo, displacement, and where revealed
enemies were, as three extra grid channels restricted to a 4-tile block.

Read first: plan §0 S13/S14 and A-H1, `brawl_sim/core/observation.py` (`_build_grid`,
`_scatter_count`), `brawl_sim/core/obs_select.py` (flatten path), `brawl_sim/env.py` (`step`,
`_build_observation`, autoreset), `brawl_deployment/perception/assemble.py`,
`brawl_deployment/perception/grid.py`.

## Step H1 -- ring buffers and the push

**DONE 2026-09-18.** As written, plus: every `state.py` shape lambda now takes a seventh arg `K`
(= `cfg.history_frames`), read only by the new `_HISTORY_FIELDS` table; the rings sit in
`_ALL_FIELD_SPECS` before `_CACHED_FIELDS`, so `zero_` (and therefore `spawn.reset_envs`) clears
them for free. `hist_enemy_pos`/`hist_enemy_seen` keep EVERY entity along the E axis (slot 0 = the
hero, whose `seen` column is forced False) so H2 can gather instead of loop. `env.step` pushes after
the dtype/device cast and before `_run_decision`, with an on-the-fly `perception.visibility` if
`_obs_vis` is still None (a step before any observation; nothing does that). `tests/test_env_step.py`
does not exist; the timing pins are `tests/test_env.py` + `tests/test_action_repeat.py`, both green.

### H1.1 Config
- Files: `brawl_sim/config.py` (`EnvConfig`, `PER_ENV_FIELDS`), `configs/default.yaml`
- Do: `history_frames: int = 3`, `history_radius_tiles: int = 4`; YAML under an `observation:` (or
  the existing obs block) with comments.
- Verify: `tests/test_config.py`
- Done when: both load and `history_frames >= 1` is validated.

### H1.2 State rings
- Files: `brawl_sim/core/state.py` (`_STATE_FIELDS`)
- Do: with K = `history_frames`, newest at index 0: `hist_valid (N,K) bool`, `hist_action (N,K,2)
  i64`, `hist_hp (N,K) f32`, `hist_ammo (N,K) f32`, `hist_pos (N,K,2) f32`, `hist_enemy_pos
  (N,K,E,2) f32`, `hist_enemy_seen (N,K,E) bool`. All zero at `zero_` (= no history).
- Verify: `tests/test_state.py`
- Done when: invariants pass and autoreset's per-row zeroing covers the rings (check how autoreset
  clears rows; if it enumerates fields, add these).

### H1.3 The push
- Files: `brawl_sim/core/history.py` (new)
- Do: `push(state, action, vis)`: shift every ring by one along K (two slice copies for K = 3) and
  write slot 0 from the pre-step state: `valid = 1`, `action` = the policy's `(N,2)` action, hero
  hp, hero ammo, hero pos, every entity's pos, `seen = alive & vis[:, 0, :]` (slot 0 = hero always
  false). Slices and `where` only; no host reads.
- Verify: H1.5.
- Done when: the unit test passes.

### H1.4 Wire into `step`
- Files: `brawl_sim/env.py` (`step`, `_build_observation`)
- Do: `_build_observation` stashes the `(N,E,E)` visibility it computes as `self._obs_vis`; `step`
  calls `history.push(state, action, self._obs_vis)` before `_run_decision`. Outside `_run_tick`.
- Verify: H1.5; `tests/test_env_step.py` timing pins still hold.
- Done when: green.

### H1.5 Tests
- Files: `tests/test_history.py` (new)
- Do: after reset `valid = [0,0,0]`, then `[1,0,0]`, `[1,1,0]`, `[1,1,1]`; `hist_action[:,0]` equals
  the last `step()` argument; `hist_pos[:,0]` equals the hero position of the previous observation;
  an autoreset row goes back to all-zero; rings are per decision regardless of `action_repeat`.
- Verify: the file.
- Done when: green, then `graphify update .`.

## Step H2 -- observation fields and the three grid channels

Depends on G3 (attack one-hot width 4).

**DONE 2026-09-21.** `build_obs` emits a `hist` group right after `hero`: `valid`, `move_onehot`,
`attack_onehot`, `hp`, `ammo_frac` and `displacement`, each `(N, K, ...)` with K = `history_frames`,
newest first, every field 0 where `valid` is False. Because H1's push runs at the top of `env.step`,
slot k of an observation is the observation k+1 decisions back plus the action that answered it, and
the grid names that slot's plane `enemy_hist{k+1}`. `_build_grid` appends one plane per slot after
the 12 base channels, scattering past sightings (`hist_enemy_seen`) at their world position into the
current window. `OBS_SCHEMA` gains the six rows and two dims, `K` = `history_frames` and `C` = 12 +
`history_frames`, and declares `view`/`world` as `(N, C, ...)`. `obs_select._build_flat_group` now
flattens any rank row-major (`_flat_columns`); before, the probe showed a rank-3 field raising
`RuntimeError: Tensors must have same number of dimensions: got 2 and 3`. The six fields flatten to
78 columns: valid [0, 3), move [3, 54), attack [54, 66), hp [66, 69), ammo [69, 72), displacement
[72, 78). Deviations and choices:

1. **The planes follow K.** R6 makes `history_frames` a real knob, so the grid has 12 + K channels
   rather than a fixed 15. `obs_select._CHANNEL_INDEX` keeps the 12 fixed names; the new public
   `obs_select.channel_index(cfg)` adds `enemy_hist1..K` at 12..11+K, and grid specs resolve through
   it. `observation._N_BASE_CHANNELS`, obs_schema's `C` and `_CHANNEL_INDEX` each restate the 12,
   and the tests pin all three.
2. **The radius is Chebyshev on tile indices:** the enemy's tile then against the hero's tile now,
   `max(|dx|, |dy|) <= history_radius_tiles`, which is exactly the 9 x 9 block of cells. The text's
   `chebyshev(hist_enemy_pos - hero.pos)` on raw positions would take a boundary cell or not
   depending on where inside their tiles the two stood: an enemy at x 14.9 with the hero at x 10.1
   is on the block's edge but 4.8 away. H4.3 can reproduce a cell rule exactly, and not the other.
3. **The one-hots are `uint8` with units `onehot`**, like `entities.kind_onehot`. Specs read them as
   float32 and `normalize` divides `onehot` by 1.0, so nothing downstream sees the difference.
4. **H2.2's "fix if it does not"**: it did not, and `_flat_columns` is the fix.
5. **`hist` sits right after `hero`** in `build_obs`, `OBS_SCHEMA` and `dump_obs_schema.py`'s group
   order. The whole schema is now pinned to `build_obs` order, not only the hero rows.
6. **Every field is masked by `valid`, and the planes AND `hist_valid` in as well.** For the
   one-hots and the displacement the mask is load-bearing: an empty slot holds action (0, 0), a real
   "idle, no attack", and position (0, 0), which would read as a displacement of `-hero.pos`. For hp
   and ammo it is defensive, since `state.zero_` already leaves 0 there.
7. **The one-hots are a comparison with an arange**, not `F.one_hot`, so the mask folds into the
   same expression and no index can raise.
8. **Deployment is untouched.** The assembler still builds a 12-channel view, which is right for
   every shipped spec: none selects `hist.*` or `enemy_hist*` before I1, and one that does fails
   loudly (`KeyError` in `assemble._channel_indices`) until H4.2 and H4.3 land. The one deployment
   file edited is a docstring: `tests/test_deployment_grid.py`'s `_sim_view` said 12 channels.

`displacement` normalizes like every other `tiles` field, by `max(map_w, map_h)`, so one tile reads
0.017 on the 60-tile default map. `docs/OBSERVATION.md` is regenerated (gitignored);
`docs/AGENT_OBS.md` is byte-identical. The CUDA sync test (`test_native_step_is_sync_free_on_cuda`)
is green. Tests: `tests/test_observation.py` +11 (empty history at reset in the declared shapes;
every field 0 in the empty middle slot of `[T, F, T]`; three `env.step`s at repeat 5, each slot 0
equal to the previous observation's hp, ammo fraction and position minus the current one; an enemy
seen two decisions back at (10.2, 10.7) lands only in `enemy_hist2` at view (4, 5) and world (10,
10); a hidden sighting and a sighting in an empty slot draw nothing; the block keeps two tiles 4
away that stand 4.4 and 4.8 away and drops two tiles 5 away inside the view; the radius counts from
the hero's tile now, not then; after one idle step `enemy_hist1` equals the previous
`enemy_revealed` plane cut to the block; K 1 and 4 give 13 and 16 channels),
`tests/test_obs_schema.py` +7 (the six rows as declared; the whole schema in `build_obs` order; K
and C at 3, 1 and 4, with `validate_obs`; `validate_obs` refuses each `hist` field missing; the docs
put `hist` between `hero` and `entities` and define K and C) and `tests/test_obs_select.py` +6 (78
wide with the column map above; row-major on a hand-built package; the normalize vector, 1.0 x 66,
20000 x 3, 1.0 x 3, 20 x 6 on debug_tiny; nothing before the first step and slot 0 after it on a
real env; `channel_index` at K 3 and 1; a grid spec selecting (6, 12, 14) and refusing
`enemy_hist4`). The five 12-channel shape assertions, three in `tests/test_observation.py` and two
in `tests/test_obs_schema.py`, read 15. Seventeen mutants, all killed: the one-hots, displacement,
hp or ammo fraction unmasked, the displacement's sign flipped, the radius on raw positions, from the
hero then, or strict `<`, the planes without `hist_valid`, without `hist_enemy_seen`, in reverse
slot order or drawing current positions, a column-major flatten, the pre-H2 flatten, `channel_index`
off by one, `C` fixed at 15, and two schema rows swapped. Full `-m "not vision"` suite: 2280 passed,
4 skipped, 85 deselected, in 20 min 17 s.

### H2.1 Float fields
- Files: `brawl_sim/core/observation.py` (`build_obs`), `brawl_sim/core/obs_schema.py`
- Do: `hist.valid (N,K) bool`, `hist.move_onehot (N,K,17)`, `hist.attack_onehot (N,K,4)`, `hist.hp
  (N,K) hp`, `hist.ammo_frac (N,K) fraction`, `hist.displacement (N,K,2) tiles` = `hist_pos -
  hero.pos`, all zero where `valid` is 0. Supplier at deployment: H4.2.
- Verify: `tests/test_observation.py`, `tests/test_obs_schema.py`.
- Done when: green.

### H2.2 Flatten of rank-3 fields
- Files: `brawl_sim/core/obs_select.py`
- Do: confirm the group flatten turns `(N,K,17)` into `K*17` columns in row-major order (today's
  rank-2 fields flatten one trailing dim); fix if it does not. The `history` group then flattens
  to `3 * (1 + 17 + 4 + 1 + 1 + 2) = 78`.
- Verify: `tests/test_obs_select.py` -- a spec with the six `hist.*` fields yields width 78.
- Done when: pinned.

### H2.3 Grid channels
- Files: `brawl_sim/core/observation.py` (`_build_grid`), `brawl_sim/core/obs_select.py`
  (`_CHANNEL_INDEX`)
- Do: channels 12..14 `enemy_hist1..3` = `_scatter_count` of `hist_enemy_pos[:, k]` masked by
  `hist_enemy_seen[:, k] & (chebyshev(hist_enemy_pos[:, k] - hero.pos) <= history_radius_tiles)`
  into the current `view_origin`. World positions into the current window; no re-centering.
- Verify: `tests/test_observation.py` -- an enemy revealed at world (10, 10) two decisions ago shows
  in `enemy_hist2` at (10, 10)'s cell in the current view and nowhere else; hidden-then shows
  nowhere; 5 tiles away then shows nowhere; every 12-channel assertion updated to 15.
- Done when: pinned; `scripts/dump_obs_schema.py` regenerates the docs.

## Step H3 -- extractor width test

Depends on I1.

**DONE 2026-09-21.** `tests/test_sb3_features.py` pins both deploy specs, not only the one the
next run trains. `DEPLOY_WIDTHS` writes out each one's CNN `in_channels`, MLP `in_features` and
per-group float widths as literals, never read back from the spec, and
`test_a_deploy_spec_builds_the_extractor_at_its_pinned_widths` builds `BrawlFeaturesExtractor`
from each at the default config and runs a batch through it. Deploy4: 13 channels and 262
floats, the sum of `self` 26, `enemies` 81, `projectiles` 72, `zone` 5 and `history` 78.
Deploy3: 10 and 182. No extractor change. Deploy3 is there for more than the record: deploy4's
grid is 13 x 13 x 21, so an extractor that took its channel count from the view-height axis
would still build 13 channels for it, and only deploy3's 10 tell the two axes apart. Tests: +2,
the one test over both specs. Two mutants, both killed: the channel count read from the
view-height axis, and the last float group dropped from the MLP.

### H3.1 Test
- Files: `tests/test_sb3_features.py`
- Do: build `BrawlFeaturesExtractor` from `configs/agent_obs_deploy4.yaml`; assert the CNN's
  `in_channels == 13` and the MLP's `in_features` equals the spec's float width. No extractor code
  change expected.
- Verify: the file.
- Done when: pinned.
- Amended by I1 (2026-09-21): measured on deploy4 with no extractor change: CNN `in_channels` 13,
  MLP `in_features` 262, the sum of `self` 26, `enemies` 81, `projectiles` 72, `zone` 5 and
  `history` 78, and 384 features out. Deploy3 gives 10 and 182. Pin the float width, not the plan
  table's field counts (I1's note, deviation 1).

## Step H4 -- deployment mirror

**DONE 2026-09-21.** `DeployLoop` keeps a `deque(maxlen=history_frames)` of `DecisionSnapshot`s, a
frozen dataclass in `assemble.py`: the move bin, the modelled attack, the hero HP read, the shadow's
`ammo_frac`, the hero's world position, and the world positions of the enemies seen that decision.
It pushes one per decision, newest first, right after `shadow.act`, and empties at the match gate.
`ObservationAssembler.assemble` takes a required `history`; `_put_history` writes the snapshots into
`core/history.py`'s ring layout and runs the sim's own expressions on them, `observation._onehot`
included. `GridSpec` gains `history_frames` and `history_radius_tiles`, `history_slots` maps
`enemy_hist{k}` to snapshot k - 1, and `GridBuilder.build(..., enemy_history=...)` draws each past
sighting inside the sim's block: tile floors, Chebyshev, `<= history_radius_tiles`, the enemy's tile
then against the hero's tile now. `assemble._N_VIEW_CHANNELS` and `_CHANNEL_INDEX` are gone, and the
view is `obs_select.channel_index(cfg)`'s 15 channels. Deploy4 is served: the zone file's
`ASSEMBLER_NOT_BUILT_YET` and the loop file's `NOT_SERVED_YET` are deleted with their machinery, and
both files run every deploy spec plainly. Deviations and choices:

1. **`enemies` holds only the tracks seen that decision**, not "every tracked enemy" as H4.1 says.
   The sim's `hist_enemy_seen` is alive AND revealed, which on this side is `Track.seen_now`, the
   set `enemy_revealed` draws. A coasted track is the tracker's prediction: it keeps its slot and
   stays out of the snapshot.
2. **The ring also empties on a new odometry segment.** A snapshot's position is in its segment's
   world frame, a new segment has no defined offset to the old one, and the tracker drops its tracks
   for the same reason. The check runs where the snapshots are read. A new segment's first decision
   is the tracker's reset tick and is skipped, so the one after reads `valid = [0,0,0]`.
3. **A skipped decision pushes nothing.** `valid` stays a prefix, as it always is in the sim, and
   the next decision's first slot is the last one that reached the policy, two windows back.
4. **No "last good" HP carry.** `_decide` skips a tick with no hero HP read before it assembles
   anything, so every snapshot holds a real read and the carry has nothing to carry.
5. **`ammo_frac` is what the decision's observation read, before its own shot.** The sim pushes the
   ring at the top of `env.step`, so `hist_ammo` is the clip the previous observation was built
   from. The loop reads `shadow.observe()` once per decision for both. `ShadowHero.act` only queues
   the shot, so a second read after it would agree today; the test pins the contract through the
   next observation instead, with the shot spent in between.
6. **The attack is the one the shadow modelled.** The sim records the requested action, and under
   its mask a requested attack always fires. Here the policy can pick an attack the shadow refuses,
   and the snapshot holds what was pressed.
7. **The loop hands `enemy_history` to `GridBuilder.build`**, not `_put_view` as H4.3's Files line
   says. The assembler receives a built grid, so the loop is the one caller holding both the builder
   and the snapshots.
8. **A missing history is never a default.** `enemy_history=None` raises for a spec with history
   planes, since an empty plane claims nobody was seen; a spec without them ignores the argument.
   More snapshots than `history_frames` raise in both the grid and the assembler.
9. **`history` is a required assembler keyword for every spec**, deploy3's included, and the `hist`
   group is always built; `obs_select` drops it for a spec that does not select it.
   `tests/test_deployment_policy.py`'s seam test passes `history=()`.
10. **`GridSpec` gained two required fields**, which `load` reads from the config's raw
    `observation` block, so every spec agrees with the sim's `history_frames: 3` and
    `history_radius_tiles: 4`.
11. **Two private names are borrowed from the sim**, `observation._onehot` and
    `obs_select._HISTORY_CHANNEL_PREFIX`, rather than defined a second time.
12. **Snapshot positions are plain floats**, converted from the tracker's numpy scalars, so a
    snapshot compares and logs exactly.
13. **The deque's `maxlen` is the loop's `sim.history_frames`**, the field the assembler and the
    grid check against.
14. **The grid parity run's coverage comment was re-measured.** The sim's random stream had moved
    since it was written: `box` 30, `pickup` 33, `enemy_revealed` 37, `in_zone` 14 and `projectile`
    21 of forty decisions, and deploy4's three history planes 16, 14 and 12. Design §6.2 carries the
    same numbers.

Docs: `BRAWL_DEPLOYMENT_DESIGN.md` §6.2 (the parity run covers deploy4, and a dated paragraph on the
history planes) and §6.3 (a dated `history` subsection with a table of the six fields and their
sources); the skill file's history section; `configs/agent_obs_deploy4.yaml`'s supplier section,
which now says deployable since H4 and still names G6. Tests: `tests/test_deployment_grid.py` +15
(deploy4 in the `from_agent_spec` and parity cases, and the yaml test pins `history_frames` 3 and
`history_radius_tiles` 4; deploy4's spec is deploy3's ten planes then one per history frame; a plane
the config does not keep is refused four ways; the plane count is the config's; a past enemy lands
where the sim's `_scatter_count` puts it, from hand-written rings with a garbage slot and a dead
enemy; plane k is k decisions ago; the block's edges on both axes; the block follows the hero while
the sighting stays; two in one cell count two; a `None`, empty and over-long history; a grid without
the planes ignores them), `tests/test_deployment_assemble.py` +7 (synthetic snapshots byte-equal to
`build_obs` and `obs_select` at zero to three valid slots, with the one-hots and floats pinned as
literals at three; an over-long history refused; `history` among the required suppliers; the view
scatter over deploy4's 15 channels; and G5's deploy4 test now assembles the whole spec over six live
steps, history byte-equal and `valid` checked per step) and `tests/test_deployment_loop.py` +8
(H4.1's verify through the real assembler, newest first by a different HP at each decision; the
modelled attack; the ammo before the shot; the world position as (x, y); a new match; a new segment;
a skipped decision; only `seen_now` enemies, down to the planes). The zone round trip and the loop's
decision test run deploy4 plainly. Twenty-three mutants, all killed: in the grid, the block's `<=`
as `<`, the block on raw distance, a Euclidean block, the hero's tile rounded, no block at all, the
planes read oldest first, `None` defaulting to empty, no over-length refusal, and the channel check
admitting any `enemy_hist` name; in the assembler, `valid` all true, the displacement's sign
flipped, the displacement unmasked, the slots filled oldest first, the position read (y, x), and no
over-length refusal; in the loop, the chosen attack recorded instead of the modelled one, no clear
on a new segment, no clear at the gate, `append` for `appendleft`, coasted enemies kept, every plane
drawn from this decision's sightings, the position stored (y, x), and each snapshot's HP copied from
the one before. Two needed a test first: `append` until the H4.1 test read a different HP at each
decision, and the stored (y, x) until a test put the hero at world (3, 1). Full `-m "not vision"`
suite: 2345 passed, 4 skipped, 85 deselected, no xfails, in 20 min 29 s.

### H4.1 Snapshots
- Files: `brawl_deployment/loop.py` (`DeployLoop`)
- Do: a `deque(maxlen=history_frames)` of per-decision snapshots taken after `shadow.act`:
  `(move_bin, modelled attack, hero hp read or last good, shadow ammo, hero map-frame pos, [enemy
  map-frame pos for every tracked enemy])`. Cleared at match-gate entry. A missed HP read carries the
  previous value.
- Verify: `tests/test_deployment_loop.py` -- the first three decisions after the gate carry
  `valid = [0,0,0], [1,0,0], [1,1,0]`.
- Done when: pinned.

### H4.2 Assembler
- Files: `brawl_deployment/perception/assemble.py` (`_put_history`, a `history` supplier)
- Do: build the 78-float vector from the deque exactly as the sim does (one-hots, normalized hp,
  ammo fraction, displacement in the `tiles` unit).
- Verify: `tests/test_deployment_assemble.py` -- three synthetic snapshots produce the same vector
  `build_obs` + `obs_select` produce for the same numbers (build both, compare).
- Done when: byte-equal.
- Amended by H2 (2026-09-21): an empty slot is all zeros, one-hots included; `hist.hp` is the
  absolute HP (`normalize` divides by 20000); `displacement` is the position then minus the
  position now, in tiles. The column map is H2's note.

### H4.3 Grid
- Files: `brawl_deployment/perception/grid.py` (`GridBuilder.build(..., enemy_history)`),
  `brawl_deployment/perception/assemble.py` (`_put_view` passes it)
- Do: scatter past map-frame enemies into `enemy_hist_k` with the same 4-tile block.
- Verify: `tests/test_deployment_grid.py` -- a past enemy at map (x, y) lands in the same cell the
  sim's `_scatter_count` puts it.
- Done when: pinned.
- Amended by H2 (2026-09-21): the sim's view has `12 + history_frames` channels and the plane
  names come from `obs_select.channel_index(cfg)`. So H4.3 also moves `assemble._N_VIEW_CHANNELS
  = 12` and `assemble._CHANNEL_INDEX` onto `channel_index(cfg)`, widens
  `tests/test_deployment_assemble.py`'s `unfilled = set(range(12)) - ...` check (the enemy_hist
  planes become filled ones), and gives `GridBuilder` the sim's block exactly: tile indices,
  Chebyshev, `<= history_radius_tiles`, the enemy's tile then against the hero's tile now
  (`observation._history_drawn`).
- Amended by I1 (2026-09-21): once `_CHANNEL_INDEX` moves, the assembler builds for deploy4.
  `tests/test_deployment_zone.py`'s round trip builds it and calls only `_put_zone`, so H4.3 deletes
  its `ASSEMBLER_NOT_BUILT_YET` entry; the strict XPASS says so. `tests/test_deployment_loop.py`'s
  decision case then fails later, on the first field the loop cannot yet supply, and that need not
  be a `KeyError`: `assemble._require` raises `ValueError`. Restate `NOT_SERVED_YET`'s exception in
  the same change, or delete the entry if the case passes (G5.1's amendment).
- Amended by G5 (2026-09-21): G5.1 is in and the shadow supplies `self`'s gadget pair, so H4 is the
  last step the loop's deploy4 case waits on. The test's comment and xfail reason already name only
  H4.

---

# Phase M -- maps, and no fences (plan §6)

Ten generated maps in the game's style; fences leave the sim's map vocabulary.

Read first: plan §1.4 (tile shares, the screenshot motifs), `brawl_sim/maps/README.md`,
`brawl_sim/maps/loader.py` (`validate_map`, `bush_waypoints`), `tests/test_map_csvs.py`
(`DIMENSIONS`), `brawl_deployment/perception/grid.py` (class table).

## Step M1 -- remove fences from the sim

**DONE 2026-09-18.** One deviation from the plan text: `brawl_vision` reads both `CHAR_TO_TILE`
and `TILE_TO_CHAR` and its label files carry `f`, so the alphabet stays complete and a separate
`MAP_CHAR_TO_TILE` (alphabet minus FENCE) is what the loader and the map tests read. Vision is
untouched, as intended. The three CSVs converted with 22 / 36 / 27 cells becoming wall and
51 / 116 / 0 becoming floor (feast_or_famine / scorched_stone / walled); every map still has one
unit-passable component.

### M1.1 Vocabulary
- Files: `brawl_sim/constants.py`, `brawl_sim/maps/loader.py`
- Do: add `MAP_CHAR_TO_TILE` = `CHAR_TO_TILE` minus FENCE; the loader reads it. Keep `Tile.FENCE = 4`
  and its `TILE_BLOCKS_UNIT` entry with the comment "not a sim tile since 2026-09; retained because
  brawl_vision's terrain classes and the trained classifier index it".
- Verify: `tests/test_constants.py`
- Done when: `load_map_csv` raises on an `f` cell (M1.4 pins it).

### M1.2 Convert the CSVs
- Files: `brawl_sim/maps/csv/feast_or_famine.csv`, `scorched_stone.csv`, `walled.csv`
- Do: `f` -> `#` where the cell 8-touches an existing `#`, else `f` -> `.`; `walled`: all `f` -> `#`.
  Use a throwaway script in the scratchpad, not a repo file. Render each with the ASCII renderer and
  check no corridor closed (scorched_stone's 3x3 fence blocks become 3x3 wall blocks).
- Verify: `validate_map` on all three; `.venv/Scripts/python.exe -m pytest tests/test_map_csvs.py -q`.
- Done when: no `f` under `brawl_sim/maps/csv/`.

### M1.3 README and renderers
- Files: `brawl_sim/maps/README.md`, `brawl_sim/render/ascii.py`
- Do: delete the fence legend row; add "fences are not a sim tile" under design rules; remove the
  `f` char from the ASCII table if it maps chars (tile-keyed entries may stay).
- Done when: the README has no `f` legend.

### M1.4 Tests
- Files: `tests/test_map_csvs.py`, `tests/test_maps.py`
- Do: `test_walled_has_water_and_fences` -> `test_walled_has_water`; new
  `test_no_map_contains_fences` over every CSV; a test that `load_map_csv` raises on `f`.
- Verify: both files.
- Done when: green.

### M1.5 Deployment class table
- Files: `brawl_deployment/perception/grid.py`, `tests/test_deployment_grid.py`
- Do: the FENCE row of the class -> `[blocks_unit, blocks_projectile, is_bush, is_water]` table
  becomes water's `[1, 0, 0, 1]`, with a comment: the policy never trains on
  `blocks_unit & ~blocks_projectile & ~is_water`; water is the trained-on combination with a fence's
  physics. Vision is untouched.
- Verify: `tests/test_deployment_grid.py` (a FENCE cell yields water's row);
  `.venv/Scripts/python.exe -m pytest tests/test_vision_*.py -m "not vision" -q` still green.
- Done when: both green, then `graphify update .`.

## Step M2 -- the generator

**DONE 2026-09-18.** `brawl_sim/maps/generate.py`, `tests/test_map_generate.py` (28 tests),
`scripts/gen_maps.py`. Every family converges at the first seed for seeds 1/100/200/300/400 in
~0.2 s a map. Deviations from the text above, each measured rather than chosen:
- **Pocket threshold is 24 of 85 (0.28), not 40 (0.47)**, scaled by what is in bounds so a corner
  cell is judged against its 28. At 0.47 every 2-wide lane between two features (a channel two
  rows inside the moat, two clusters a margin apart -- reach 26) was a "pocket" and 10 of 11
  seeds failed; the reference maps are full of such lanes and `unit_radius` is 0.4. 0.28 still
  rejects a 2x3 notch (0.24) and a dead-end lane (0.18); `test_pocket_fractions_flag_a_dead_end_lane_but_not_a_two_wide_lane_between_features` pins both sides.
- **No T-shaped clusters, and an L arm only off a side >= 5 long**: T stems make 2-wide notches
  that are pockets by the check, so every T map was reseeded. Channels are 6-12 long, not 6-14.
- **The moat sits 4 tiles in with a 3-wide floor rim outside it.** Flush against the border the
  plan's four gaps lead nowhere; a rim narrower than 3 fails the pocket check by construction.
  The moat alone is ~11% water, a pond or two brings the family into 12-20%.
- **Solid stamps keep a 2-cell margin** from each other and the border (bush may touch anything),
  so no 1-wide corridor exists anywhere -- the loader's connectivity check is tile-level and
  would pass one. Repair runs BEFORE spawns and crates so it can never fill a marker.
- `generate(seed, family, symmetry=None)`: the symmetry override is how the standard family's
  one mirror map is made; `stats["rejected"]` carries `(seed, reasons)` for the README.

### M2.1 Grid primitives
- Files: `brawl_sim/maps/generate.py` (new)
- Do: a 60x60 char grid type; `border()` (ring of `#`); optional 2-tile `~` moat with four 3-tile
  floor gaps, one per side, off-centre; `mirror_point((r,c)) = (59-r, 59-c)` and
  `mirror_lr((r,c)) = (r, 59-c)`; a `stamp(cells, symmetric)` that writes a cell set and its mirror.
- Verify: `tests/test_map_generate.py` (new) -- symmetry helpers, border, moat gap count.
- Done when: pinned.

### M2.2 Stamps
- Files: `brawl_sim/maps/generate.py`
- Do: wall cluster (rect 2x2..4x6, optional L/T, 1-tile bush skirt on 1-2 sides); pond (filled
  ellipse r 2-4, bush rim on 50% of the perimeter); channel (1-2 wide, 6-14 long, straight or one
  bend); bush patch (blob 6-30 cells); crate spot (`X` single or pair, 1 tile from cover); centre
  feature for point-symmetric maps (2x2 wall + 4 bushes, small pond, or open cross). All seeded via
  `random.Random(seed)`.
- Verify: each stamp's cell count and bounds in the test.
- Done when: pinned.

### M2.3 Spawns and repair
- Files: `brawl_sim/maps/generate.py`
- Do: 16 `S` on a ring 6-8 tiles inside the border at near-uniform angles, nudged to the nearest
  floor cell; repair pass: punch a 2-tile gap in any straight wall run > 8; fill 1-wide dead ends;
  carve the shortest floor path between the two largest components if more than one.
- Verify: test -- a hand-built grid with a 12-run and two components comes out with one component
  and no run > 8.
- Done when: pinned.

### M2.4 Validators and the loop
- Files: `brawl_sim/maps/generate.py`
- Do: `Family` table with the style bands from plan §6 Step M2 (standard 6: wall 7-12%, bush 20-30%,
  water 3-8%, boxes 20-32; open 1: 4-6/12-18/0-3/16-24; dense 1: 8-12/32-38/2-5/24-36; water-border
  2: 6-10/20-28/12-20/20-32); pocket check (BFS radius 6 from every floor cell reaches >= 40 floor
  cells); bush spread (a bush cell in >= 80% of the 10x10 `hunt_cell_tiles` cells);
  `loader.validate_map`; `bush_waypoints <= 63`. `generate(seed, family) -> (grid, stats)` reseeds
  on failure and returns the seed that passed.
- Verify: test -- each family at a fixed seed passes every band; byte-identical across two calls.
- Done when: pinned.

### M2.5 CLI
- Files: `scripts/gen_maps.py` (new)
- Do: `--family --seed --count --out-dir --preview`; prints seed and shares per map; `--preview`
  renders ASCII.
- Verify: run it for one map of each family into the scratchpad.
- Done when: four previews produced.

## Step M3 -- the ten maps and the rotation

**DONE 2026-09-18; the operator approved the render review on 2026-09-21.** Ten maps in
`brawl_sim/maps/csv/`: `broken_wall`, `stone_fort`, `twin_ponds`, `cross_creek`, `split_river`
(standard, point, seeds 2/3/4/100/303), `narrow_pass` (standard mirror, 502), `dry_gulch` (open,
702), `thorn_field` (dense mirror, 800), `reed_marsh` and `hollow_ring` (water_border, 901/904),
picked from 26 candidates by reading every ASCII render against plan §1.4; the README has the
name/family/seed/symmetry/shares table, one line per rejected seed, and the swap procedure. Each
loads through `load_map_csv` + `validate_map`, has 16 spawns, and `generate(seed, family,
symmetry=)` reproduces it byte-for-byte at attempt 1 (pinned). M3.3 was blocked inside the wave:
`brawl_sim/config.py`'s `_KNOWN_MAP_NAMES` is a literal tuple `validate` checks every `world.maps`
name against and no wave-2 step owned it, so the agent staged the flip (xfail(strict) rotation test,
a TEMPORARY registry monkeypatch for the smoke episodes). The orchestrator then finished it: the
ten are in `_KNOWN_MAP_NAMES`, `world.maps` lists sixteen, the marker and fixture are gone, and the
`scripts/watch.py --map` help string and README paragraphs say so. Existing checkpoints are now
mismatched, as the plan accepts. "1000-step" was read as 1000 SIM TICKS (200 decisions at
`action_repeat 5`): the sim costs ~14 ms/tick on CPU regardless of `n_envs` (kernel-launch bound),
so the ten smoke episodes add ~2 min to the suite; they draw both action columns over the full
`action_nvec` width (super included, and G3's fourth value automatically). Two water_border picks
have bush spread 0.94 / 1.00 and `reed_marsh` yields 34 waypoints, within the generator's band.
M4 suggestion: hold out one standard point map and one non-standard (e.g. `split_river` +
`hollow_ring`). Tests: `tests/test_map_csvs.py` +10 DIMENSIONS rows + a 16-spawn pin;
`tests/test_maps.py` +`GENERATED_MAPS` table, loads x10, reproduced-by-seed x10, the 16-map
rotation test, smoke x10.

### M3.1 Choose and save
- Files: `brawl_sim/maps/csv/<ten names>.csv` (new)
- Do: generate more candidates than needed; pick by ASCII render against plan §1.4's motifs;
  8 point-symmetric, 2 mirror; names in the repo's style (e.g. `twin_ponds`, `stone_fort`,
  `cross_creek`, `hollow_ring`, `dry_gulch`, `reed_marsh`, `broken_wall`, `quiet_lake`,
  `thorn_field`, `split_river`).
- Needs: human review of the ten renders (approved 2026-09-21).
- Done when: ten CSVs exist and load.

### M3.2 README table
- Files: `brawl_sim/maps/README.md`
- Do: table name / family / seed / wall / bush / water / boxes; a line per rejected seed and why.
- Done when: present.

### M3.3 Rotation and tests
- Files: `configs/default.yaml` (`world.maps`), `tests/test_map_csvs.py` (`DIMENSIONS`),
  `tests/test_maps.py`
- Do: sixteen maps in `world.maps`; a `DIMENSIONS` entry per new map; bank tests for
  `len(map_names) == 16`; a 1000-step smoke episode per new map with `check_invariants` on.
- Verify: both files.
- Done when: green, then `graphify update .`.

## Step M4 -- map-overfitting eval

**DONE 2026-09-18.** Holdout pair: `split_river` (standard, point) + `hollow_ring` (water_border),
confirmed by the operator on 2026-09-21.
`configs/default.yaml` stays at sixteen (the sim's full list); the TRAINING rotation is the other
fourteen via `configs/train.yaml` `run.env_overrides.world.maps` (a list override REPLACES, pinned
for both merges), with `eval.holdout_maps: [split_river, hollow_ring]`. **A map added only to
default.yaml never trains; it needs a train.yaml entry too.** M4.1: `EvalConfig.holdout_maps:
tuple | None` (the plan text's `map_names` name was not used); `None` and `[]` both mean no holdout
eval, so pre-M4 configs load. `validate_train_config` rejects, naming the map: an unknown name, a
duplicate, a bare string, an overlap with the RESOLVED training rotation (`load_config(env_config,
overrides=env_overrides).map_names`, the same call the env builder makes; checked at load and in
`with_overrides`), and a registered map that cannot load in the holdout world (`blank`, 20x20).
M4.2: `TierEvaluator(..., maps=None)` builds the holdout twin through the same code path (same
tiers, episodes, seed), forcing `map_selection: uniform` so a fixed-map training run cannot crash
it; `build_evaluators(tcfg)` returns `(training, holdout_or_None)` and `scripts/train.py` builds
both BEFORE the run directory exists. `TierEvalCallback` logs `eval/holdout_win_rate_<tier>`,
`eval/holdout_win_rate` (mean over tiers, plus the per-tier TensorBoard overlay under the same tag)
and **`eval/holdout_gap`** (training mean minus holdout mean; added beyond the brief because it is
the number the plan says matters -- read its trend from the `at_start` row, not its sign).
`best_model.zip` is chosen on the training-map mean only. `eval/*` now means the fourteen training
maps. `--smoke` turns the holdout eval off (debug_tiny's world is 20x20). Review found and fixed:
(major) `watch.py --save` replayed matches on `map_id >= 10` and on any `--map <holdout>` over the
WRONG terrain, because the viewer builds its bank from default.yaml -- `save_rollout` now writes
`<save>.preset.yaml` and prints the `--preset` replay command, and `watch_env_overrides` is the one
source for the env, the spec and the sidecar; `--map <holdout>` appends the map to the bank (so
`--map walled` works too); a pre-existing crash at the first eval of any `normalize.obs: true` run
(SB3's sync helper asserts equal wrapper depth) is fixed by copying `obs_rms` directly.
`tests/test_training.py` 80 -> 104, all literal; 21 mutations killed from scratch scripts.
**Unmeasured (GPU was not used):** VRAM with 4096 training envs plus two 1800-slot eval envs, and
eval wall-clock with two rollouts per eval; time the `at_start` eval of the next run. Open: the
viewer CLI cannot DETECT an npz replayed without its sidecar. If M3's render review swaps out either
holdout map, `eval.holdout_maps`, the fourteen-map list and the `TRAINING_MAPS` / `HOLDOUT_MAPS`
literals in `tests/test_training.py` change together. (The review swapped nothing, 2026-09-21.)

**Review 2026-09-21 (second round).** (1) `TierEvaluator.tcfg` is now the config its own env was
built from; the holdout twin used to name the training maps. Only `eval.deterministic` is read from
it, so no eval number moves. (2) `load_train_config(path, overrides=None, *, check_holdout=True)`:
`False` skips `_validate_holdout_maps` (registry, overlap, CSV loads) and nothing else.
`load_config` never checks map names (only `config.validate` does, from `BrawlVecEnv.__init__`),
so the holdout list was the one thing that could make an archived run unloadable after a holdout
map is renamed or retired. The loaders that never run a holdout eval opt out:
`DeployedPolicy.from_run` (keep the keyword when G5 or H4 edit it;
`test_an_archived_run_still_deploys_after_its_holdout_map_leaves_the_registry` fails otherwise),
and, added after the round, `scripts/watch.py` and the cadence audit
(`test_watch_and_the_audit_load_an_archived_run_without_the_holdout_check`; both strict-load
mutants fail it). `scripts/train.py` (start and resume) and `with_overrides` stay strict. A run
whose TRAINING map left the registry is still refused, by the env's own validation. (3)
`tests/test_training.py` 104 -> 113, all literal: the slow end-to-end test asserts the map list of
the env each of the callback's two evaluators BUILT (before, handing the training evaluator in as
the holdout passed the suite, with `eval/holdout_gap` a flat 0.0 for the whole run); the per-tier
TensorBoard overlay is pinned by value, not only by tag; the overlap test matches the overlap
message, not the map name. 19 in-memory mutants killed. If the render review renames or retires a
holdout map after a run archived it, deployment, watch and the audit still load that run and
`scripts/train.py --resume` refuses it, naming the map.

### M4.1 Config
- Files: `brawl_sim/training/config.py` (`EvalConfig.holdout_maps: tuple[str, ...] | None`),
  `configs/train.yaml` (`eval.holdout_maps`)
- Do: two of the ten held out; validation rejects a holdout map that is also in `world.maps`.
- Verify: `tests/test_training.py`
- Done when: pinned.

### M4.2 Eval callback
- Files: `scripts/train.py` (or the evaluation module the callback lives in)
- Do: build a second eval env over the holdout maps; log `eval/holdout_win_rate` next to the
  existing metric.
- Verify: a test that the callback emits both keys on a tiny run.
- Done when: pinned.

---

# Phase I -- integration and the run (plan §7)

## Step I1 -- `configs/agent_obs_deploy4.yaml`

Depends on G4 and H2.

**DONE 2026-09-21.** `configs/agent_obs_deploy4.yaml` is deploy3 plus the gadget pair in `self`, a
`history` group with H2's six `hist.*` fields, and `enemy_hist1`, `enemy_hist2` and `enemy_hist3`
after deploy3's ten grid planes. `obs_select.load_agent_spec` loads it with six groups, which is
`obs_select._MAX_GROUPS`, so the next field joins an existing group. The widths the policy sees:
`self` 26, `enemies` (9, 9), `projectiles` (12, 6), `zone` 5, `history` 78, `grid` (13, 13, 21). A
probe built `BrawlFeaturesExtractor` on it with no code change: CNN `in_channels` 13 and MLP
`in_features` 262, the sum of 26, 81, 72, 5 and 78, with 384 features out; deploy3 gives 10 and 182.
Deviations and choices:

1. **The plan's widths count fields.** I1.1's "(22 floats)" is 22 fields: `pos_norm`, `vel`,
   `facing_vec` and `dash_dir` are two floats each, so `self` is 26 floats (deploy3: 20 fields, 24
   floats). The plan table's `enemies` 9 x 7 and `zone` 2 count fields too, and its `projectiles` 5
   per slot matches neither the group's 4 fields nor its 6 floats. The load test pins the float
   widths above.
2. **The gadget pair sits right after the super pair**, at columns 22 and 23, not at the end, so the
   three ability pairs read together: long dash, super, gadget. `meta.time_frac` and
   `meta.n_enemies_alive` move to 24 and 25, which costs nothing because deploy4 trains from
   scratch.
3. **`history` is the fifth group, before `grid`**, in the plan table's order; I1.1's "sixth group"
   counts groups. The load test pins the order, and nothing else does: the history-before-zone
   mutant survived every other test.
4. **The header says "a sixth file"**, not the plan's "fifth": agent_obs, lowinfo, deploy, deploy2
   and deploy3 come first, and deploy3's own header already says fifth.
5. **Deployment refuses deploy4 today, and the two tests that glob `configs/agent_obs_deploy*.yaml`
   run it as a strict xfail.** The glob picked deploy4 up the day it landed, and both cases raised
   `KeyError('enemy_hist1')` from `ObservationAssembler._channel_indices` at construction;
   `GridSpec.load` refuses the file too, with `ValueError: unknown grid channel 'enemy_hist1'`.
   `tests/test_deployment_zone.py`'s round trip (`ASSEMBLER_NOT_BUILT_YET`) waits only on H4.3,
   because it builds the assembler and calls nothing but `_put_zone`.
   `tests/test_deployment_loop.py`'s decision test (`NOT_SERVED_YET`) waits on G5.1 and H4.1 to
   H4.3. Each is `xfail(strict=True, raises=KeyError)`, so the marker turns red the day its case
   passes and any other failure stays a failure. The zone file's estimator test passes for deploy4
   as it stands. G5.1 and H4.3 carry the hand-off.
6. **`docs/AGENT_OBS.md` is byte-identical after regenerating.** `scripts/dump_obs_schema.py`
   renders only `configs/agent_obs.yaml` (`AGENT_OBS_YAML`); the `--spec` flag is I3.1's.

The header, in deploy3's style: why a sixth file (runs name their spec by path), what changed, with
H2's column map and the empty-slot rule, the normalization under the plan's "Keep it" (one tile of
`hist.displacement` reads 0.017 on the 60-tile map), the six-group cap, the deployment supplier of
every new field (G5.1's proprioceptive shadow per plan S18, H4.2 from H4.1's deque, H4.3 with
`observation._history_drawn`), the measured refusal, and the width changes as a train-from-scratch.
Tests: `tests/test_configs_files.py` +8. Deploy4 joins `DEPLOY_SPECS`, which runs it through the six
per-spec rules (the lowinfo type masking, the live-loop fields, absolute hp, no unnormalized large
field, train.yaml's cube-pickup reward through `check_reward_is_observable`, no cube counts), and
the pickups table says it sees pickups. `test_deploy4_agent_obs_yaml_loads_as_a_real_agent_spec`
pins `fair` and `normalize`, the six group names in order at `_MAX_GROUPS`, every shape, the history
column map against H2's, the gadget pair at columns 22 and 23, and the grid's channel indices 0 to
4, 6 and 8 to 14. `test_deploy4_differs_from_deploy3_by_exactly_the_history_and_gadget_additions`
compares both directions: `history` is the only new group and none is lost; `enemies`, `projectiles`
and `zone` are equal; `self` is deploy3's plus exactly the gadget pair, right after
`hero.super_charge_frac`; `grid` is deploy3's planes plus the three history planes at the end;
`history` is exactly H2's six fields. The two deployment files gain the two xfails and one passing
case, the zone estimator's. Sixteen mutants, all killed: the gadget pair after `meta`, its fraction
dropped, `gadget_ready` twice, `hero.hp_frac` back, a cube count in `self`, `history` before `zone`,
`history` without `hist.valid`, `hist.hp` and `hist.ammo_frac` swapped, `enemies` without `in_bush`,
`enemy_hist3` dropped, `enemy_hist1` and `enemy_hist2` swapped, the pickup plane dropped, `fair:
false`, the pickups table without deploy4, the zone xfail on `ValueError`, and the loop xfail over
deploy3 too (a strict XPASS). Full `-m "not vision"` suite: 2289 passed, 4 skipped, 2 xfailed
(deploy4's two), 85 deselected, in 20 min 19 s.

### I1.1 The spec
- Files: `configs/agent_obs_deploy4.yaml` (new)
- Do: copy deploy3; `self` += `hero.gadget_ready`, `hero.gadget_charge_frac` (22 floats);
  new sixth group `history` with the six `hist.*` fields (78 floats); `grid` += `enemy_hist1`,
  `enemy_hist2`, `enemy_hist3` (13 channels). Header comment in deploy3's style: why a new file,
  what changed, the deployment supplier of every new field (G5.1, H4.2, H4.3), train-from-scratch.
- Verify: `obs_select` loads it with six groups.
- Done when: loads.

### I1.2 Tests and docs
- Files: `tests/test_configs_files.py`, `docs/AGENT_OBS.md`
- Do: `DEPLOY_SPECS` += deploy4; the pickups table; a
  `test_deploy4_differs_from_deploy3_by_exactly_the_history_and_gadget_additions` pin in both
  directions like every pair in that file; regenerate the docs.
- Verify: the file.
- Done when: green.

## Step I2 -- `configs/train.yaml`

Depends on B4, M4.

**DONE 2026-09-21.** `run.agent_obs` is `configs/agent_obs_deploy4.yaml`, so a bare
`python scripts/train.py` trains the deployable spec. The comment above it said the reverse, that
full information is the default and a deploy spec is picked with `--set`; it now lists the other
specs by `--set`. The tiers, maps, holdout pair and rewards were already final (the hand-off
below). Deviations and choices:

1. **`gadgets_used` is an eval metric, not an env event.** Nothing in the sim state counts
   throws, so `TierEvaluator.evaluate` counts the ones it sends, by the sim's own rule: attack
   value 3 where the attack mask allows it, at most once per decision, only in a slot's first
   episode, and counted before the step so that a slot's last decision still belongs to it. The
   mask is the one the policy was given. The sim re-derives its own after ticking the timers,
   and the two differ only on the tick before a charge completes, where a `maskable_ppo` policy
   cannot press, so for the run's algo the count is exact. It is the fifth of
   `evaluation.METRICS`: progress.csv gains `eval/gadgets_used_<tier>`, throws per episode on the
   training maps, and each tier's writer draws `eval/gadgets_used` on the overlaid chart, as it
   does `eval/win_rate`.
2. **The eval cost is measured** and replaces train.yaml's three "not re-measured" notes. RTX
   5070 Ti, 6 tiers x 300, untrained policy: 49.3 s per training-map rollout and 49.4 s per
   holdout one, each ~200 decisions at ~245 ms. A rollout stops at 600 decisions, so one eval is
   ~100 s now and up to ~5 min once the policy survives to the time cap, which is ~2.5 h to
   ~7.5 h over the run's ~92 evals. Peak allocated VRAM is 1.14GB for 4096 training envs and the
   model, and 1.24GB with both eval envs; the ~2.95GB the file quoted was 4 tiers x 1000.
3. **SB3 transposed deploy4's grid for any view under 13 cells a side**, fixed outside I2's file.
   SB3 takes a rank-3 uint8 Box on [0, 255] for an image, calls it channels-last unless the first
   axis is the smallest, and then wraps the env in `VecTransposeImage`. Deploy4's 13 x 13 x 21
   grid is channels-first only on that tie, so the real run was safe by luck, but debug_tiny's
   13 x 10 x 14 trained on 14 x 13 x 10, view columns for channels, `train.py --smoke` included.
   `builder.no_image_transpose` puts the env behind SB3's own opt-out,
   `VecTransposeImage(skip=True)`, and `build_model` and both of `scripts/train.py`'s `_resume`
   hand-offs go through it: the load, and the VecNormalize `set_env`. `scripts/sb3_smoke.py`
   still hands SB3 the bare env. Its `agent_obs.yaml` grid has 10 channels, which SB3 keeps at
   both configs, so only deploy4 on debug_tiny would trip it there.

Tests: `tests/test_training.py` +5, `tests/test_configs_files.py` +1.
`test_evaluator_counts_the_gadgets_a_real_rollout_throws` runs the real env: an idle policy
throws 0 per episode and one that presses on its first decision throws 1, where a count keyed to
the super's column would read 0 for both, since the super starts uncharged.
`test_evaluator_counts_a_gadget_only_where_legal_and_only_in_the_first_episode` scripts four
slots' legality and dones, and `test_eval_callback_means_and_gap_are_per_evaluator` checks the
CSV column and the overlaid chart by value.
`test_the_model_sees_the_grid_channels_first_as_the_env_builds_it` and
`test_a_resumed_model_keeps_the_grid_channels_first` build and resume a debug_tiny run, and
`test_the_shipped_run_trains_deploy4_at_its_pinned_input_widths` builds the extractor through
train.yaml's own env config and overrides: 13 channels, 262 floats. Eleven mutants, all killed:
the gadget count without the first-episode gate, without the mask, on the super's mask column,
on the super's action value, and after the step's dones; `gadgets_used` dropped from `METRICS`;
the overlay dropped; `build_model` and each `_resume` hand-off given the bare env; and train.yaml
back on deploy3. Full `-m "not vision"` suite after wave 6's code, H3, I2 and G6.1's probe:
2369 passed, 4 skipped, no xfails, 85 deselected, in 19 min 14 s.

Hand-off from wave 3: the tiers block (B4), `run.env_overrides.world.maps` (fourteen) and
`eval.holdout_maps` with their comments (M4) are already in the file and final, and so is the
reward block's `attack_in_reach: 0.05` (R2); I2 only switches `run.agent_obs` and adds the optional
event. Leave the eval-cost comments alone except to replace
"not re-measured" with a real GPU timing.

### I2.1 Edits
- Files: `configs/train.yaml`
- Do: `run.agent_obs: configs/agent_obs_deploy4.yaml`; tiers from B4.2; `eval.holdout_maps` from
  M4.1; rewards unchanged beyond R2's `attack_in_reach`; optionally a `gadgets_used` per-episode
  event for the eval log.
- Verify: `tests/test_training.py`; `check_reward_is_observable` passes for deploy4.
- Done when: green.
- Amended by I1 (2026-09-21): the second check already runs.
  `test_the_shipped_cube_pickup_reward_builds_only_with_a_spec_that_sees_cubes` in
  `tests/test_configs_files.py` puts the real train.yaml reward through `check_reward_is_observable`
  with deploy4, which the pickups table marks as seeing pickups.

## Step I3 -- docs, schema, graph -- DONE 2026-09-22

### I3.1 Regenerate and record
- Files: `docs/OBSERVATION.md`, `docs/AGENT_OBS.md`, `BRAWL_DEPLOYMENT_DESIGN.md` §9/§10,
  `CHARACTER_DETAILS.md`
- Do: `.venv/Scripts/python.exe scripts/dump_obs_schema.py` (extend it with `--spec` if it only
  handles `agent_obs.yaml`); `graphify update .`; append the dated decisions to the design doc; the
  gadget block in CHARACTER_DETAILS if G1.3 did not already add it.
- Also (carried from wave 3's reviews, no other owner): `brawl_sim/maps/README.md` "What actually
  trains" must say default.yaml lists the sim's sixteen while training draws the FOURTEEN in
  train.yaml's `run.env_overrides.world.maps`, that `split_river` + `hollow_ring` are eval-only
  holdouts, and that a new map needs a train.yaml entry too; `TRAINING.md` (~:72-75, :106-131):
  rewrite the ladder paragraph from B4's table (hp/damage capped at 1.5x, the two new axes, no eval
  continuity before 2026-09-18), list `eval/holdout_win_rate_<tier>`, `eval/holdout_win_rate`,
  `eval/holdout_gap`, say `eval/*` means the fourteen training maps and best_model is chosen on
  them, and mention the `.preset.yaml` sidecar in the `--save ... --no-view` workflow.
  (`docs/OBSERVATION.md` matched `dump_obs_schema.render()` exactly on 2026-09-21; only G4's and
  H2's new rows are missing.)
- **DONE 2026-09-22.**
- **`dump_obs_schema.py` grew `--spec`.** It rendered `configs/agent_obs.yaml` and nothing else, so
  rendering the spec that actually trains would have overwritten the full-information doc. `--spec`
  names the output after the spec it was given, so `agent_obs_deploy4.yaml` produces
  `docs/AGENT_OBS_DEPLOY4.md`. `docs/OBSERVATION.md` is rendered from `OBS_SCHEMA` itself and is
  rewritten either way, so its content does not depend on the flag.
- **"The docs match the schema" was checked by REGENERATING, not by reading.** All three docs
  re-render byte-identical on 2026-09-22, and `graphify update .` ran the same day.
- Tests: `tests/test_obs_schema.py` +2, `test_render_is_deterministic` and
  `test_render_covers_every_schema_field`. The second is the load-bearing one: a field added to
  `OBS_SCHEMA` and forgotten in the renderer now fails here instead of shipping a short doc.
- **`CHARACTER_DETAILS.md` was a correction, not an addition.** Gadgets sat under "Still
  unimplemented" next to Star Powers and Hypercharges. Phase G implemented Mortis's, so gadgets
  moved OUT of that list, with `gadget_cooldown: 0` named as the "this kind has no gadget"
  sentinel -- the same shape as `super_charge_hits: 0` in the bullet directly below it.
- `brawl_sim/maps/README.md`: "What actually trains" now states the sixteen-versus-FOURTEEN split,
  names `split_river` and `hollow_ring` as eval-only holdouts, and says a new map needs a
  `train.yaml` entry as well as a sim entry.
- **`TRAINING.md` needed no change, which is a finding rather than an omission.** Every item on the
  carried-forward list was already written by B4 and M4 on 2026-09-18 and is in the last commit,
  not the working tree: the ladder paragraph and its nine-column table (:73-88), the "no tier's
  eval column is comparable with any run before 2026-09-18" warning (:99), the three holdout
  metrics plus the `eval/*` = fourteen-training-maps statement (:157-163), best_model chosen on
  the training maps only (:152-155), and the `.preset.yaml` sidecar (:194). Checked line by line.
- **Done when: SATISFIED.**

## Step I4 -- test suite and throughput

### I4.1 Full suite
- Do: `.venv/Scripts/python.exe -m pytest tests/ -x -q -n auto` (~13 min). Fix what fails.
- Also (G3's review): `scripts/smoke_test.py` (`_random_action`, :66) draws the attack column from
  {0, 1}, so the all-presets smoke battery never throws a gadget or a super. Draw
  `randint(0, env.cfg.action_nvec[1])`, as `scripts/record_rollout.py` does, and run it.
- Done when: green.
- DONE 2026-09-21. `_random_action` draws `randint(0, env.cfg.action_nvec[1])` with
  `record_rollout`'s comment, and `tests/test_smoke_test.py` pins both columns against literals
  (+1 test; 4 mutants, all killed). `scripts/smoke_test.py` passed all four config variants
  (default, debug_tiny, no_zone, single_archetype; 500 decisions each at 8 envs on CPU with
  `debug_checks` on). The full `-m "not vision"` suite ran after wave 6's code (I2's note:
  2369 passed, no xfails); only the smoke script and its new test changed since.

### I4.2 Benchmark (removed)
- Removed 2026-09-21 by the operator, who watches `time/fps` during training instead, so there
  is no before/after number and no 3% budget. SB3's `time/fps` is decisions over wall clock
  since `learn()` began, eval pauses included. The last run (`mortis_deploy3_elite`, 4096 envs)
  logged 8,560 at its midpoint; its ~90 evals cost ~50 s each, 8% of its 15.5 h. This config
  adds the holdout rollout to every eval, which alone puts the same sim at ~7,900 mid-run, and
  the first row reads low because the start-of-run eval sits inside it. Lower than ~7,900 is
  the new sim work or deploy4's larger observation. The suspects, from the plan's Step I4: the
  per-entity march in `hero.gadget_target` every tick (G2), two (N,E) gathers per tick (B2,
  B3), the history push (H1: store enemies as int16 tile coordinates) and three grid scatters
  per observation (H2).
- Outcome 2026-09-22: the run logged ~6,900 against this note's ~7,900, with an iteration
  median of 63.0 s against the last run's 56.0 s. So the doubled eval was ~8% of it and the
  new sim work plus deploy4 about 12% per decision. The operator accepted it; I5 has the rest.
- `scripts/benchmark.py` (:101, :107) still draws attack from {0, 1}, so no spinner is ever in
  flight. Widen it as I4.1 widened `smoke_test.py` before using it again.

## Step I5 -- the run checklist

- [x] A1.5 audit numbers recorded (baseline before the change; A2.4 is deferred, so there is no
      deploy-side baseline). A1's DONE note (2026-09-18) holds them.
- [x] Train: DONE 2026-09-22, `runs/mortis_deploy4-20260921-185945`, 450M decisions in 18.15 h
      at ~6,900 fps (I4.2's note). Eval on the training maps 0.530 mean over the six tiers,
      best 0.549, elite 0.240; holdout 0.368 with a 0.162 gap; `gadgets_used` 1.6 on elite to
      1.9 on easy, so the policy does press the button. Every curriculum stage force-advanced
      on its timestep budget at a 0.24 to 0.32 win rate and never through the 0.35 gate, which
      is B4's harder ladder showing up exactly where this checklist predicted it.
- [x] Sim cadence audit (`--run`) on the new `best_model.zip`: DONE 2026-09-22,
      `runs/audit/cadence_mortis_deploy4.md`, 200 episodes against elite, 677 fights. R2's
      `attack_in_reach` bought both numbers it was taken for: utilization 0.511 against A1's
      0.339, long-dash waiting 0.042 against 0.114. Phasing loss rose to 0.107 from 0.081,
      which is the 0.35 s cooldown against 0.25 s decisions biting more often now that the
      policy attacks more; that is A3's structural cap, still deferred. The interval mode is
      still 10 ticks and tighter, 383 of 871 intervals against A1's 141 of 454. Ammo at the
      first attack 2.37 against 2.41.

- [x] Repoint `configs/deployment.yaml` at the new run: DONE 2026-09-22. It still named
      `runs/mortis_deploy3_elite-20260913-015933`, which is pre-gadget AND had been deleted, so a
      live run would have died on a missing path before reaching anything interesting. The
      pre-flight check that exists to catch this -- `test_deployment_policy.py -k real_checkpoint`
      -- SKIPPED instead of failing, because "this machine does not have the artifact" and "the
      config names a run that does not exist" were one branch. They are separate now, and only
      `DEPLOYED_RUN` is held to the strict one; the other parameters are historical constants and
      a machine without them is just a machine. The deploy4 checkpoint now loads and decides from
      a real assembled observation offline, which is as far as the suite can take it.

After training is complete (operator, 2026-09-21). G6 landed 2026-09-22, so both items are
live:

- [ ] `control.backend: null` dry run against the new spec: no gate exit at the first gadget
      decision (G6), `history` and `self` assembled without `_require` errors (H4, G5), telemetry
      rows carry A2.1's fields.
- [ ] Deploy `best_model.zip`; rerun the cadence audit on the resulting telemetry and compare with
      the baseline.
