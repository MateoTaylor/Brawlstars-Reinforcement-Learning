# Enemy memory and privileged critic: plan

**Status (2026-10-13): BUILT, TRAINED, and DEPLOYED.** You chose both changes on 2026-10-11. Everything below was built and passed C on 2026-10-12, and you trained it as `runs/mortis_ppo-20261012-082348`. `configs/deployment.yaml` has served it since 10-13. The results are in E.

**All five decisions settled on 2026-10-12.**
- Memory lasts 3 s (12 decisions).
- Every run gets per-track gates, with no flag.
- The critic gets no bot-skill rows.
- One run carries both changes.
- The run uses the narrow arm's widths.

On the same day you also made two changes:
- The run is shortened to 400M decisions, and the curriculum is scaled to match (D).
- Compatibility work for old models and deploy5 runs is ruled out. Where keeping them working would add code, the simpler option wins.

Two other things were settled the same day. The learning-rate test in MODEL_SIZE_STUDY.md §9 is declined: your own learning-rate experiments showed no large effect on speed or final quality. No maps are added for now.

Two changes for the next fresh run:

- **A. Enemy memory.** An enemy that drops out of view keeps showing where it was last seen, with its last HP and how long ago that was. Today its row goes blank the moment it is unseen.
- **B. Privileged critic.** The value head reads the true state of all nine bots, on top of everything the policy sees. The policy never reads that state, and live play never runs the value head.

Both land in one new spec, `configs/agent_obs_deploy6.yaml`. Like every new spec, it means a from-scratch run.

**Size.** Roughly 100 lines of code for A and 150 for B, plus about 300 lines of tests. Most of the work is the parity and guard tests.

## Decisions

All five were settled on 2026-10-12.

| # | Decision | Options | Status |
|---|---|---|---|
| 1 | How long a lost enemy stays in memory: `slots.max_misses`, in 0.25 s decisions | 3 (today), 6, 8, 12 | **12 (3 s).** In 3 s a walking enemy moves at most 8.2 tiles. |
| 2 | Per-track tracker gates (A8) | every run, deploy5 included / deploy6 runs only (a tracker flag) | **Every run, with no flag** ("pick whatever option minimizes code complexity"). deploy5 changes only while one enemy is coasting and another is visible: the visible one stops borrowing the coasting one's wider gate. |
| 3 | Bot skill in the critic (B2) | yes / no | **No.** You allowed it only if it was already in the observation, and it is not (B2). |
| 4 | Run shape | one run with A and B / separate runs | **One run.** Two runs would each need 60–70 % of the GPU. Explained variance isolates the critic, and an eval with the memory rows blanked shows whether the policy uses its memory (E). |
| 5 | Widths | control / narrow | **Narrow.** At 500M it still leads the control by about .05 at matched steps, and it has matched the control's best since 260M (MODEL_SIZE_STUDY.md §10). |

## The picture

```
TRAINING (sim)
  full obs ─┬─> actor groups (6, fair) ──> CNN + MLP ──> actor features ─┬─> pi head ──> action
            └─> critic groups (2, true state) ──────────────────────────┴─> vf head ──> V(o, s)

LIVE (BlueStacks)
  assembler ──> actor groups ──> CNN + MLP ──> actor features ──> pi head ──> action
                critic groups: zeros, never written, and nothing on the pi side reads them
```

## Part A: enemy memory

**What changes for the agent.** deploy5 already keeps a lost enemy's slot for 3 decisions. But `fair: true` multiplies the whole row by `revealed_to_hero`, so the row reads as an empty slot from the moment the enemy is unseen. Live, the assembler already writes the coasting track's last position, velocity, HP and bush flag, and the gate throws them away. deploy6 keeps that row and adds its age:

| column | seen this decision | coasting (unseen, still slotted) |
|---|---|---|
| `tracked` | 1 | 1 |
| `seen` | 1 | 0 |
| `rel_pos`, `dist` | now | last-seen position minus the hero's position now |
| `rel_vel` | now | last-seen velocity minus the hero's velocity now |
| `hp` | now | last-seen HP |
| `in_bush` | now | bush at the last-seen position |
| `age_s` | 0 | unseen decisions × 0.25 s |

An empty slot reads all zeros, as today.

**A1. Sim state.** In `brawl_sim/core/state.py`, add three float32 fields to `_SLOT_FIELDS`: `ent_seen_pos` (N,E,2), `ent_seen_vel` (N,E,2) and `ent_seen_hp` (N,E). They reset with the other slot fields.

**A2. Snapshot.** In `brawl_sim/core/slots.update`, once `seen` is computed, copy `ent_pos`, `ent_vel` and `ent_hp` into the three fields wherever `seen` is true. It is a `torch.where`, so there is still no host sync. The age needs no new state: `ent_misses` already counts a slotted enemy's unseen decisions.

**A3. Full observation.** `brawl_sim/core/observation.py build_obs` gets a new top-level `obs["memory"]`, one row per entity like `entities`, plus 8 rows in `obs_schema._ROWS`. Every value comes from the snapshot. That works for seen enemies too, because `slots.update` runs just before `build_obs` in the same decision, so a seen enemy's snapshot is its current state.

- `memory.tracked` (bool): `ent_slot > 0`.
- `memory.seen` (bool): tracked and `ent_misses == 0`. This is the tracker's `seen_now`.
- `memory.rel_pos` (tiles): `ent_seen_pos` minus the hero's position. `memory.dist` (tiles) is its length.
- `memory.rel_vel` (tiles/s): `ent_seen_vel` minus the hero's velocity.
- `memory.hp` (hp): `ent_seen_hp`.
- `memory.in_bush` (bool): the bush lookup at `ent_seen_pos`, using a variant of `_in_bush` that takes positions.
- `memory.age_s` (seconds): `ent_misses × cfg.agent_dt`.

*Why a new top-level key rather than more `entities.*` fields.* obs_select picks a group's fairness mask from the first part of the field name. So any `entities.*` group gets multiplied by `revealed_to_hero`, which is exactly what blanks a lost enemy. `memory.*` gets its own mask, `memory.tracked`.

It is fair by construction. Every value is something the hero saw, and a row is non-zero only while the tracker's slot rule holds the slot.

**A4. obs_select.** In `brawl_sim/core/obs_select.py`:
- `_HERO_AXIS_PREFIX` becomes `_HERO_AXIS_PREFIXES = ("entities", "memory")`. Both drop the hero's row 0, and both may use `slots: tracked`.
- Add `_FAIRNESS_MASK_FIELD["memory"] = "memory.tracked"`.
- `load_agent_spec` refuses a `memory.*` group without `slots: tracked`, because the live side cannot reproduce entity order.

The existing rule that a group's fields share one prefix already keeps `memory.*` and `entities.*` out of the same group.

**A5. Spec.** Add a new file, `configs/agent_obs_deploy6.yaml`. It is deploy5 verbatim except for the enemies group, and Part B adds a `critic:` section to the same file:

```yaml
  - name: enemies
    per_entity: true
    slots: tracked
    dtype: float32
    fields: [memory.tracked, memory.seen, memory.rel_pos, memory.dist, memory.rel_vel,
             memory.hp, memory.in_bush, memory.age_s]
```

The group is 9 rows × 10 columns, against deploy5's 9 columns, and the spec still has 6 groups.

**A6. Memory length.** In `configs/train.yaml`, set `run.env_overrides: {slots: {max_misses: 12}}` (decision 1). This reaches training, eval and the live tracker, which `loop.py` builds from the run's `slots.*`. `memory.age_s` then runs from 0 to 3.0 s.

**A7. Live assembler.** `brawl_deployment/perception/assemble.py _put_entities` writes `full["memory"]` from values it already computes:
- `tracked` is "this row was written";
- `seen` is `track.seen_now`;
- `age_s` is `track.misses × cfg.agent_dt`.

No new reader is needed:
- `track.pos` already is the last-seen position while a track coasts.
- `loop._update_enemy_hp` already holds the last committed HP.
- `enemy_in_bush` is already looked up at `track.pos`.

`full["entities"]` stays, for deploy5.

**A8. Live tracker and loop.** Three changes in `brawl_deployment/perception/tracker.py` and one in `loop.py`:

- **Per-track gates (decision 2).** Today `gate_for` sizes one gate for all tracks, from the largest gap among them. At M = 12, one enemy lost for 3 s would give every fresh track a gate of about 9 tiles.
  - Fix: `_update_enemies` hands `_greedy_match` one gate per track, `GATE_NOISE_TILES + MAX_WALK_TILES_S × (t − tr.t)`.
  - A plain float still works, for `loot.py` and `projectiles.py`.
- **Capped extrapolation.** `Track.predict` uses `min(t − self.t, 1.0 s)`.
  - At nominal timing, 1.0 s is the longest gap deploy5 can match across (3 misses plus one decision), so deploy5's predictions are unchanged.
  - Past that, a constant-velocity guess walks out of the area the enemy could have reached, and the growing gate covers the rest anyway.
- **The hero's coast stays at 0.75 s.** The tracker uses one `max_misses` for the hero and the enemies, and `loop.py` takes `tracked.hero.pos` as the hero's position on every decision.
  - At M = 12, a missed player box would leave the hero on a stale position for 3 s.
  - Fix (built): `EntityTracker.hero_max_misses`, a class attribute fixed at 3. The hero coasts it whatever `max_misses` is, so no caller passes it.
- **Auto-aim keeps the 0.75 s coast too.** `DeployLoop._auto_aim` counts coasting tracks on purpose: the sim aims at enemies out of view, and a track lost a few ticks ago is the best stand-in for one. A 3 s ghost is not; the enemy may have moved 8 tiles.
  - Fix (built): `_auto_aim` and the `enemy_in_reach` telemetry column skip tracks with more than `AIM_MAX_MISSES` (3) misses, so both behave exactly as today at any M.
  - The enemy-HP matcher needs no change. A reading lands on the track that took its detection, at a distance of about zero. A coasting track can take a reading only within 0.5 tiles of where it was last seen, as today.

**A9. Telemetry.** Two `TickRow` columns, appended last:
- `n_ghosts` counts the slots coasting this decision, −1 on a tick with no decision.
- `reacquired` lists each coasting track matched again this tick as `gap:jump` pairs joined by `;`: the unseen decisions it coasted, and the tiles between its last sighting and the detection that took it.

Together they show how often the identity errors below happen. `scripts/probes/replay_clip.py` also prints every coast, built from the track ids (C3).

**A10. Tests.**
- `tests/test_slots.py`:
  - snapshots move only when the enemy is seen;
  - a coasting enemy keeps its last position, velocity and HP;
  - the age runs 0, 0.25, 0.5 s and so on, and the slot frees after M.
- `tests/test_obs_select.py`:
  - in a memory group a coasting row survives and an untracked row is zero;
  - an `entities.*` tracked group still zeroes the coasting row, so deploy5 is unchanged;
  - a memory group without `slots: tracked` is refused.
- `tests/test_deployment_assemble.py`: the tracked-slot parity scenes (`_tracked_spec`) gain a coasting enemy and a retired one. The live memory rows must equal the sim's.
- `tests/test_deployment_tracker.py`:
  - a lost enemy no longer widens a fresh track's gate;
  - `predict` is capped;
  - at M = 12 the hero still retires after 3 misses.
- `tests/test_deployment_loop.py`: at M = 12, a track with 4 or more misses is neither an auto-aim target nor counted in `enemy_in_reach`.
- `tests/test_configs_files.py`: pin deploy6 against deploy5 in both directions, as deploy5 is pinned against deploy4. Every new field's units either have a divisor or are already of order 1.
- `scripts/dump_obs_schema.py` regenerates `docs/AGENT_OBS*.md`.

**Known divergence, accepted and bounded by M.** The sim's slot identity is perfect; the tracker's is not. A9 measures how often each case happens.
1. An enemy that reappears outside its ghost's gate, after a dash or Edgar's jump, gets a new slot live while its ghost lingers for M decisions. The sim keeps one slot.
2. Live, a different enemy appearing inside a ghost's gate can take that ghost's slot.
3. A segment reset (death, respawn or an odometry cut) clears every ghost live. The sim never does.
4. The ghost of an enemy that died unseen lingers for M decisions in both the sim and live. `meta.n_enemies_alive` in `self` drops at once, so the agent can learn to discount it.

If C3 shows case 1 often, the fix is a second matching pass that lets an unmatched ghost take a fresh detection within dash reach. That needs its own decision and is not in this build. C3 found case 1 rare: one clear instance in 25 coasts that ran out (C3 result, below).

## Part B: privileged critic

**Feasible, with no SB3 internals.** It needs our own features extractor plus one documented hook, `_build_mlp_extractor`, the method SB3's custom-policy docs override. Nothing in site-packages changes.

The value head estimates V(o, s): everything the policy sees, plus the true sim state. It never takes the state alone: the policy acts on o, and a state-only baseline is biased under partial observability. `DeployedPolicy` calls `predict`, which runs only the actor, so nothing live has to supply the critic's inputs. They are zero-filled live, and a test proves the actor cannot read them.

**B1. Spec.** Add a `critic:` section to `configs/agent_obs_deploy6.yaml`, in the same group syntax as `groups:`:

```yaml
critic:   # value head only: the policy never reads these, and live play zero-fills them
  - name: critic_enemies        # all 9 bots, true values, entity order, dead rows zero
    per_entity: true
    dtype: float32
    fields: [entities.alive, entities.kind_onehot, entities.rel_pos, entities.dist,
             entities.rel_vel, entities.hp, entities.max_hp, entities.cubes,
             entities.ammo_frac, entities.can_attack, entities.dashing, entities.in_bush,
             entities.revealed_to_hero, entities.hero_revealed_to,
             entities.privileged.targets_hero, entities.privileged.react_t]
  - name: critic_world
    per_entity: false
    dtype: float32
    fields: [hero.cubes, zone.hero_margin, zone.safe_area_frac]
```

`critic_enemies` is 9 × 25 = 225 floats. `critic_world` is 6 floats; its `zone.hero_margin` is the unclamped one.

No bot-skill rows (decision 3). The critic still gets a partial tier signal for free: the curriculum scales each bot's base HP and walk speed at reset, and `entities.max_hp` and `entities.rel_vel` carry both.

The critic lives in the spec file on purpose. Every tool that loads a run's spec picks the critic groups up with no call-site change: training, eval, `watch.py`, the probes and `DeployedPolicy.from_run`. So does the vec env's `terminal_observation`, which needs V for the truncation bootstrap.

**B2. One new full-obs row.** In `observation.py`, inside the existing `obs_include_privileged` block, with a matching privileged row in `obs_schema`: `entities.privileged.targets_hero` (bool), `state.ent_target == 0`, meaning the bot is hunting the hero.

It is a bool view of `target_id`, which the block already holds. The raw index (−1 for none, 0 for the hero, 1–9 for other bots) is a poor network input, since only "is it 0" matters here.

The block today holds `target_id`, `react_t`, `reveal_t`, `decision_phase` and `move_intent`, and no skill parameters. So decision 3 adds nothing.

**B3. obs_select.**
- Add `GroupSpec.critic: bool = False` and `AgentObsSpec.critic_groups: tuple = ()`. The default leaves every existing spec and caller as is.
- `load_agent_spec` parses `critic:` with the actor's group parser, except that:
  - `entities.privileged.*` is allowed;
  - groups are float32 only;
  - `slots`, `max_slots` and `view_channels` are refused;
  - names must start with `critic_` and be unique across both lists;
  - there are at most 2 critic groups, since each is one more host copy per step. The actor keeps its own 6-group cap.
- Critic groups skip the fair gate. A per-entity critic row is multiplied by `entities.alive` instead, so dead bots read zero.
- `agent_space`, `make_agent_obs_buffers` and `agent_obs_index_map` cover both tuples.
- `build_agent_obs(..., critic=True)` builds both. The live assembler passes `critic=False`, so its critic buffers are never written and stay zero.

**B4. Features extractor.** `brawl_sim/wrappers/sb3_features.py BrawlFeaturesExtractor` gets a new kwarg, `critic_keys=()`:
- critic keys stay out of the actor MLP;
- `forward` returns `[cnn, mlp, flattened critic inputs]`;
- the extractor exposes `actor_dim` (the cnn width plus the mlp width).

With no critic keys it builds exactly today's modules, so every old checkpoint loads unchanged. `builder.build_policy_kwargs` passes `critic_keys` only when the spec has critic groups.

**B5. Policy.** A new `brawl_sim/wrappers/sb3_policy.py`, about 50 lines:

```python
class SplitMlpExtractor(nn.Module):
    """policy_net reads features[:, :actor_dim]; value_net reads all of them."""
    # latent_dim_pi, latent_dim_vf, forward, forward_actor, forward_critic: SB3's documented interface

class PrivilegedCriticPolicy(MaskableMultiInputActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SplitMlpExtractor(self.features_extractor.actor_dim, self.features_dim,
                                               self.net_arch, self.activation_fn)
```

- **The pi head is today's `net_arch.pi`:** the same widths, Tanh and orthogonal init.
- **The vf head is today's `net_arch.vf` with a wider first layer.** The critic inputs reach it already normalized by the spec, so no new width key is needed.
- **The trunk stays shared.** Value-loss gradients still reach the CNN and the actor MLP, as today.

**B6. Builder.** `brawl_sim/training/builder.py build_model` passes `PrivilegedCriticPolicy` instead of `"MultiInputPolicy"` when the spec has critic groups. It refuses critic groups under `run.algo: ppo`.

`MaskablePPO.load` restores the class from the zip by its import path, so that path becomes a contract. The module docstring should say so.

**B7. Deploy.** In `brawl_deployment/policy.py from_run`, after `check_spaces`, assert two things:
- `model.policy.features_extractor.critic_keys` equals the spec's critic group names;
- if there are any, the policy's `mlp_extractor` is a `SplitMlpExtractor`.

This is the one place a zero-filled live input could reach an actor, so it fails closed. `assemble` already returns every buffer, so `act` needs nothing else.

**B8. Guard tests** in `tests/test_privileged_critic.py`, on CPU with a tiny cfg:
1. Perturbing every critic input leaves every action logit `torch.equal`, and changes the value.
2. `DeployedPolicy.act` gives the same action and the same probabilities whether the critic inputs are zero or the sim's true values.
3. A save, then a `MaskablePPO.load`, gives back the same class, logits and values.
4. The loader rules:
   - privileged fields are still refused in actor groups, and the existing pins in `test_integration.py` and `test_obs_select.py` stay untouched;
   - they are allowed under `critic:`;
   - a hidden bot reads zero in `enemies` and its true values in `critic_enemies`.
5. A rollout and an update run, and both heads move.
6. `DeployedPolicy.from_run` refuses a deploy6 checkpoint that lacks the split, built either with the stock `MlpExtractor` or with no critic keys.

The planned deploy5-load test was dropped with backwards compatibility on 2026-10-12.

**Cost.**
- **Compute:** about zero. The critic inputs are extra columns on one Linear layer.
- **Host RAM:** the rollout buffer grows by about 0.5 GB at 4096 envs × 128 steps (231 floats a sample). At 8192 envs it grows by 1.0 GB, and RAM there was already tight with BlueStacks or CapCut open.
- **Transfers:** two more small device-to-host copies per step. Read their cost off `time/fps` in the run's first minutes; no benchmark is needed.

## Found while building: two slot updates per step

With autoreset on, `env.step` builds the observation twice:
- once in `_observe`;
- again in `_autoreset`, for every env. The sync-free design rebuilds them all, not only the ones it reset.

Both builds ran `slots.update`, and each call counts a sighting or a miss. So every step moved the slot table two decisions:
- An enemy took its slot on its first sighting, not its second.
- A slot coasted half its length: 1 unseen decision at `max_misses` 3, against the tracker's 3, and 6 at 12.
- `memory.age_s` would have read double.

**Fixed without a key, as a parity fix.**
- `_autoreset`'s build now moves only the envs it reset (`slots.update(..., envs=done)`).
- `tests/test_slots.py::test_one_env_step_is_one_decision_of_the_slot_table` steps the env and counts one miss per step.

**Old models are left as they are.** Every deploy5 run trained with the double count, including the served `runs/mortis_ppo-20261007-161117`.
- For those models, a live row appears one decision later than it did in training.
- The short coast matters little to them, because deploy5's fair mask blanks a coasting row anyway.

## C. Verification before you train

All of this runs on CPU, so the narrow run keeps the GPU. The one exception: C1's suite only half-hid the GPU, so its few small CUDA-guarded tests ran on it (C1 result).

- **C1.** The new and touched tests, then the full suite with `-m "not vision"` (about 35–45 min).
- **C2.** A CPU training smoke on deploy6 into `runs/smoke` (learn, checkpoint, reload), then `tests/test_deployment_policy.py` against it.
- **C3.** A tracker replay with `scripts/probes/replay_clip.py --device cpu` over the five BlueStacks recordings. It runs at M = 12 through a new `--max-misses` flag and reports A9's ghost counts and re-acquisitions, so we see how often the known divergence happens on real footage.

**C1 result (2026-10-12).** The full suite, `-m "not vision"`, collected 2920 tests, 44 more than on 2026-10-07: 2917 passed, 1 skipped and 2 failed, in 39 min. Both failures are understood:
- `test_observation.py::test_privileged_and_world_grid_toggles` pinned the privileged block's key set, and B2 adds `targets_hero`. The pin now names it, and the file passes.
- `test_sb3_integration.py::test_run_smoke_completes_all_8_steps_on_cuda` is a false failure, caused by how the suite hid the GPU. On Windows an empty `CUDA_VISIBLE_DEVICES` hides nothing from the CUDA runtime, but torch's `device_count()` reads it as 0. So the test trained on the GPU and then failed to load its own save. It skips under `CUDA_VISIBLE_DEVICES=-1`, its CPU twin passes, and it can run on the GPU once the GPU is free.
- `scripts/dump_obs_schema.py` was last edited after the suite had collected, so its two test files were rerun on the current file: 69 passed.

**C2 result (2026-10-12).** `scripts/train.py --smoke` on CPU ran 8192 steps in 2.6 min, through all five stages, with eval and both checkpoints, into `runs/smoke`. That replaced the 2026-10-11 deploy5 smoke. The saved model is a `PrivilegedCriticPolicy`. Its pi head reads 272 inputs and its vf head 328, the extra 56 being the critic of the tiny world, which has 2 enemies. `tests/test_deployment_policy.py` and `tests/test_privileged_critic.py` then gave 29 passed and 1 skipped; the skip is an old run folder with no `best_model.zip`. The real-checkpoint test loaded and drove `runs/smoke` and all six local deploy5 runs through the new code, the served one included.

**C3 result (2026-10-12).** The served model (`runs/mortis_ppo-20261007-161117`) replayed all five recordings at `--max-misses 12`, one process per clip, and every one exited 0. That also shows the served deploy5 checkpoint loads and decides through the new live code. The footage is fixed and the tracker runs on frame time, so these counts are the tracker's, whatever the model decides.

| clip | match | decisions | with a ghost | most ghosts | coasts | reacquired | coasted out | open at the end |
|---|---|---|---|---|---|---|---|---|
| bottom-left | 11.9–39.5 s | 103 | 45 | 2 | 9 | 5 | 4 | 0 |
| bottom-right | 8.3–58.0 s | 184 | 82 | 4 | 23 | 11 | 9 | 3 |
| top-right | 5.3–42.2 s | 143 | 55 | 3 | 15 | 11 | 4 | 0 |
| holdout-left-center-bottom | 9.8–35.9 s | 90 | 28 | 2 | 6 | 4 | 2 | 0 |
| holdout-left-center-top | 9.0–46.7 s | 137 | 89 | 3 | 20 | 13 | 6 | 1 |
| **all five** | | **657** | **299 (46 %)** | **4** | **73** | **44** | **25** | **4** |

- **The memory is in use.** Nearly half of all decisions hold at least one remembered enemy, and up to 4 at once.
- **What M = 12 adds.** 14 of the 44 reacquisitions (32 %) came after 4–12 unseen decisions. At M = 3 each of those tracks would have been retired, and its enemy re-slotted as a new track two sightings later.
- **The jumps stay small.** The median jump is 0.81 tiles after 1–3 unseen decisions and 0.90 after 4–12. The four largest are 5.59, 4.59, 4.23 and 3.78 tiles. Every jump is within what a walking enemy covers in the time, plus the tracker's 1-tile noise floor.
- **Case 1 is rare.** Of the 25 coasts that ran out, one had a new track promoted close to its last sighting: bottom-right track 11, 3.1 tiles away. Four more had one 7–11 tiles away, about as far as a walking enemy covers in 3 s, so possibly the same enemy and possibly not. C3 does not show case 1 often, so the second matching pass is not needed.
- **Case 2 can't be measured here.** A reacquisition is the tracker's own call, and nothing in an unlabelled replay says whether it took the same enemy. The jumps above are consistent with the same enemy, but they don't prove it.

## D. The run (yours)

These are built into `configs/train.yaml` as of 2026-10-12, so a bare `python scripts/train.py` trains this run:
- `run.agent_obs: configs/agent_obs_deploy6.yaml`.
- `run.env_overrides.slots.max_misses: 12`.
- The narrow widths: `cnn_output_dim` 104, `mlp_hidden_dims` [168, 168], `net_arch` pi and vf [256, 256].
- `run.total_timesteps` 400M, and `curriculum.max_timesteps_at_stage` 83M. That is 125M × 2/3, so the run reaches elite by 332M at the latest.

Everything else is the control's. The LR and clip schedules run over `total_timesteps`, so they decay 1.5× faster per step than in the 600M arms. `tests/test_configs_files.py` and `tests/test_training.py` pin all of it.

## E. Readout

- **The critic: `train/explained_variance`.**
  - The baseline: the control read .43 at 25M and .47 at 50M. Every arm sits at .48–.49 from about 100M on, at every width from 0.54× to 3.2×.
  - If it reads clearly higher (about .55 or more by 50M), hidden information was what capped the old critic.
  - If it reads about .49, what remains is luck. The critic costs almost nothing to keep, and a later spec can drop it.
  - EV is scored against λ-returns that contain the critic's own bootstrap, so compare it only at equal gamma and λ. Every run here shares both.
  - The curriculum moves faster in this run, so read EV against the narrow arm at a matched stage, not only at a matched step.
- **The policy:** the eval/* six-tier and holdout means against the arm with the same widths, at matched steps. At narrow widths that is `mortis_narrow`: .304 at 160–200M, .344 at 460–500M. Win rate measures A and B together.
  - **Schedule confound.** `mortis_narrow` ran the 600M schedule: LR and clip decayed over 600M, with a 125M stage cap. This run decays over 400M and force-advances at 83M.
  - Its slot table also moves once per step, where every deploy5 run's moved twice.
  - So at a matched step it is further along both schedules, and its slot timing differs. The gap measures A and B plus those.
  - The fairest single number is each schedule's end: this run's last evals against the narrow arm's best, .344 at 460–500M.
- **Memory use** (optional, after training): evaluate the final model twice, once as trained and once with every coasting row zeroed, so each remembered enemy reads as an empty slot. The win-rate drop is how much the policy relies on its memory. It costs one eval, not a training run.
- **Live:** A9's ghost counts on the first BlueStacks matches.

### Results (2026-10-13)

You trained the run as `runs/mortis_ppo-20261012-082348`. It ran 400M decisions and ended in the elite stage. The gates advanced it at 36M, 54M and 82M, and the 83M cap forced elite at 165M.

- **The critic:** explained variance read .65 at 25M, .675 at 50M and .70 at 400M. The control read .43 and .47, and every earlier arm sat at .48–.49. It rose immediately and stayed above .65 at every logged step, so stage timing can't explain it. Hidden information was what capped the old critic.
- **The policy:** the 400M eval is the run's best, and `configs/deployment.yaml` serves it since 10-13. Its six-tier mean is .424 and its holdout mean .391. Against the end of each schedule:

  | | six-tier | easy | medium | hard | veteran | expert | elite | holdout |
  |---|---|---|---|---|---|---|---|---|
  | this run, 400M | .424 | .87 | .66 | .53 | .24 | .18 | .06 | .391 |
  | served 10-07 run, 580M | .333 | .763 | .527 | .347 | .177 | .140 | .043 | .273 |

  `mortis_narrow`'s best was .351, and it averaged .344 over 460–500M. The gain of about .08 is well outside the ±.03 bar. It is the sum of A, B and the confounds listed above, so it doesn't tell you which change earned it. The optional memory-use eval would split out A.
- **Live:** not yet played.

## Out of scope

- **The learning-rate test.** Declined 2026-10-11; your own experiments settle it.
- **More maps, procedural ones included.** Deferred 2026-10-11.
- **Recurrent PPO.** The memory slots are the cheap alternative we discussed instead.
