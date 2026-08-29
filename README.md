# brawl-sim

A batched, GPU-resident approximation of Brawl Stars Solo Showdown, built as a native PyTorch
`BrawlVecEnv` plus a Stable-Baselines3 `VecEnv` adapter. See `BRAWL_SIM_BUILD_PLAN.md` for the
full design and build plan (including Appendix A's known divergences from the real game), and
`CONVENTIONS.md` for the coding rules every module follows.

GIT IS JUST SOMETHING I AM USING TO SAVE MY CODE, FOLLOW LOCAL VERSION AS THE MAIN SOURCE OF TRUTH AND STOP COMPARING
ACROSS THE TREE DURING EDITS

## Setup

```
pip install -e .[dev,sb3,render]
```
### Install trap: Blackwell (sm_120) GPUs

If you're on an RTX 50-series card (Blackwell, compute capability `sm_120`), a plain
`pip install torch` can pull a build whose kernels stop at `sm_90`, which fails at *runtime*
(not at install time) with:

```
no kernel image is available for execution on the device
```

Install from the CUDA wheel index instead:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

(or a newer `cu12x`/`cu13x` channel — check https://pytorch.org/get-started/locally for the
current one). Windows-native is fine; no WSL required. This sim is tensor-op bound, not matmul
bound.

**The definitive check is `torch.cuda.get_arch_list()` containing `'sm_120'`**, not
`torch.cuda.is_available()` — the latter returns `True` even on a build that cannot actually
launch a kernel on the device.

Run `python scripts/check_install.py` after installing to verify.

## Quickstart

```python
from brawl_sim.config import load_config
from brawl_sim.core.obs_select import load_agent_spec
from brawl_sim.env import BrawlVecEnv
from brawl_sim.wrappers.sb3_vecenv import BrawlSB3VecEnv
from brawl_sim.wrappers.sb3_features import default_policy_kwargs
from stable_baselines3.common.vec_env import VecMonitor
from sb3_contrib import MaskablePPO

cfg  = load_config("configs/default.yaml", overrides={"entities": {"n_enemies": 6}})
spec = load_agent_spec("configs/agent_obs.yaml", cfg)

sim  = BrawlVecEnv(cfg, n_envs=256, device="cuda", seed=0)
venv = VecMonitor(BrawlSB3VecEnv(sim, spec, reward_fn=MyReward()))

model = MaskablePPO("MultiInputPolicy", venv,
                    policy_kwargs=default_policy_kwargs(spec, cfg), verbose=1)
model.learn(total_timesteps=1_000_000)

# --- or, native path, zero-copy, for a GPU-side learner ---
obs = sim.reset()
obs, reward, terminated, truncated, info = sim.step(action)   # reward is zeros by design
```

`load_config`'s `overrides` is a **nested dict matching the YAML's own section structure**
(`{"entities": {"n_enemies": 6}}`, not a flat dotted key like `{"entities.n_enemies": 6}`) — it's
deep-merged onto the base file section by section. Every script in `scripts/` and every test in
`tests/` uses this nested form; see [Config schema](#config-schema-and-scalar-or-range-syntax)
below.

`MyReward` is a placeholder — write your own `__call__(self, obs, info, cfg) -> torch.Tensor`
(see `brawl_sim/core/reward.py`'s `Reward` protocol and `ExampleReward` for a working example).
`BrawlVecEnv`'s own default reward is `ZeroReward` (all zeros) — the sim ships no opinion on
what "good" means; that's the one piece explicitly left for you to write.

## Two paths: native vs. SB3

| | native `BrawlVecEnv` | `BrawlSB3VecEnv` |
|---|---|---|
| Observation | full-information, device-resident torch tensors | agent's `AgentObsSpec`-selected fair view, numpy on host |
| `info` | dense torch tensors, every env, every tick | Python `dict` per env (SB3's own contract) |
| Host transfers | none — zero-copy the whole way | required every step (SB3's contract, not this sim's choice) |
| Scales to | `n_envs` in the thousands-to-tens-of-thousands | `n_envs` in the hundreds-to-low-thousands before the host round trip dominates (`scripts/benchmark.py` finds the actual knee on your hardware) |
| Use it for | a custom/GPU-native training loop (CleanRL-style PPO operating directly on the torch tensors), or anything that wants max throughput | Stable-Baselines3's algorithms (`PPO`, `sb3_contrib.MaskablePPO`) and ecosystem (`VecMonitor`, checkpointing, etc.) |

Both wrap the same underlying `BrawlVecEnv` — `BrawlSB3VecEnv` never re-simulates anything, it
just adapts the native env's tensors to the shape SB3's `VecEnv` ABC requires. The native API
stays fully supported regardless of which path you pick; nothing about the SB3 wrapper's
existence removes it as an option later. `wrappers/gym_single.py`'s `BrawlGymEnv` is a third,
narrower option: a single-env (`n_envs=1`) `gymnasium.Env` for `stable_baselines3.common.
env_checker.check_env` and manual/interactive play (`scripts/play_manual.py`).

## Config schema and scalar-or-range syntax

Two YAML files load into two different places, and they follow **different rules**:

- **`configs/default.yaml`** — sections (`world`, `view`, `sim`, `action`, `observation`,
  `entities`, `limits`, `cubes`, `boxes`, `zone`, `regen`, `bots`, `perception`, `engine`) map
  onto `EnvConfig`'s static fields (map size, episode length, feature toggles — anything that
  shapes tensor allocation or branches Python control flow). These are **plain scalars only**;
  `EnvConfig` is resolved once at construction and never randomized per env.
- **`configs/brawlers.yaml`** (per-brawler-kind stats) plus the handful of numeric fields under
  `entities`/`cubes`/`boxes`/`zone`/`perception`/`regen` in `default.yaml` feed `SimParams`
  instead — the numbers that CAN be randomized per env. **Every one of these fields accepts
  either a plain scalar (same value for every env) or a `{low: X, high: Y}` range** (sampled
  independently per env, resampled on every reset):
  ```yaml
  base_hp: 3800                        # same for every env
  base_hp: {low: 3200, high: 4400}     # uniform per-env, resampled on reset
  ```
  `mode: multiplicative` (default `additive`) scales the env's own base value instead of
  replacing it outright — see `configs/randomization.yaml`'s own comments for both forms side
  by side.
- **`configs/randomization.yaml`** is a *third* file: an optional overlay, applied once at
  `BrawlVecEnv` construction, that promotes any `SimParams`-backed `"section.field"` dotted key
  to a range WITHOUT editing `default.yaml`/`brawlers.yaml` directly. It ships with every line
  commented out (a template, not an active default) — uncomment what you want, or pass your own
  `RandomizationSpec` dict straight to `BrawlVecEnv(..., randomization=...)`.
- **`configs/presets/*.yaml`** (`debug_tiny`, `no_zone`, `single_archetype`) are override
  fragments meant to be layered onto `default.yaml`, the same way `configs/agent_obs.yaml`'s
  own `fair`/`normalize`/`groups` structure is a fourth, unrelated schema (agent-observation
  field selection, not simulation config at all — see below).

## Action space and masking

`action` is `(N, 2)` int64: `action[:, 0]` is the move bin (`0` = idle, `1..n_move_bins` = a
direction, evenly spaced by `core.geometry.dir_from_bin`/`bin_from_dir`); `action[:, 1]` is
raw fire intent (`0`/`1`). `cfg.action_nvec` gives `(n_move_bins + 1, 2)`, the shape a
`gymnasium.spaces.MultiDiscrete` expects.

Illegal fire (out of ammo, on cooldown, mid-dash, or dead) is a **silent no-op**, never an
error — `core.hero.decode_action` re-gates it before it ever reaches the sim, so you never need
to pre-filter actions yourself. The same gate is exposed directly as `core.hero.action_mask(state,
params, cfg)` (`{"move": (N,17) bool, "attack": (N,2) bool}` — move is always all-legal today;
attack's `[1]` column is `alive & ammo>=1 & cooldown<=0 & dash_t<=0`) for anything that wants to
mask actions proactively, e.g. `sb3_contrib.MaskablePPO` via `BrawlSB3VecEnv.action_masks()`.

## Decision rate vs tick rate

The world simulates at `1/sim.dt` (20 Hz) but the agent decides at `1/(sim.dt *
sim.action_repeat)` — 4 Hz at the defaults. One `BrawlVecEnv.step()` is one **decision**, and it
runs `sim.action_repeat` sim ticks. **Movement is held** across the window (the hero keeps
walking or dashing on the same heading); **fire is applied on the first sub-tick only**, so one
decision is at most one attack attempt. That makes "fire" mean the same thing at
`action_repeat=5` as at 1, and lines the sim up exactly with `hero.action_mask`, which
MaskablePPO evaluates once per decision. It is also a structural bound on any weapon that fires
faster than the decision rate — see the `attack_cooldown` note below.

The unit split is the thing to keep straight:

| In **sim ticks** | In **decisions** |
| --- | --- |
| `sim.max_episode_steps`, `state.step_count`, `state.time` | `cfg.max_agent_steps`, SB3 timesteps, `ppo.n_steps` |
| `zone.step_seconds`, `decision_period_ticks`, cooldowns, regen | `info["episode"]["l"]`, `run.total_timesteps`, `eval.every_timesteps` |
| every `ShapedReward` weight | `ppo.gamma` |

Per-tick event deltas are summed across the window; the hero's outcome fields (`terminated`,
`truncated`, `hero_rank`, `hero_alive`) are **latched at the sub-tick that env's episode actually
ended**, so a hero that becomes last-alive partway through a window cannot be killed by the
window's leftover ticks and reported as a death. See `core/events.advance_decision_tally` for the
one thing latching deliberately does not fix — a finished env's `final_observation` can be up to
`action_repeat - 1` ticks stale. Set `sim.action_repeat: 1` for per-tick decisions; behavior is
then identical to before the knob was wired up.

Because both `build_obs` calls, the host transfers, the action-mask loop, and the policy forward
all happen once per *decision* rather than once per tick, raising `action_repeat` cuts wall-clock
per simulated second substantially — that, plus representing a longer horizon in fewer
transitions, is the reason to turn it up.

Replays are **not** sampled at the decision rate. `BrawlVecEnv.tick_hook` fires after every
sub-tick, and `scripts/watch.py` / `scripts/record_rollout.py` capture frames through it, so a
recording is always at the sim's own 20 Hz and looks identical at `action_repeat=1` and `5`.
The hook is debug tooling only — it is a Python callback in the hot loop and the useful things to
do inside it are host syncs; nothing in the training path installs one.

## Tick order

Each `BrawlVecEnv.step()` runs 17 phases in a fixed order (`env.py`'s own `_run_tick` ×
`action_repeat`, then `_observe`/`_autoreset` once); the one thing worth internalizing before
writing a reward function: **attack resolves before movement.** A dash starts inside the attack phase (clipping its path against
walls immediately) and is excluded from that same tick's movement phase — that's what makes a
dash *replace* the walk it happens on, not stack with it (verified end to end in
`tests/test_play_manual.py`'s dash feel-check, using the real `configs/brawlers.yaml` numbers).
`scripts/benchmark.py --phase-n-envs <N>` prints a full per-phase timing breakdown if you want
the complete 17-phase list with real numbers attached.

## Bot behavior: archetype x personality

A bot's **archetype** (`Kind`: sniper / artillery / melee / rifle) decides how it *shoots* —
`combat()` in each `bots/<archetype>.py`, returning fire/aim only. Its **personality** (`Person`,
drawn independently per entity on every reset) decides how it *moves*, in `bots/personality.py`.
Both are randomized per entity, so a sniper can be a rusher and a melee bot can be a camper.

The **artillery** shell is the one weapon that isn't a straight shot, and it is modeled on Grom's:
it arcs over walls without needing line of sight, takes the same `proj_flight_seconds` (0.9) to
land whatever the distance — so backing off does not buy you reaction time, and crowding it does
not deny you any — and detonates on the *landing tile only*, then throws four half-damage shards
two tiles out along the world axes. Full damage if it lands on you, half if an arm catches you,
never both: the shards start outside the blast rim, so the two regions never overlap. Walls stop
the shards even though they didn't stop the shell. The numbers all live under `bot_artillery` in
`configs/brawlers.yaml` (`proj_flight_seconds`, `aoe_radius`, `split_distance`,
`split_damage_fraction`); the mechanic is `core/projectiles._spawn_splits`.

| personality | behavior |
| --- | --- |
| `rush` | closes on anything it can see; explores at random when it can't |
| `camper` | sits in a bush and fires only once something can actually see it. Does not leave, even when idle, until the zone closes to `bots.camper_zone_flee_tiles` |
| `hunter` | engages what it sees; otherwise sweeps the map's **bush waypoints** for what it can't. The counter to a hidden player |
| `trapper` | holds a bush and shoots freely from it, drifting bush-to-bush while nothing is visible |
| `kite` | works the open map holding its own archetype's ideal range |

Rules that hold for every personality: nobody walks into the green zone (a predictive inward push
starts `bots.zone_avoid_tiles` from the edge, before any damage); a personality that wants cover
and can't find any safe cover behaves as `rush`; and with nothing visible everyone reverts to
randomized exploration rather than a shared destination, so bots spread out and can wander into an
ambush *you* set. Bots also shoot loot boxes when they have no enemy target, so cubes are
contested.

Three settings do most of the work, and each replaced something measurably broken:

- **`bots.sight_tiles` (14).** `perception.visibility` has no range limit by design — it answers
  "is this entity concealed", and concealment comes from bushes, not distance. Feeding that
  straight into bot targeting gave every bot perfect sight of a 60x60 map: 84% of live bot-ticks
  had a target locked at a mean 22.7 tiles, ~3x the longest weapon's reach, so no bot ever had a
  reason to go looking for anyone. `bot_visibility` clips targeting only; the agent's observation
  is untouched.
- **`bots.hunt_cell_tiles` (10).** Hunters sweep one precomputed bush waypoint per map cell, not
  "the nearest unsearched bush tile". The tile version measured 75 tiles walked for 15.7 net
  displacement on a 24%-bush map — in a dense bush field there is always another unsearched tile
  one step away, so hunters shuffled around their spawn.
- **`bots.min_aggressive` (1).** Forces at least one `rush`/`hunter` per env after the weighted
  draw, at a *randomly chosen* slot, so an unlucky roll can't produce a lobby you beat by standing
  still — and so "slot 1 is the dangerous one" isn't a pattern the agent can read instead of
  learning to fight.

Measured against a hero that camps a bush and sends **nothing but idle actions** for 128 full
episodes on `bushy` — the degenerate policy a 53M-step run had actually learned. Both rows use the
same zone and regen settings, so the difference is the bot AI alone:

| hero camps... | survived the episode | killed by a bot | damage taken |
| --- | --- | --- | --- |
| a peripheral bush, before | 88.3% | 5/128 | 1,267 |
| a peripheral bush, now | **9.4%** | 45/128 | 8,079 |
| a central bush, before | 95.3% | 6/128 | 562 |
| a central bush, now | **12.5%** | 112/128 | 9,203 |

The "before" rows also finished with a mean of 1.05 players alive: the bots really did grind each
other down to exactly one survivor while the hero sat out the game. Under the shipped config (which
also starts the zone earlier and enables regen) the same test gives 7.8% and 15.6%.

`bots.personalities: false` makes every bot a `rush`, the closest thing to the pre-personality
behavior, and `bots.sight_tiles: 0` restores unlimited bot sight.

## Out-of-combat regeneration

`regen` (on by default) heals `regen.max_hp_fraction_per_second` of an entity's **own max HP** per
second, once it has been out of combat for `regen.delay_seconds`. Two things reset that stopwatch
(`ent_out_of_combat_t`): taking damage, and **attacking**. Both halves matter — with only the
first, an entity could hold a target down while out-healing the chip damage coming back, because
only the victim's timer ever reset.

The rate is a fraction, not flat HP/s, so every brawler takes the same time to top up regardless of
pool size, and collecting power cubes (which raise max HP) doesn't make healing slower. At the
shipped `0.2`, a full heal takes ~5s once the 4s delay has elapsed.

Regen runs in tick phase 2, before that tick's attack phase, so an entity that fires banks one tick
of regen (~1% of max HP) before its own shot zeroes the stopwatch. That ordering is deliberate: it
is also what stops a regen tick that hasn't seen this tick's combat from un-killing a fatal hit.

## Autoreset semantics

`BrawlVecEnv.step()` resets any env that finished **inside that same call** — the `obs` it
returns is the FIRST observation of a fresh episode for any env that finished this tick, not a
view of the tick that just ended. That finished episode's own final state lives in
`info["final_observation"]`/`info["final_info"]` instead. Getting this backwards silently
misaligns every reward/observation pair for a finished episode — `reward`/`terminated`/
`truncated`/`info` describe what just happened; `obs` describes what comes next.

`BrawlVecEnv(..., autoreset=False)` disables this (used by `wrappers/gym_single.py`, since
standard single-env `gymnasium.Env`s are contractually forbidden from resetting inside
`step()`) — with it off, `obs` IS the just-finished (possibly terminal) tick's own observation,
and you call `reset()` yourself.

## Observation package

Two generated references (`scripts/dump_obs_schema.py`, regenerate after touching
`core/obs_schema.py` or `configs/agent_obs.yaml` — do not hand-edit either doc):

- **`docs/OBSERVATION.md`** — the full, privileged-inclusive observation `BrawlVecEnv` itself
  produces (every field `core/obs_schema.py`'s `OBS_SCHEMA` declares).
- **`docs/AGENT_OBS.md`** — the actual SB3-facing space `configs/agent_obs.yaml` resolves to
  (what `BrawlSB3VecEnv`/`BrawlGymEnv` hand your policy) — column-level, ready to cross-reference
  against a trained model's input layer.

The full package **never hides anything** — visibility/bush-hiding/fairness are reported as
flags (`revealed_to_hero`, `hidden_by_bush`, ...), never used to mask, truncate, or sort data
out. Fairness gating happens in exactly one place, `core/obs_select.py`, driven by
`AgentObsSpec.fair` — see Appendix A item 11 below before changing it.

## Map CSV legend

Maps live in `brawl_sim/maps/csv/*.csv`, one character per tile, row-major (`origin="upper"`,
y increases downward):

| char | tile |
|---|---|
| `.` | floor |
| `#` | wall (blocks units and projectiles) |
| `b` | bush (hides an occupant; see Appendix A item 6) |
| `~` | water (blocks units, not projectiles) |
| `f` | fence (blocks units, not projectiles — opposite of walls) |
| `S` | spawn point |
| `X` | box spawn point |

`maps/loader.validate_map` enforces: the outer border is entirely wall, 8-32 spawn markers,
8-64 box markers, and the unit-passable region is a single connected component.

## Scripts

| script | what it's for |
|---|---|
| `scripts/check_install.py` | verify the `sm_120` install trap (above) didn't bite |
| `scripts/dump_obs_schema.py` | regenerate `docs/OBSERVATION.md`/`docs/AGENT_OBS.md` |
| `scripts/record_rollout.py` | record an `n_envs=1` rollout to `rollout.npz` (random actions) |
| `scripts/play_manual.py` | live keyboard play at 20 Hz — the human feel-check for the dash |
| `scripts/train.py` | **the PPO training loop** — see [Training](#training) below |
| `scripts/watch.py` | load a checkpoint and watch it play one match in the matplotlib viewer |
| `scripts/sb3_smoke.py` | proves the SB3 plumbing (check_env, VecMonitor, MaskablePPO, save/load) actually connects — not a real training run |
| `scripts/smoke_test.py` | every config variant, 500 random steps, `debug_checks=True` |
| `scripts/benchmark.py` | env-steps/sec, per-phase timing, memory, observation footprint — native vs. both SB3 `info_mode`s |

`brawl_sim/render/ascii.py` (`render_ascii`) and `brawl_sim/render/viewer.py` (`ReplayViewer`,
matplotlib) turn a recorded rollout — or, for the viewer, a live `BrawlVecEnv` (see
`play_manual.py`) — into something you can actually look at.

## Training

```
python scripts/train.py                      # configs/train.yaml as-is
python scripts/train.py --smoke              # ~30s CPU wiring check, proves the loop end to end
python scripts/train.py --set run.n_envs=1024 --set learning_rate.initial=1e-4
tensorboard --logdir runs
```

Everything tunable lives in **`configs/train.yaml`** — PPO hyperparameters, the learning-rate and
clip-range schedules, the shaped-reward weights, and the difficulty curriculum. `--set
dotted.key=value` patches single keys for one-off sweeps. The merged config is archived into the
run directory, so a run is reproducible from its own output.

The training layer is `brawl_sim/training/` and is deliberately separate from the simulator —
`BRAWL_SIM_BUILD_PLAN.md` puts the reward function, policy architecture, and hyperparameters
explicitly out of scope for `core/`/`bots/`/`env.py`. It touches the sim only through public
seams: `BrawlVecEnv`'s `reward_fn` and `params_hook`, and `BrawlSB3VecEnv`'s `infos`.

| module | what it owns |
|---|---|
| `training/config.py` | `TrainConfig` — the typed, eagerly validated form of `configs/train.yaml` |
| `training/schedules.py` | LR/clip schedules (`constant`/`linear`/`cosine`/`exponential`) over SB3's `progress_remaining` |
| `training/reward.py` | `ShapedReward` — device-resident, sync-free, every weight in YAML |
| `training/curriculum.py` | `CurriculumManager` — per-env difficulty tier sampling, installed as `params_hook` |
| `training/callbacks.py` | `CurriculumCallback` (stage advancement) and `TierEvalCallback` (stationary eval) |
| `training/evaluation.py` | `TierEvaluator` — fixed-difficulty, fixed-seed scoring against every tier |
| `training/builder.py` | assembles a `TrainConfig` into a ready-to-`learn()` `(model, venv)` |

### Reward

`ShapedReward` is dense by necessity: an episode runs up to 3000 sim ticks (600 decisions) and
ends in a 1-of-10 placement, so a terminal-only signal is one bit per episode. Terminal terms
(`win_bonus`, `death_penalty`, `rank_bonus`) plus per-tick terms (damage dealt/taken, kills, cube
pickups, survival, a zone-exposure penalty). Set the per-tick weights to `0.0` to fall back to
sparse once a policy is competent. Per-term contributions are logged under `reward_terms/` so you
can see which term actually dominates — the thing you need when tuning weights.

Every weight is priced **per sim tick and stays that way when `sim.action_repeat` changes**: the
delta terms read `info` fields the env already summed across the decision's sub-ticks, and the two
rate terms read `info["alive_ticks"]`/`info["in_zone_ticks"]` rather than firing once per
`step()`. Total episode return is therefore invariant to the decision rate, so retuning
`action_repeat` never silently rescales the reward. `gamma` is the one knob that does *not* come
along for free — it discounts per decision, so it needs `gamma ** action_repeat` to hold a fixed
real-time horizon.

### Curriculum

A **tier** is a set of multipliers on `configs/brawlers.yaml`'s own per-archetype numbers, so tier
`hard` (all `1.0`) is exactly the bots as authored. Tiers are drawn **per (env, archetype) on
every reset**, so within one batch you face the configured mixture and within one episode
different archetypes can sit at different difficulties. Archetype *selection* stays uniformly
random throughout — the curriculum changes how well the bots play, not which ones show up.

The shipped ladder walks `easy_ramp` (70/25/5 easy/medium/hard) → `mixed` → `hard_lean` →
`hard_only` → `elite`. The first four stages move aim and reaction quality only; `elite` sharpens
those past the reference bot *and then* raises HP and damage. Advancement is **adaptive**: the win
rate over the last `window_episodes` finished episodes must clear that stage's
`advance_win_rate`, with at least `min_episodes_at_stage` episodes played since the last
transition. Optional `demote_win_rate` drops back a stage on collapse; optional
`max_timesteps_at_stage` force-advances a stalled one.

A win means the hero was genuinely **last alive** (`info["outcome"]["rank"] == 0`), not merely
alive at the timeout — otherwise the agent could advance the curriculum by learning to hide.

Why a hook rather than the `{low, high}` range syntax: a range is a single continuous uniform, and
a curriculum needs a *weighted discrete mixture* over named tiers. The two compose — a range
jitters the base value per env, the tier then scales it.

**Stage visibility.** The current stage is reported everywhere it could be needed: a startup
banner listing every stage with the current one marked, a loud console banner on each transition,
`curriculum/stage_index` + `stage_name` + `win_rate` + configured-vs-actual tier mixture on every
logger dump, and a `curriculum.json` in the run directory holding the stage plus full transition
history (which `--resume` reads back). Overlay `curriculum/stage_index` on `rollout/ep_rew_mean`
when reading results — a shaped reward's absolute level is only comparable *within* a stage, so a
reward drop right after a stage bump is expected progress, not a regression.

### Stationary evaluation

The one signal in the run that is comparable **across time**. Everything else — `rollout/
ep_rew_mean`, `curriculum/win_rate` — is measured against the training distribution, which the
curriculum deliberately makes harder, so it cannot tell "the policy improved" apart from "the
curriculum got easier".

Every `eval.every_timesteps`, `TierEvaluator` plays `eval.episodes_per_tier` matches against
**each** difficulty tier with the curriculum's sampling switched off and the same seed every
time, so the maps, spawns, and bots are identical at 2M steps and at 20M. All tiers are scored in
**one batched rollout**: a single env of `len(tiers) * episodes_per_tier` envs is partitioned into
blocks, and `FixedTierHook` pins each block to one tier. Only the first episode per env slot
counts, so short episodes (deaths) can't over-represent themselves and bias the win rate down.

Results land in two complementary places:

- **`eval/win_rate_easy`, `eval/win_rate_hard`, …** (plus `mean_rank`, `mean_ep_length`,
  `mean_reward` per tier) on the run's own logger — separate TensorBoard charts, and columns in
  `logs/progress.csv`.
- **One overlaid `eval/win_rate` chart** with a line per difficulty, labelled `eval_easy`,
  `eval_hard`, … This works by writing a separate event file per tier under `logs/eval_<tier>/`,
  all using the same tag: TensorBoard treats each subdirectory as its own run and overlays
  matching tags. Watching the easy-vs-hard gap close is the single most informative view of a
  curriculum run, and separate scalar *names* cannot draw it on one chart.

`best_model.zip` is saved whenever the mean win rate across tiers improves — stationary eval is
the only signal "best" can honestly be defined against.

Cost is bounded: one rollout of `len(tiers) * episodes_per_tier` envs, at most
`cfg.max_agent_steps` **decisions** (`sim.max_episode_steps / sim.action_repeat`, so 600 at the
defaults — not 3000; the eval loop is bounded in the same unit it steps in). Tens of seconds on
GPU.

### Watching a checkpoint

```
python scripts/watch.py runs/mortis_ppo-.../best_model.zip                  # vs hard bots
python scripts/watch.py runs/.../final_model.zip --tier easy
python scripts/watch.py runs/.../checkpoints/model_4000000_steps.zip --tier elite --episodes 5
python scripts/watch.py runs/.../best_model.zip --ascii      # terminal, no GUI
python scripts/watch.py runs/.../best_model.zip --save match.npz --no-view
```

Plays one match with the policy driving Mortis against bots pinned to a single tier (the same
`FixedTierHook` evaluation uses, so `--tier hard` is exactly the population `eval/win_rate_hard`
scores), then opens it in `ReplayViewer` — a scrubbable recording, not a live stream, so you can
step back through the moment the hero died. Prints per-match stats (placement, kills, damage
dealt/taken, cubes, death cause, survival time) either way.

The run's own archived `train.yaml` is found automatically next to the model and used to rebuild
the environment; `--train-config` overrides it. That matters beyond convenience — `map_id` indexes
`cfg.map_names` and the observation layout must match what the policy trained on, so a mismatched
config renders the wrong map or won't feed the network at all. `watch.py` compares the model's
observation space against the rebuilt env's and fails with a pointer at the config rather than a
bare shape error from inside the policy.

### Run layout

```
runs/<name>-<timestamp>/
  train.yaml          the exact merged config this run used
  curriculum.json     current stage + transition history
  logs/progress.csv   every logged scalar, one row per dump -- plot this
  logs/log.txt        the same tables the console printed
  logs/events.*       TensorBoard, if the `tensorboard` package is installed
  logs/eval_<tier>/   per-difficulty event files -- overlaid on one eval/win_rate chart
  checkpoints/        periodic model .zip + VecNormalize .pkl
  best_model.zip      best mean stationary-eval win rate so far
  final_model.zip     + final_vecnormalize.pkl
```

### Where the logs go

Three surfaces, all written to the same `logs/` directory, all carrying the identical set of
scalars:

- **Console** — SB3's key/value table, printed every `run.log_interval` rollouts.
- **`logs/progress.csv`** — one row per dump, one column per scalar. Read it with
  `pandas.read_csv` and plot whatever you want; this is written **unconditionally**, so it works
  with no TensorBoard installed and survives the terminal closing.
- **`logs/events.*`** — TensorBoard, via `tensorboard --logdir runs` (which recurses, so it picks
  up every run at once for comparison). Needs the optional `tensorboard` package; if it's
  missing, training prints a warning and continues with CSV + console rather than failing.

SB3's own default is stdout-only (or stdout + TensorBoard when `tensorboard_log` is set), with no
machine-readable file in either case — `training/builder.make_logger` replaces it via
`model.set_logger` for exactly that reason.

Scalar groups: `rollout/` (SB3: `ep_rew_mean`, `ep_len_mean`), `train/` (SB3: losses,
`learning_rate`, `clip_range`, `approx_kl`, `explained_variance`), `time/` (`fps`,
`total_timesteps`), plus this project's own `curriculum/` (stage index and name, win rate, mean
rank, episodes at stage, configured vs. actually-sampled tier mixture) and `reward_terms/` (the
mean per-tick contribution of each `ShapedReward` weight — which term is actually driving the
policy).

## Known divergences from the real game

Summarized from `BRAWL_SIM_BUILD_PLAN.md` Appendix A (full list and rationale there): no
Supers/Gadgets/Star Powers/Hypercharges; 16-bin movement with no independent aim (the dash
inherits move direction); soft unit collision, no knockback/stuns/slows; simplified bushes;
20 Hz tick; no cube
redistribution/timed spawns; bot policies are hand-written heuristics, not real players
(randomize their difficulty or a trained agent will overfit to them); trajectories are
deterministic only for a fixed `n_envs`; **the simulator does not restrict the hero's
information by itself** — fairness is one config flag (`AgentObsSpec.fair`) away from being
switched off, which is deliberate (a one-line cheating-vs-fair experiment) but means a
misconfigured spec can silently train a cheating  . `tests/test_integration.py`'s
privileged-isolation tests are the guard; keep them green.

### Camera divergences (measured 2026-08, `Terrain_Perception_Build_Plan.md` Phase K)

The egocentric view was `20x40` — a guess that predated any measurement. Pushing the normalized
viewport's corners through the Phase C homography gives the ground quad the Solo Showdown camera
actually sees, and the largest hero-centred rectangle inside it is **21 wide x 13 tall**, now what
`configs/default.yaml` declares. The old number gave the policy ~1.7x the horizontal world it will
have at deploy time. **This changes `obs["view"]`'s shape, so every checkpoint in `runs/` predates
it and cannot be loaded against this config.**

Two divergences remain, both measured and both left in place:

- **The real camera sits ~0.80 tiles above the hero on the ground plane** (the hero is drawn below
  the centre of frame); the sim centres the hero exactly. This costs the view height rather than
  shifting it — the rectangle has to fit within the shorter of the two vertical reaches — so the
  sim is conservative here rather than wrong.
- **The real camera does not always centre the hero, and the sim always does.** While the camera
  is actively tracking, the hero is centred to **+0.09 tiles** horizontally (IQR ±0.3). But whole
  stretches of footage have the hero sitting **4 to 8 tiles right of centre**, and in every one of
  them the camera's horizontal travel is essentially zero (0.08 and 0.12 tiles across entire
  clips). Two of those clips are the training range, where the camera is fixed by design and does
  not follow the player at all. The remaining case, in a live match, is **unexplained**: the sign
  is what a camera clamped at a map boundary would produce, but a direct test — do the large
  offsets coincide with the camera sitting at an extreme of its own travel? — did not confirm it.
  Treat "clamping" as an untested guess, not a measurement.

  The sim's `_view_origin` is unconditionally `hero_ix - view_w // 2`. This does not affect the
  view dimensions above, which are derived from the tracking case, and it errs toward *less*
  information rather than more. The Phase I occupancy map is in world coordinates, so a deployed
  policy can always be handed a hero-centred crop regardless of where the camera actually sits.

`Terrain_Perception_Build_Plan.md` Phase F, note 2 asked whether the real camera *leads* the
player. It does not — it lags, by a fraction of a tile (offset correlates positively with camera
velocity, sd 0.28 tiles). That is camera smoothing, not a divergence worth modelling.
