# Brawl Stars Solo Showdown — Vectorized PyTorch Sim Build Plan

**v6 — final. All decisions resolved. Stable-Baselines3 target added. Viewer switched to
matplotlib. Terrain roster and vision model corrected to match the real game.**

A step-by-step implementation plan for a batched, GPU-resident approximation of Brawl Stars
Solo Showdown, with Mortis as the learner-controlled hero.

**Deliverable:** a `BrawlVecEnv` (native torch, full information) plus a
**Stable-Baselines3 `VecEnv` adapter** that feeds the agent a configurable fair-information
observation while routing ground-truth and privileged data to `infos` for logging and
reward computation.

**Explicitly out of scope:** the reward function itself, policy architecture, training runs,
hyperparameters. The plan builds the plumbing so `model.learn()` runs; what it learns is yours.

---

# 0. Status: no open decisions

Everything is locked. Two implementation notices you should read once, then proceed.

### Notice 1 — 1 ms latency rounds to zero at 20 Hz
You chose 1 ms. One tick is 50 ms, so `round(0.001 / 0.05) = 0` ticks and the latency ring
buffer is a pass-through. The machinery is still built and tested. I changed the config key
from `action_latency_ticks` (int) to **`action_latency_seconds` (float)**, converted to ticks
at load, so it engages automatically if you ever raise the tick rate or the latency. No action
needed — this is a note, not a question.

### Notice 2 — SB3 will be your throughput ceiling, not the sim
This is worth knowing before you tune anything. Stable-Baselines3's `VecEnv` contract requires,
every single step:
- observations as **numpy arrays on the host** (a device→host copy),
- `infos` as a **Python list of `num_envs` dicts** (Python object allocation per env per step),
- rollouts buffered in numpy, then copied back to GPU for the update.

So the GPU→CPU→GPU round trip is unavoidable while SB3 is in the loop. Practical consequences:

- Run SB3 at **`n_envs` in the 128–1024 range**. The native torch API scales past 4096, but
  SB3's boundary cost grows linearly with `n_envs` and will dominate well before that.
- Step 41 benchmarks both paths separately so you can see exactly where the cliff is on your
  hardware rather than guessing.
- The native `BrawlVecEnv` API stays fully intact and zero-copy. If SB3 becomes the
  bottleneck, a GPU-native PPO (CleanRL-style, operating directly on the torch tensors) drops
  in without touching the simulator. **Do not delete or bypass the native API to simplify the
  SB3 path** — it's your escape hatch.

### Notice 3 — Interactive viewer (Step 38) is matplotlib, not pygame
Discovered during Step 1 on this machine: `pygame` ships no Windows wheel past `cp313`, this
project's only available interpreter is Python 3.14 (`cp314`), and the source build falls back
to an MSYS2/pacman toolchain that isn't present here. Rather than pin the whole project to an
older Python just for one debug tool, `render/viewer.py` (Step 38) is built on `matplotlib`
instead — same controls, same acceptance bar, using `FuncAnimation`/`blit=True` and
`key_press_event` to hit 20 fps. `[render]` is now `matplotlib` only; `pygame` is not a
dependency anywhere in the plan. `render/ascii.py` (Step 37) was never affected — no action
needed there.

### Notice 4 — Terrain roster and vision model corrected (discovered during Step 2)
Two corrections to the original Step 2 draft, made against the real game rather than a guess:

1. **`Tile.ROPE` is gone.** It was a stray leftover with no real-game basis and no map ever
   referenced it. The roster is now `FLOOR, WALL, BUSH, WATER, FENCE, SPAWN, BOX` —
   **`N_TILES = 7`**, not 8.
2. **`Tile.FENCE` behavior flipped, and vision is no longer tile-based at all.** Fence blocks
   *movement* but not *shots* (opposite of the original draft). And separately: the game is
   played from a fixed bird's-eye camera, so **no terrain — wall, fence, or water — ever blocks
   sight**. The only thing that hides an entity is standing in a bush, per D12
   (`bush_reveal_radius` + the post-attack reveal timer), and that rule is exactly the same for
   bot targeting as it is for the hero's `revealed_*` observation flags — bots get no
   wall-piercing awareness a human on the same camera wouldn't have.

   This means `TILE_BLOCKS_VISION` is gone as a distinct tensor: with fence no longer blocking
   shots, the only tile that's ever opaque to a straight line of travel is `WALL` — the same
   set `TILE_BLOCKS_PROJ` already describes. Wherever a module needs a genuine **physical**
   line-of-sight check (melee hit validation in Step 14, the sniper/rifle fire-gate in Steps
   17/20, the `obs["visibility"]["los"]` annotation field), it reuses `TILE_BLOCKS_PROJ` via
   `core/terrain.line_of_sight()` — that check answers "is there a wall in the way," which is a
   different question from "can this bot target/see that entity" (bush-only, computed in
   `bots/perception.visibility()`, Step 15, with no terrain lookup at all).

   Concretely: `bots/perception.py` (Step 15), `core/terrain.py`'s Step 8 acceptance text,
   `core/combat.py`'s melee LOS comment (Step 14), the artillery archetype's "wants a wall
   between" behavior (Step 18), and the OOM mitigation note (Step 41) are all updated in place
   below to match. `MapBank` (Step 6) drops its `blocks_vision` field.

---

# 1. Locked decisions

| ID | Resolution |
|----|-----------|
| D01 | Two grids. Map `map_h x map_w` = **60x60**, configurable. Agent view **landscape 20 tall x 40 wide**, configurable, egocentric crop centered on the hero. |
| D02 | All 4 archetypes implemented. `n_enemies` is a parameter (default 6). |
| D03 | **PyTorch**, native Windows. Step 1 covers the sm_120 / Blackwell install trap. |
| D04 | Action = `MultiDiscrete([17, 2])`. Move: 0 = idle, 1..16 = 16 direction bins. Attack: 0/1. Attack masked during dash and when `ammo < 1`; mask exposed via `action_masks()`. Reload rate configurable. |
| D05 / D06 | All stats in `configs/brawlers.yaml`. Archetype behavior matters, exact numbers don't. |
| D07 | `dt = 0.05` (20 Hz), `max_episode_steps = 3000`, both configurable. |
| D08 / N12 | **Dict observation, nested and named, full information.** All entities, all projectiles, all boxes, all pickups — always present, index-stable, never truncated, never masked. Visibility is annotation. |
| D09 / N05 | **The simulator returns zero reward.** Reward is computed at the SB3 adapter boundary from `obs` + `info`. |
| D10 | Zone: rect, starts at `0.2 * episode length`, shrinks `tiles_per_step` every `step_seconds`, escalating DPS. |
| D12 | Bush reveal radius 2.0 tiles; also revealed 1.0 s after attacking. Drives bot perception and the `revealed_*` flags. Never removes data from the observation. |
| D13 | Soft separation (one pushout iteration). Dashers pass through everything. |
| D14 | Bots break boxes, collect cubes, flee the zone, fight each other. All toggleable. |
| D15 | 4 maps: `blank` (debug), `open`, `bushy`, `walled`. |
| D16 | Terminate on hero death, hero-last-alive, or timeout. No respawn. |
| D17 | Full ammo system for hero and all bots. |
| D18 / N04 | Dash stops at walls and water, damages each enemy once per dash, passes through units. **i-frames block combat damage only — zone damage still applies.** |
| D19 | Per-archetype difficulty: `aim_noise_std_rad`, `reaction_delay`, `lead_target_fraction`, `decision_period_ticks`. |
| D20 | `action_repeat: 1`, no frame stacking in the sim. |
| N02 / N03 / R02 | **Dash aims along the move direction chosen that same tick and fully replaces the walk.** Idle move input → dash along `facing`. No steering during the dash; the path is clipped at dash start. |
| N06 | Eager-first, compile-ready. `torch.compile` behind a flag (Step 31). |
| N07 | One device `Generator`. Reproducible for fixed `(seed, n_envs, device, actions)`. |
| N08 | **Every tunable accepts a scalar or a `{low, high}` range. Ranges resample per env on every reset.** |
| N09 | `step()` accepts an optional `override` tensor to drive any entity externally. |
| N10 / R01 | `action_latency_seconds: 0.001` → 0 ticks at 20 Hz. Buffer built and tested. |
| N11 | Zone kills drop cubes. |
| R03 | Full 60x60 `world` grid included in the observation, `uint8`, toggleable via `obs.include_world_grid`. |
| R04 | Bot internals namespaced under `obs["entities"]["privileged"]`, never routed to the agent. |
| **NEW** | **SB3 integration.** `BrawlSB3VecEnv` implements `stable_baselines3.common.vec_env.VecEnv`. A declarative `AgentObsSpec` selects the agent's fair-information observation; everything else goes to `infos` for logging and reward. |

---

# 2. Global conventions

Step 1 writes this section verbatim to `CONVENTIONS.md`. **Every delegated task reads it first.**

## Stack
Python 3.11+, PyTorch 2.x, `numpy`, `PyYAML`, `pytest`.
Extras: `[sb3] stable-baselines3, sb3-contrib, gymnasium`, `[render] matplotlib`.
**No JAX, no numba, no custom CUDA.** Plain PyTorch tensor ops.

## The one rule that matters
**Everything is a batched tensor with a leading `(N,)` environment dimension, and there is
never a Python loop over environments.** If you write `for i in range(n_envs)`, the design is
wrong — **except** inside `wrappers/sb3_vecenv.py`, where SB3's list-of-dicts `infos` contract
makes it unavoidable. That file is the single sanctioned exception and must say so at the top.

Python loops are otherwise permitted only over compile-time constants: the 4 bot archetypes,
the fixed raycast step count, the fixed volley size, the fixed channel list.

## No host synchronization in the hot path
Forbidden inside `BrawlVecEnv.step()` and anything it calls:
`.item()`, `.cpu()`, `.numpy()`, `.tolist()`, `bool(t)`, `int(t)`, `float(t)`,
`if some_tensor:`, `assert some_tensor`, `print(some_tensor)`, `torch.nonzero()`,
`t[bool_mask]` (data-dependent shape → forced sync).

Use `torch.where`, masked arithmetic, `masked_fill_`, `scatter_`,
`index_put_(accumulate=True)`, `clamp`, `topk`.

**Host transfers are legal in exactly one place: `wrappers/sb3_vecenv.py`.** Nowhere else.

Debug assertions go behind `if self.cfg.debug_checks:` where `debug_checks` is a Python bool,
off by default.

## Mutation model
Systems **mutate state in place** on preallocated buffers, and declare it:
```python
def apply_movement(state: SimState, move_dir: Tensor, ...) -> None:
    """MUTATES: state.ent_pos, state.ent_vel, state.ent_facing."""
```
Allocate nothing per step that can be preallocated.

## Dtypes and units
- Reals: `float32`. Indices: `int64`. Counters/enums: `int32`. Masks: `bool`.
  Observation occupancy grids: `uint8`.
- **Distance in tiles** (1 tile = 1.0). **Time in seconds.** **Angles in radians**,
  `atan2(dy, dx)`, range `(-pi, pi]`.
- **Coordinates:** `pos = (x, y)`, x = column, y = row. Tile `(col=i, row=j)` covers
  `[i, i+1) x [j, j+1)`. **y increases downward**, matching CSV row order.

## Shapes
`N` envs, `E` entities (`1 + n_enemies`), `P` projectile slots, `B` boxes, `U` pickups,
`M` maps, `H`/`W` map grid, `VH`/`VW` view grid, `K` = `N_KINDS` (5), `C` channels.

**Entity index 0 is always the hero. Entity slot order is stable for the whole episode.**
Never sort or compact entity slots.

## Devices and RNG
The env owns `self.device` and `self.gen = torch.Generator(device=device)`. Every random call
passes `generator=self.gen`. Never touch the global RNG.

## Testing
Every step ships `tests/test_<module>.py`, running on CPU by default with `@pytest.mark.gpu`
variants where relevant. Use `n_envs = 8` and `presets/debug_tiny.yaml`.

## Config
Nothing is hardcoded. Every number reaches code through `EnvConfig` (static Python values),
`SimParams` (per-env tensors), or `AgentObsSpec` (declarative agent observation).

---

# 3. Repository layout

```
brawl_sim/
├── pyproject.toml   CONVENTIONS.md   README.md
├── docs/
│   ├── OBSERVATION.md          # generated, the full obs contract
│   └── AGENT_OBS.md            # generated, the SB3-facing obs contract
├── configs/
│   ├── default.yaml  brawlers.yaml  randomization.yaml  agent_obs.yaml
│   └── presets/{debug_tiny,no_zone,single_archetype}.yaml
├── brawl_sim/
│   ├── constants.py  config.py
│   ├── core/
│   │   ├── state.py geometry.py terrain.py stats.py
│   │   ├── hero.py movement.py projectiles.py combat.py
│   │   ├── boxes.py zone.py spawn.py
│   │   ├── observation.py obs_schema.py obs_select.py events.py reward.py
│   ├── maps/{loader.py, csv/*.csv}
│   ├── bots/{perception,steering,policy,sniper,artillery,melee,rifle}.py
│   ├── env.py
│   ├── wrappers/
│   │   ├── episode_stats.py
│   │   ├── gym_single.py        # gymnasium.Env, n_envs=1, for check_env
│   │   ├── sb3_vecenv.py        # stable_baselines3 VecEnv  <-- the training entry point
│   │   └── sb3_features.py      # minimal features extractor so SB3 accepts the Dict obs
│   └── render/{ascii,viewer}.py
├── scripts/
│   ├── check_install.py smoke_test.py benchmark.py
│   ├── play_manual.py record_rollout.py
│   ├── dump_obs_schema.py sb3_smoke.py
└── tests/
```

---

# 4. Canonical tick order

`BrawlVecEnv.step()` executes exactly these phases, in order, once per `dt`.
**Attack resolution comes BEFORE movement** — that's what makes the dash replace the walk.

```
 1. action latency buffer      pop the action from L ticks ago, push the new one
 2. advance timers             ammo/reload, attack_cd, dash_t, invuln_t, reveal_t, react_t, no_damage_t
 3. decode hero action         move bin -> direction; attack bool ANDed with the action mask
 4. bot policies               all 4 archetypes computed batched, selected by kind
 5. external overrides         optional, drives any entity index
 6. ATTACK RESOLUTION          consume ammo; START DASH (clip path now); spawn volleys; melee hitscan
 7. movement                   ONLY non-dashing entities. intent -> terrain slide -> soft separation
 8. dash advance               dashers move the clipped path; capsule sweep; once-per-victim damage
 9. projectile update          integrate, wall collide, unit collide, artillery detonate, AoE
10. zone damage                outside the safe rect; i-frames do NOT block this
11. box damage                 destroyed boxes -> spawn pickups
12. pickup collection
13. death resolution           hp <= 0 -> dead, record rank + cause, drop cubes
14. zone shrink schedule
15. bookkeeping                time += dt, step_count += 1, n_alive, cumulative counters
16. perception + observation   visibility matrix (annotation only), full obs package, done, info
17. autoreset                  reset done envs in place; stash final obs/info
```

Deaths resolve after all damage sources, so simultaneous kills work.

---

# PHASE A — Scaffolding

## Step 1 — Toolchain and repo skeleton

**Files:** `pyproject.toml`, `CONVENTIONS.md`, `README.md`, all `__init__.py`,
`scripts/check_install.py`, `.gitignore`

1. Create the tree from Section 3 with empty `__init__.py` files.
2. `pyproject.toml` deps: `torch`, `numpy`, `pyyaml`. Extras:
   `[dev] pytest, pytest-xdist`, `[sb3] stable-baselines3, sb3-contrib, gymnasium`,
   `[render] matplotlib`.
3. Copy Sections 2 and 4 verbatim into `CONVENTIONS.md`.
4. `scripts/check_install.py`:
```pythonQ
torch.__version__, torch.version.cuda, torch.cuda.is_available()
torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)   # expect (12, 0)
torch.cuda.get_arch_list()                                           # MUST contain 'sm_120'
# time a 4096x4096 matmul and a large index_put_(accumulate=True)
# also report whether stable_baselines3 / sb3_contrib import cleanly, and their versions
# exit(1) if sm_120 is absent
```

**Install trap for the README.** The 5070 Ti is Blackwell, **sm_120**. A plain
`pip install torch` can pull a build whose kernels stop at sm_90, failing at runtime with
`no kernel image is available for execution on the device`. Install from the CUDA index:
```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
(or a newer `cu12x`/`cu13x` channel — check pytorch.org/get-started/locally).
**The definitive check is `torch.cuda.get_arch_list()` containing `sm_120`**, not
`torch.cuda.is_available()`, which returns True on a build that cannot launch a kernel.
Windows-native is fine; no WSL required. This sim is tensor-op bound, not matmul bound.

**Acceptance:** `check_install.py` prints `sm_120` and exits 0.

---

## Step 2 — `constants.py`

**Files:** `brawl_sim/constants.py`, `tests/test_constants.py`

```python
class Tile(IntEnum):  FLOOR=0; WALL=1; BUSH=2; WATER=3; FENCE=4; SPAWN=5; BOX=6
class Kind(IntEnum):  HERO_MORTIS=0; BOT_SNIPER=1; BOT_ARTILLERY=2; BOT_MELEE=3; BOT_RIFLE=4
class Proj(IntEnum):  NONE=0; SNIPER_BOLT=1; ARTILLERY_SHELL=2; RIFLE_ARROW=3
class DeathCause(IntEnum): ALIVE=0; COMBAT=1; ZONE=2
N_TILES=7; N_KINDS=5; N_PROJ_KINDS=4; BOT_KINDS=(1,2,3,4)
```

| Tile | Unit | Projectile |
|------|------|-----------|
| FLOOR / SPAWN / BOX | pass | pass |
| WALL | block | block |
| BUSH | pass | pass (hides occupants — see below) |
| WATER | block | **pass** |
| FENCE | block | **pass** |

Derived `(N_TILES,)` bool tensors: `TILE_BLOCKS_UNIT`, `TILE_BLOCKS_PROJ`,
`TILE_IS_BUSH`, `TILE_IS_WATER`, `TILE_IS_SPAWN`, `TILE_IS_BOX_SPAWN`.
Plus `CHAR_TO_TILE` / `TILE_TO_CHAR`.

**No tile-level vision table (Notice 4).** The camera is fixed bird's-eye, so no tile ever
blocks sight — not even `WALL`. There is no `TILE_BLOCKS_VISION`. Bush-hiding is a *perception*
rule (`bots/perception.py`, Step 15: occupancy + `bush_reveal_radius` + reveal timer), not a
terrain-blocking rule, and it's the only thing that ever hides an entity. A genuine **physical**
line-of-sight check (can a straight line of travel reach this point at all) reuses
`TILE_BLOCKS_PROJ` — since only `WALL` is opaque, "blocks a shot" and "blocks a sightline" are
now the same question.

**Acceptance:** the table above is reproduced exactly by `TILE_BLOCKS_UNIT`/`TILE_BLOCKS_PROJ`.

---

## Step 3 — `config.py`

**Files:** `brawl_sim/config.py`, `tests/test_config.py`

```python
@dataclass(frozen=True)
class EnvConfig:
    map_h: int = 60; map_w: int = 60
    map_names: tuple[str, ...] = ("open", "bushy", "walled")
    map_selection: str = "uniform"; fixed_map: str = "open"
    view_h: int = 20; view_w: int = 40                  # landscape
    n_enemies: int = 6
    randomize_enemy_types: bool = True
    enemy_type_weights: tuple[float, ...] = (.25,.25,.25,.25)
    fixed_enemy_types: tuple[str, ...] = ()
    max_projectiles: int = 128; max_boxes: int = 16; max_pickups: int = 32
    dt: float = 0.05; max_episode_steps: int = 3000; action_repeat: int = 1
    action_latency_seconds: float = 0.001               # -> 0 ticks at 20 Hz
    n_move_bins: int = 16                               # + idle => 17
    dash_on_idle: str = "facing"
    obs_include_world_grid: bool = True
    obs_include_privileged: bool = True
    bots_break_boxes: bool = True; bots_collect_cubes: bool = True
    bots_avoid_zone: bool = True; bots_fight_each_other: bool = True
    zone_enabled: bool = True; zone_mode: str = "rect"
    iframes_block_zone: bool = False; regen_enabled: bool = False
    los_step_tiles: float = 0.5; max_ray_tiles: float = 24.0
    debug_checks: bool = False; compile: bool = False; device: str = "cuda"

    @property
    def n_entities(self): return 1 + self.n_enemies
    @property
    def action_nvec(self): return (self.n_move_bins + 1, 2)
    @property
    def ray_steps(self): return int(math.ceil(self.max_ray_tiles / self.los_step_tiles))
    @property
    def action_latency_ticks(self): return int(round(self.action_latency_seconds / self.dt))
    @property
    def latency_buf_len(self): return max(1, self.action_latency_ticks + 1)
```

```python
class SimParams:
    """Per-env numerics. EVERY field is a tensor with a leading (N,) dim, even when the
    config value is a scalar — this keeps downstream code uniform. (N08)"""
    # per-kind (N, K):
    base_hp base_damage move_speed attack_range attack_cooldown max_ammo reload_seconds
    proj_kind(i64) proj_speed proj_radius proj_count(i64) proj_spread_rad
    aoe_radius attack_arc_rad
    aim_noise_std_rad aim_noise_tiles reaction_delay lead_target_fraction decision_period(i64)
    dash_distance dash_duration dash_radius
    # per-env (N,):
    enemy_hp_mult enemy_damage_mult unit_radius pickup_radius
    cube_hp_bonus cube_damage_bonus max_cubes(i64) cubes_per_box(i64) cubes_on_kill_base(i64)
    box_hp n_boxes(i64)
    zone_start_time zone_step_seconds zone_tiles_per_step zone_dps zone_dps_growth
    bush_reveal_radius reveal_after_attack regen_delay regen_per_second
```

```python
def load_config(path, overrides=None) -> EnvConfig
def load_randomization(path) -> RandomizationSpec
def build_params(cfg, n_envs, device, gen, spec) -> SimParams
def resample_params(params, reset_mask, cfg, gen, spec) -> None   # MUTATES
def validate(cfg, params) -> None
```

**Scalar-or-range syntax (N08).** Any parameter may be written either way:
```yaml
base_hp: 3800
base_hp: {low: 3200, high: 4400}
move_speed: {low: 0.9, high: 1.1, mode: multiplicative}
```
Ranges resample **per env, on every reset**. `resample_params` takes a boolean
`reset_mask: (N,)` and writes with `torch.where` over the full batch — no `nonzero()`,
no boolean indexing.

Validation errors for: view larger than map; `n_enemies < 1`;
`max_projectiles < n_entities * max(proj_count) * 4`; unknown map name;
`zone_start_time >= max_episode_steps * dt`; `dt >= zone_step_seconds`;
`action_latency_seconds < 0`; any range with `low > high`.

**Acceptance:** `EnvConfig` frozen and hashable; every `SimParams` field is a leading-`N`
tensor on the right device; scalars identical across envs; ranges differ and resample only
for masked rows; sampling is sync-free.

---

## Step 4 — Config files

**Files:** `configs/default.yaml`, `configs/brawlers.yaml`, `configs/randomization.yaml`,
`configs/agent_obs.yaml`, `configs/presets/{debug_tiny,no_zone,single_archetype}.yaml`

`configs/default.yaml`:
```yaml
seed: 0
device: cuda

world: {map_h: 60, map_w: 60, maps: [open, bushy, walled],
        map_selection: uniform, fixed_map: open}
view:  {height: 20, width: 40}

sim:
  dt: 0.05
  max_episode_steps: 3000
  action_repeat: 1
  action_latency_seconds: 0.001    # rounds to 0 ticks at 20 Hz; buffer is a pass-through

action: {n_move_bins: 16, dash_on_idle: facing}

observation:
  include_world_grid: true
  include_privileged: true
  grid_dtype: uint8

entities:
  n_enemies: 6
  randomize_enemy_types: true
  enemy_type_weights: {sniper: 0.25, artillery: 0.25, melee: 0.25, rifle: 0.25}
  fixed_enemy_types: [sniper, artillery, melee, rifle, sniper, rifle]
  enemy_hp_mult: 1.0
  enemy_damage_mult: 1.0
  unit_radius: 0.4

limits: {max_projectiles: 128, max_boxes: 16, max_pickups: 32}

cubes:
  hp_bonus_per_cube: 0.15
  damage_bonus_per_cube: 0.15
  max_cubes: 12
  cubes_per_box: 1
  cubes_on_kill_base: 1
  drop_victim_cubes: true
  pickup_radius: 0.6

boxes: {n_boxes: 8, hp: 3000}

zone:
  enabled: true
  mode: rect
  start_fraction: 0.2
  step_seconds: 4.0
  tiles_per_step: 1
  dps: 1000.0
  dps_growth_per_step: 200.0
  iframes_block_zone: false

regen: {enabled: false, delay_seconds: 3.0, per_second: 400.0}

bots: {break_boxes: true, collect_cubes: true, avoid_zone: true, fight_each_other: true}

perception:                     # drives BOT behavior and the revealed_* flags.
  bush_reveal_radius: 2.0       # It never removes data from the observation package.
  reveal_after_attack: 1.0
  los_step_tiles: 0.5
  max_ray_tiles: 24.0

engine: {debug_checks: false, compile: false}
```

`configs/brawlers.yaml` — rough placeholders (D05/D06), speeds in tiles/s (~2.4 = normal tier).
Every value accepts scalar-or-range.
```yaml
hero_mortis:
  base_hp: 3800; base_damage: 1300; move_speed: 2.56
  max_ammo: 3; reload_seconds: 1.7; attack_cooldown: 0.30
  dash_distance: 5.0; dash_duration: 0.30; dash_radius: 0.70

bot_sniper:
  base_hp: 2600; base_damage: 1600; move_speed: 2.40
  attack_range: 11.0; max_ammo: 3; reload_seconds: 1.3; attack_cooldown: 0.25
  proj_kind: SNIPER_BOLT; proj_speed: 14.0; proj_radius: 0.25
  proj_count: 1; proj_spread_rad: 0.0; aoe_radius: 0.0
  aim_noise_std_rad: 0.06; reaction_delay: 0.15; lead_target_fraction: 0.9
  decision_period_ticks: 4

bot_artillery:
  base_hp: 3400; base_damage: 1100; move_speed: 2.30
  attack_range: 10.0; max_ammo: 3; reload_seconds: 1.8; attack_cooldown: 0.40
  proj_kind: ARTILLERY_SHELL; proj_speed: 6.0; proj_radius: 0.30
  proj_count: 1; proj_spread_rad: 0.0; aoe_radius: 1.5
  aim_noise_tiles: 0.8
  reaction_delay: 0.25; lead_target_fraction: 0.6; decision_period_ticks: 4

bot_melee:
  base_hp: 6500; base_damage: 500; move_speed: 2.60
  attack_range: 2.0; attack_arc_rad: 0.9
  max_ammo: 3; reload_seconds: 0.9; attack_cooldown: 0.15
  proj_kind: NONE
  aim_noise_std_rad: 0.10; reaction_delay: 0.10; lead_target_fraction: 0.0
  decision_period_ticks: 2

bot_rifle:
  base_hp: 3800; base_damage: 800; move_speed: 2.40
  attack_range: 8.0; max_ammo: 3; reload_seconds: 1.5; attack_cooldown: 0.35
  proj_kind: RIFLE_ARROW; proj_speed: 9.0; proj_radius: 0.25
  proj_count: 3; proj_spread_rad: 0.18; aoe_radius: 0.0
  aim_noise_std_rad: 0.08; reaction_delay: 0.20; lead_target_fraction: 0.7
  decision_period_ticks: 4
```

`configs/randomization.yaml` — ships commented out, with a worked example:
```yaml
# entities.enemy_hp_mult:       {low: 0.8, high: 1.25}
# entities.enemy_damage_mult:   {low: 0.8, high: 1.25}
# bot_sniper.aim_noise_std_rad: {low: 0.03, high: 0.20}
# bot_sniper.reaction_delay:    {low: 0.05, high: 0.40}
# bot_melee.move_speed:         {low: 0.9, high: 1.15, mode: multiplicative}
# hero_mortis.reload_seconds:   {low: 1.5, high: 1.9}
```

`configs/agent_obs.yaml` — **the agent's fair-information view.** Full spec in Step 32.

Presets: `debug_tiny` (20x20 map, 10x14 view, 2 enemies, 300 steps, zone off, `blank` map,
world grid off), `no_zone`, `single_archetype`.

**Acceptance:** all configs load and validate; `debug_tiny` yields `n_entities == 3`;
uncommenting a randomization line produces per-env variation with no other change.

---

# PHASE B — Maps

## Step 5 — Map CSVs

**Files:** `brawl_sim/maps/csv/{blank,open,bushy,walled}.csv`, `brawl_sim/maps/README.md`

One row per grid row, comma-separated single characters, no header, exactly `map_h` rows x
`map_w` columns. Legend: `.` FLOOR, `#` WALL, `b` BUSH, `~` WATER, `f` FENCE, `S` SPAWN
candidate, `X` BOX spawn candidate.

Per map: border ring entirely `#`; at least 12 `S` markers at even angular intervals around
the perimeter (the loader sorts by angle and the spawner depends on it); at least 16 `X`
markers; all floor mutually reachable.

`open` = sparse cover. `bushy` = ~25% bush in large patches. `walled` = dense symmetric walls,
a water channel across the middle, several fences. `blank` = all floor + wall border + 8 `S`
in a ring + 8 `X` in a ring (test fixture).

A generator script is fine, but **CSVs must be committed as plain text** so they're hand-editable.

**Acceptance:** every CSV parses to the configured dimensions; `blank.csv` has one connected
floor region; marker counts meet the minimums.

---

## Step 6 — `maps/loader.py`

**Files:** `brawl_sim/maps/loader.py`, `tests/test_maps.py`

```python
load_map_csv(path, cfg) -> np.ndarray            # (H, W) int64
validate_map(tiles, cfg) -> None
spawn_points(tiles) -> np.ndarray                # (K,2) f32 tile centers, SORTED BY ANGLE
box_points(tiles) -> np.ndarray                  # (M,2) f32

class MapBank:
    tiles blocks_unit blocks_proj is_bush is_water   # each (M, H, W)
    pad_tiles pad_blocks_unit pad_blocks_proj pad_is_bush pad_is_water
        # padded by (view_h//2, view_w//2) with WALL semantics, for the view crop
    spawns (M, 32, 2) f32   n_spawns (M,) i64
    box_spots (M, 64, 2) f32   n_box_spots (M,) i64

build_map_bank(cfg, device) -> MapBank
```

**Critical performance rule:** never materialize a per-env `(N, H, W)` map slice. Point
queries use gather indexing (`bank.blocks_unit[map_id, iy, ix]`). The only per-env grids that
ever exist are the observation grids, built by scatter into a zeroed buffer.

**Acceptance:** leading dim equals `len(cfg.map_names)`; `blank.csv` spawns are angle-sorted;
padded banks have shape `(M, H + view_h, W + view_w)` and their padding reads as blocking.

---

# PHASE C — Core math

## Step 7 — `core/geometry.py`

**Files:** `brawl_sim/core/geometry.py`, `tests/test_geometry.py`

Batched, branch-free, broadcasting over arbitrary leading dims.
```python
safe_norm normalize angle_of from_angle wrap_angle angle_diff
dir_from_bin(idx, n_bins) bin_from_dir(v, n_bins)      # bin 0 == angle 0
dist dist2 rotate perp
point_in_cone(p, origin, facing, radius, half_angle) -> bool
segment_circle_hit(p0, p1, c, r) -> (hit, t)
capsule_contains(p0, p1, r, q) -> bool
closest_point_on_segment(p0, p1, q)
closest_approach(p, v, q) -> (t, dist)                 # projectile-vs-point, for obs features
lead_target(shooter, target, target_vel, proj_speed, fraction)
```

**Acceptance:** `bin_from_dir(dir_from_bin(k,16),16) == k` for all k; `segment_circle_hit`
matches a brute-force reference to 1e-3 on 1000 random cases; no NaN on zero-length inputs;
all pass with a `(64, 7)` leading batch.

---

## Step 8 — `core/terrain.py`

**Files:** `brawl_sim/core/terrain.py`, `tests/test_terrain.py`

```python
to_tile(pos) -> (ix, iy)
oob(ix, iy, cfg) -> bool
sample(bank_mask, map_id, pos, cfg) -> bool             # OOB returns True (blocked)
circle_blocked(bank_mask, map_id, pos, radius, cfg) -> bool
resolve_move(bank_mask, map_id, pos, delta, radius, cfg) -> new_pos
    # axis-separated sliding: try x alone, keep if legal; then y alone. Branch-free.
march(bank_mask, map_id, p0, dir, max_dist, cfg) -> (hit, hit_pos, hit_t)
    # uniform march, cfg.ray_steps FIXED samples at los_step_tiles spacing; mask beyond
    # max_dist; first blocked index via argmax over the bool tensor.
line_of_sight(bank, map_id, p0, p1, cfg) -> bool
```

**Memory note.** This is a generic geometric primitive — it doesn't know about bots or
targeting, only terrain. Its main caller is `bots/perception.raw_los()` (Step 15, physical LOS
for the `obs["visibility"]["los"]` annotation and the sniper/rifle fire-gate), which needs it
all-pairs: `(N,E,E,S)`. At `N=4096, E=7, S=48` that's 9.6M sample points; int64 index tensors
are ~77 MB each. Mitigations in order: `los_step_tiles=0.5` (default, halves `S`); derive a
smaller `S` as a **Python int** from a smaller `max_dist` at the call site; chunk the entity
dimension. Never materialize per-env maps. **Bot targeting itself (`perception.visibility()`)
never calls this — see Notice 4** — so this cost does not scale with how aggressively bots
acquire targets, only with how much LOS annotation the observation and fire-gates ask for.

**Acceptance:** a unit walking into a wall slides; `line_of_sight` (via `blocks_proj`) is False
through WALL only, True through WATER, BUSH, and FENCE (Notice 4 — fence no longer blocks
shots); `march` on `blank.csv` terminates at the border in every direction; nothing allocates
an `H*W` dimension.

---

# PHASE D — State

## Step 9 — `core/state.py`

**Files:** `brawl_sim/core/state.py`, `tests/test_state.py`

```python
class SimState:
    # entities (N, E, ...)
    ent_pos(N,E,2) ent_vel(N,E,2) ent_facing(N,E)
    ent_hp ent_max_hp (N,E)  ent_alive(N,E)bool  ent_kind(N,E)i64  ent_cubes(N,E)i64
    ent_ammo ent_attack_cd (N,E)
    ent_dash_t(N,E) ent_dash_dir(N,E,2) ent_dash_speed(N,E) ent_dash_hits(N,E,E)bool
    ent_invuln_t ent_reveal_t ent_react_t ent_no_damage_t (N,E)
    ent_target(N,E)i64  ent_move_smooth(N,E,2)
    ent_death_step(N,E)i32  ent_death_cause(N,E)i32  ent_last_hit_by(N,E)i64
    ent_damage_dealt ent_damage_taken (N,E)  ent_kills(N,E)i32  ent_shots_fired(N,E)i32
    # projectiles (N, P, ...)
    prj_pos prj_vel prj_target (N,P,2)
    prj_dist_left prj_damage prj_radius prj_aoe prj_age (N,P)
    prj_owner prj_kind (N,P)i64   prj_lobbed prj_alive (N,P)bool
    # boxes / pickups
    box_pos(N,B,2) box_hp box_max_hp(N,B) box_alive(N,B)bool
    pku_pos(N,U,2) pku_cubes(N,U)i64 pku_alive(N,U)bool pku_age(N,U)
    # zone / episode
    zone_lo zone_hi (N,2)  zone_next_t(N,)  zone_step(N,)i32
    map_id(N,)i64  time(N,)  step_count(N,)i32  n_alive(N,)i32  boxes_broken(N,)i32
    # action latency ring buffer
    act_buf(N, L, 2)i64  act_head(N,)i64
    # cached
    env_idx(N,)i64
```
```python
allocate(cfg, n_envs, device) -> SimState
zero_(state, reset_mask) -> None                 # torch.where over the full batch
check_invariants(state, cfg) -> None             # only when cfg.debug_checks
snapshot(state, env_index: int) -> dict          # CPU, rendering and tests ONLY
```
`check_invariants` covers: no NaN/Inf; `0 <= pos <= (map_w, map_h)`; `0 <= hp <= max_hp`;
dead entities have `hp == 0`; `cubes <= max_cubes`; `dash_t <= dash_duration`; live
projectiles `<= P`; `act_head in [0, L)`. It replaces the guarantees functional purity would
have given you — make it thorough.

**Acceptance:** `allocate(n_envs=4096)` reports under ~250 MB with a per-field breakdown;
every tensor on `device` with the documented dtype; `zero_` on a mask touching rows 1 and 5
leaves other rows bit-identical.

---

## Step 10 — `core/stats.py`

**Files:** `brawl_sim/core/stats.py`, `tests/test_stats.py`

```python
gather_kind(param_nk, kind_ne) -> (N,E)          # param_nk.gather(1, kind_ne)
effective_max_hp(kind, cubes, params)            # base_hp*(1+cube_hp_bonus*cubes)
                                                 #   * enemy_hp_mult where kind != HERO
effective_damage(kind, cubes, params)
effective_speed(kind, params)
is_hero(kind) -> bool
apply_cube_gain(hp, kind, old_cubes, new_cubes, params) -> hp
    # current HP rises by the same ABSOLUTE amount as max HP
```
A bug in `gather_kind` silently gives every entity the hero's stats — test it directly.

**Acceptance:** hero with 2 cubes at `0.15` has exactly `1.30x` base HP and damage;
`enemy_hp_mult = 2.0` doesn't change the hero; `gather_kind` matches a naive loop.

---

# PHASE E — Simulation systems

## Step 11 — `core/hero.py` (action decoding + dash)

**Files:** `brawl_sim/core/hero.py`, `tests/test_hero.py`

```python
def action_mask(state, params, cfg) -> dict[str, Tensor]
    # {"move": (N,17) bool, "attack": (N,2) bool}
    # move always all-True. attack[:,1] ("fire") True only where
    #   alive & ammo >= 1.0 & attack_cd <= 0 & dash_t <= 0
    # Surfaced to SB3 via action_masks() in Step 33.

def decode_action(action, state, params, cfg) -> (move_dir (N,2), fire (N,) bool)
    # action (N,2) i64. [:,0] in [0,16]: 0 = idle, k = dir_from_bin(k-1,16).
    #                    [:,1] in [0,1]: fire, ANDed with the mask — illegal fire is a
    #                    silent no-op, never an error.

def tick_timers(state, params, cfg) -> None
    """MUTATES: ent_ammo, ent_attack_cd, ent_dash_t, ent_invuln_t, ent_reveal_t,
    ent_react_t, ent_no_damage_t. ammo += dt/reload_seconds clamped to max_ammo;
    reload continues during a dash; countdowns clamp at 0."""

def start_dash(state, fire, move_dir, bank, params, cfg) -> None
    """MUTATES: ent_ammo, ent_attack_cd, ent_dash_t, ent_dash_dir, ent_dash_speed,
    ent_dash_hits, ent_invuln_t, ent_facing, ent_shots_fired.
    Direction: move_dir where non-zero, else from_angle(facing).
      cfg.dash_on_idle == "block" instead suppresses the dash on idle input.
    Clip the path NOW: march along dash_dir out to dash_distance against blocks_unit,
      so WALL and WATER both stop it. dash_speed = clipped_distance / dash_duration.
    Consume 1 ammo, set attack_cd, dash_t = dash_duration, invuln_t = dash_duration,
    zero this entity's dash_hits row."""

def advance_dash(state, params, cfg) -> (dmg_ent (N,E), dmg_by (N,E,E), dmg_box (N,B))
    """MUTATES: ent_pos, ent_vel, ent_dash_t, ent_dash_hits.
    Dashers move by dash_dir*dash_speed*dt. Sweep a capsule of radius dash_radius from
    old to new pos; every live enemy inside it not already flagged in dash_hits takes
    effective_damage and is flagged. Boxes in the capsule take damage too. Dashers pass
    through units; no terrain re-resolve (path was clipped at start); no separation.
    dash_t -= dt; on reaching 0 clear dash_dir/dash_speed and the dash_hits row."""
```

**The dash replaces the walk.** `start_dash` runs in tick phase 6, `apply_movement` in phase 7
skipping anything with `dash_t > 0`. Test this explicitly. Keep both functions
**kind-agnostic**, gated on `dash_distance[kind] > 0`, so melee lunges and Mortis's Super can
reuse them later.

**Acceptance:** dashing at a wall 2 tiles away travels ~2 tiles, not 5; dashing at water stops
at the shoreline; one dash through a stationary enemy damages it exactly once despite
overlapping for several ticks; a dashing tick shows zero walk displacement; firing at
`ammo = 0.9` does nothing and the mask says so; ammo goes 0 → full in
`max_ammo * reload_seconds`; the hero takes no projectile damage while `dash_t > 0` but does
take zone damage.

---

## Step 12 — `core/movement.py`

**Files:** `brawl_sim/core/movement.py`, `tests/test_movement.py`

```python
def apply_movement(state, move_dir, bank, params, cfg) -> None
    """MUTATES: ent_pos, ent_vel, ent_facing."""
```
1. `move_dir` `(N,E,2)`, normalized or zero.
2. **Skip `~alive` or `dash_t > 0`** — multiplicative mask, not indexing.
3. Delta = `normalize(move_dir) * effective_speed * dt`.
4. Terrain collision via `resolve_move`.
5. **Soft separation:** dense `(N,E,E)` pairwise displacement; where `0 < d < 2*unit_radius`
   push each apart by `0.5*(2r-d)` along the connecting axis; sum per entity; one iteration;
   exclude dead entities and dashers.
6. Re-run `resolve_move` on the separation push.
7. `ent_vel = (new_pos - old_pos)/dt`. Update `ent_facing` **only** where `move_dir` is
   non-zero — the idle-dash fallback depends on facing persisting.

**Acceptance:** walking into a wall stops and slides; coincident units separate within ~10
ticks; after 10k random ticks nothing is inside a `blocks_unit` tile; `ent_facing` survives
idle ticks; dashers are untouched.

---

## Step 13 — `core/projectiles.py`

**Files:** `brawl_sim/core/projectiles.py`, `tests/test_projectiles.py`

```python
def alloc_slots(prj_alive, demand (N,E) i64, max_per_entity: int) -> (idx, ok)
```
`max_per_entity` is a Python int (3). Collision-free across all `E` shooters in the same env
on the same tick: `free_rank = cumsum(~prj_alive) - 1` over `(N,P)`;
`offset = exclusive_cumsum(demand)` over `(N,E)`; map rank → slot with `searchsorted`;
`ok = claimed_rank < total_free`. **A full buffer drops the shot silently** — never overwrite
a live projectile, never raise.

```python
def spawn_volley(state, fire_mask, origin, aim_dir, aim_point, kind, damage, params, cfg) -> None
    """MUTATES: all prj_*. proj_count[kind] projectiles in a symmetric fan of
    proj_spread_rad. lobbed = (proj_kind == ARTILLERY_SHELL). dist_left = attack_range."""

def step_projectiles(state, bank, params, cfg) -> (dmg_ent, dmg_by, dmg_box)
    """MUTATES: all prj_*."""
```
Order: (1) integrate, decrement `dist_left`, increment `age`; (2) wall collision for
non-lobbed via `march`, kill at hit point; (3) unit collision over `(N,P,E)` excluding owner
and dead, earliest `t` wins, accumulate into `dmg_by[n, owner, victim]` with
`index_put_(accumulate=True)`; (4) box collision over `(N,P,B)`; (5) artillery detonates when
it passes its landing point (`dot(target - pos, vel) <= 0`) or `dist_left <= 0`, damaging every
live entity and box within `aoe` — lobbed shells ignore units and walls in flight entirely;
(6) expiry at `dist_left <= 0`. Damage is *reported* here, *applied* in Step 14.

**Acceptance:** a bolt fired at a wall dies at the wall; over water it reaches the far side;
an artillery shell aimed past a wall lands and damages a unit behind it; a rifle volley
occupies exactly 3 slots; simultaneous volleys never collide in allocation; firing into a
full buffer changes nothing.

---

## Step 14 — `core/combat.py`

**Files:** `brawl_sim/core/combat.py`, `tests/test_combat.py`

```python
apply_damage(state, dmg, cause, attacker, params, cfg) -> None
    """MUTATES: ent_hp, ent_damage_taken, ent_last_hit_by, ent_no_damage_t.
    Where invuln_t > 0, COMBAT damage is zeroed. ZONE damage is NOT zeroed unless
    cfg.iframes_block_zone (default false)."""
melee_hitscan(state, fire_mask, bank, params, cfg) -> (dmg_ent, dmg_by, dmg_box)
    # all live entities/boxes within attack_range, inside attack_arc of facing, with a clear
    # physical path (terrain.line_of_sight via blocks_proj — walls only, Notice 4)
    # dmg_box added retroactively in Step 22, once boxes existed to hit — see that step's note.
resolve_deaths(state, cfg) -> newly_dead (N,E) bool
    """MUTATES: ent_alive, ent_death_step, ent_death_cause, ent_kills (via
    ent_last_hit_by), n_alive."""
drop_cubes_on_death(state, newly_dead, params, cfg) -> None
    """MUTATES: pku_*. One pickup per corpse with cubes_on_kill_base +
    (victim cubes if drop_victim_cubes). Zone deaths drop cubes too."""
collect_pickups(state, params, cfg) -> gained (N,E) i64
    """MUTATES: ent_cubes, ent_hp, ent_max_hp, pku_alive. Build the (N,E,U) eligibility
    matrix, argmax over E (lowest index wins ties), mask so only the winner claims.
    Capped entities do not claim and leave the pickup."""
apply_regen(state, params, cfg) -> None
    """MUTATES: ent_hp. No-op when disabled."""
```

**Acceptance:** an entity at exactly 0 HP is dead; a pickup with 3 entities on it is consumed
once; a capped entity leaves it; a dashing hero takes 0 combat damage but full zone damage;
killing a bot holding 3 cubes drops 4.

---

# PHASE F — Bots

## Step 15 — `bots/perception.py`

**Files:** `brawl_sim/bots/perception.py`, `tests/test_perception.py`

**Bots are limited by bush-hiding only — that's the real game's rule (fixed bird's-eye camera,
terrain never occludes) and it stays. The observation package is not further restricted beyond
that.** This module drives bot targeting *and* supplies the annotation flags, and both read the
exact same `visibility()` result — bots get no advantage a human on the same camera wouldn't
have (Notice 4).

```python
visibility(state, bank, params, cfg) -> (N,E,E) bool
    # vis[n,i,j] = i sees j. True iff i alive AND j alive AND
    #   ( j not in bush OR dist <= bush_reveal_radius OR j.reveal_t > 0 ).
    # No terrain lookup at all — walls/fences/water never hide anything (Notice 4).
    # vis[n,i,i] = True. Dead observers see nothing.
raw_los(state, bank, cfg) -> (N,E,E) bool
    # Physical wall LOS only (terrain.line_of_sight via bank.blocks_proj), independent of
    # bush. Feeds obs["visibility"]["los"] and any archetype fire-gate that needs a genuine
    # clear shot (sniper Step 17, rifle Step 20) — NOT bot targeting, which is bush-only
    # (visibility(), above). An entity can be vis=True/raw_los=False (behind a wall, not in
    # a bush) or vis=False/raw_los=True (bush-hidden, no wall in the way).
in_bush(state, bank) -> (N,E) bool
team_id(kind, cfg) -> (N,E) i64
select_target(state, vis, cfg) -> None             # MUTATES ent_target, sticky
incoming_threat(state, params, cfg) -> (N,E,2)
nearest_alive(points_pos, points_alive, from_pos) -> (idx, dist)
in_zone(pos, zone_lo, zone_hi) -> bool
nearest_safe_point(pos, zone_lo, zone_hi) -> (...,2)
```

**Acceptance:** a bot in a bush 5 tiles away isn't visible; at 1.5 tiles it is; a bot behind a
wall but not in a bush is still visible in `vis` (targeting ignores walls) while `raw_los` is
False for it; a bot that fired within `reveal_after_attack` is visible in a bush;
`select_target` doesn't oscillate between equidistant targets.

---

## Step 16 — `bots/steering.py`

**Files:** `brawl_sim/bots/steering.py`, `tests/test_steering.py`

Each returns an unnormalized `(N,E,2)` direction:
`seek`, `flee`, `strafe(sign)`, `maintain_range(desired, deadband)`,
`avoid_walls(probe_dist)` (8 preallocated probe offsets), `escape_zone`,
`combine(*weighted)` (weighted sum then normalize).

**Acceptance:** `maintain_range(desired=8)` moves a bot at 12 closer and one at 3 away;
`avoid_walls` steers out of a dead-end; `combine` never returns NaN on all-zero input.

**Finalized signatures** (for Steps 17-20 to call against): every primitive is a pure function
of plain tensors, never `state`/`params` directly, so archetypes pass in whatever positions
they've already gathered (e.g. `state.ent_pos[gather ent_target]`):
`seek(pos, target_pos)`, `flee(pos, threat_pos)`, `strafe(pos, target_pos, sign)` (`sign`:
python +-1 or an `(N,E)` tensor), `maintain_range(pos, target_pos, desired, deadband)`,
`avoid_walls(pos, map_id, bank, probe_dist, cfg)` (8 fixed `dir_from_bin` compass probes
against `bank.blocks_unit`, same pattern as `terrain.circle_blocked`), `escape_zone(pos,
zone_lo, zone_hi)` (bounds passed explicitly, matching Step 15's `perception.in_zone` /
`nearest_safe_point` -- `core/zone.py`, Step 23, is the eventual source), `combine(*weighted)`
where each item is a `(direction, weight)` pair and `weight` is a python scalar or `(N,E)`
tensor.

---

## Steps 17–20 — The four archetypes (parallelizable)

```python
def policy(state, vis, bank, params, cfg, gen) -> BotIntent

class BotIntent:                  # every field batched (N,E)
    move_dir(N,E,2)  fire(N,E)bool  aim_dir(N,E,2)  aim_point(N,E,2)
```
Each archetype computes its intent for **all** `(N,E)` entities; Step 21 selects by kind.

**Shared rules (in the `bots/policy.py` docstring):** pure w.r.t. state, no memory beyond
`ent_target`/`ent_react_t`/`ent_move_smooth`; fire gate
`alive & ammo>=1 & attack_cd<=0 & react_t<=0 & has_target & dist<=attack_range`;
aim via `lead_target` + `randn(generator=gen)*aim_noise_std_rad`; zone escape weight 3.0 when
`in_zone`; box approach when `bots_break_boxes` and no visible enemy and a box within 10 tiles;
cube collection when `bots_collect_cubes` and a pickup within 8 tiles and no enemy within 6.
**Never return NaN** — guard every `normalize`.

**`bots/policy.py` built early, during Step 17** — `BotIntent` and the shared-rule helpers
above (`gather_rows`, `target_info`, `fire_gate`, `aim`, `zone_contribution`,
`box_contribution`, `cube_contribution`) already exist; Step 21 only needs to add
`all_bot_intents`'s dispatch/selection-by-kind and reaction-delay/decision-period logic on top
of this same file, not the shared rules themselves. Every archetype imports this module as
`policy as shared` (plain `policy` collides with each archetype's own `policy()` function
name). Filled-in decisions made there, not given literally by the spec text above:
  - `zone_contribution` is also gated on `cfg.bots_avoid_zone` (D14 lists zone-fleeing as
    toggleable exactly like box-breaking/cube-collecting; the preamble's zone line just
    doesn't repeat the toggle name).
  - It also treats a degenerate `zone_hi <= zone_lo` rect as "no zone active" rather than
    trusting `state.zone_lo`/`zone_hi` outright, since `core/zone.py` (Step 23) hasn't run yet
    for any state built before that step — `allocate()`'s zero-init leaves both at `(0,0)`, a
    zero-area rect, not the full-map "everything safe" rect a real reset produces. Costs
    nothing once Step 23 lands.
  - Box/cube approach weight is fixed at 1.0 (personality-neutral baseline, matching Step 19
    melee's own `seek` 1.0) — the plan gives an exact weight for zone escape (3.0) but not
    these two.
  - Every archetype gathers stats via `state.ent_kind` (`stats.gather_kind`, the same pattern
    every other module uses) rather than a hardcoded kind constant. "Each archetype computes
    its intent for all (N,E) entities" only matters for the rows Step 21 actually keeps (where
    `ent_kind` already equals that archetype's own kind), so the two approaches agree exactly
    on every row that isn't discarded anyway.

**Retroactive fix to `bots/steering.maintain_range` (Step 16), found while building Step 17** —
its distance was computed with `keepdim=True` (shape `(N,E,1)`), then compared directly against
`desired`/`deadband`. When those are `(N,E)` tensors (as every real caller passes, e.g. sniper's
`0.85 * attack_range`), broadcasting silently treated `E` as if it were a second entity axis
instead of raising a shape error whenever `E` happened to equal 2 (the position vector's own
last-dim size) — comparing the wrong pairs instead of crashing. Step 16's own tests never caught
it because they all used `E=1`. Fixed by dropping `keepdim` and unsqueezing only the final bool
masks; a regression test with `E=2` and tensor `desired`/`deadband` now covers it.

**Step 17 `sniper.py`** — `maintain_range(0.85*range, deadband 1.5)`; pull toward the nearest
bush within 4 tiles when no enemy visible (weight 0.4); fires on `raw_los` (physical, walls
only — Notice 4) + range; `flee` weight 1.0 below 40% HP. The nearest-bush search (a fixed
9x9-grid-clipped-to-a-radius-4-circle tile scan) lives locally in `sniper.py`, not in
`perception.py`/`steering.py`, since no other archetype needs a tile-search primitive like it
yet — move it to a shared module if one eventually does.

**Step 18 `artillery.py`** — **does not need physical LOS to fire**, only range. Uses `vis`
(bush-only targeting, Notice 4) for target *selection*, then keeps firing at that target's
current position while in range even without `raw_los` (document the choice). Aim = leaded
position + `randn * aim_noise_tiles` on the **landing point**. Weight 0.5 toward directions
where `raw_los` to the target is False (wants a wall for cover, even though it can still fire
without one). `maintain_range(0.7*range, deadband 2.0)`.

Implementation: "weight 0.5 toward directions where `raw_los` is False" is a private
`_cover_direction(pos, target_pos, map_id, bank, cfg)` helper local to `artillery.py` (same
"no other archetype needs this yet, so it doesn't live in a shared module" call as sniper's
`_nearest_bush`) — it probes the same 8 fixed `dir_from_bin` compass offsets as
`steering.avoid_walls`/`terrain.circle_blocked`, at a fixed `_COVER_PROBE_DIST = 2.0` tiles,
and sums the offsets of whichever probes would have `raw_los` (`terrain.line_of_sight`) to the
target come back False, i.e. steers toward wherever cover actually exists nearby rather than
an arbitrary "away from the wall" heuristic. It's evaluated unconditionally (not gated on the
bot's own current `raw_los`), matching "wants a wall for cover, even though it can still fire
without one" — the pull toward cover is a standing personality trait, not a reaction to
currently being exposed. Positional aim noise (`randn * aim_noise_tiles` added directly to the
leaded landing point) is implemented locally too, not via `bots/policy.aim` (Step 17's shared
*angular*-noise helper) — the two archetypes genuinely use different noise models, per the
plan text above.

**Step 19 `melee.py`** — `seek` 1.0 + `strafe` 0.3, sign `(-1)**entity_index`. Fires when the
target is in `attack_range` and inside `attack_arc` — **no projectile**; damage comes from
`combat.melee_hitscan` in phase 6. Never retreats. Patrols toward the zone center when no
enemy is visible.

Implementation notes:
  - The arc check in `fire` reads `state.ent_facing` as-is (via `geo.point_in_cone`, the same
    primitive `combat.melee_hitscan` itself uses) rather than trying to anticipate this tick's
    movement. Section 4's tick order runs attack resolution (phase 6, `melee_hitscan`) BEFORE
    movement (phase 7) — and bot policies (phase 4) run before phase 6 too — so `policy()` and
    `melee_hitscan` read the exact same pre-movement facing within a tick; turning to face a
    target costs an extra tick or two of `seek`, same as a real melee attacker having to turn
    before swinging.
  - "Patrols toward the zone center when no enemy is visible" falls back to the **map's**
    geometric center whenever the zone rect is degenerate (`zone_hi <= zone_lo`) — the exact
    same guard `bots/policy.zone_contribution` (Step 17) already uses for the same reason:
    `core/zone.py` (Step 23) hasn't run yet, or the zone is disabled, and `allocate()`'s
    zero-init leaves `zone_lo == zone_hi == (0,0)`, a corner of the map, not a usable center.
  - `bot_melee.aim_noise_std_rad` (`configs/brawlers.yaml`) is a **dead field for this
    archetype** — melee's attack is a deterministic geometric arc-check (`fire`'s value
    already *is* the miss/hit determination), not an aimed shot with angular noise, so
    `melee.py` never reads it. Steps 17/18/20's shared "with `aim_noise_std_rad = 0` it never
    misses a stationary target at half range" acceptance line doesn't have a literal analogue
    here (melee has neither an aim-noise-driven miss chance nor a `maintain_range`-style
    `desired_range`/half-range concept); Step 19's acceptance instead treats "closes to and
    holds `attack_range`, lands a hit within 200 ticks" as covering both.

**Step 20 `rifle.py`** — `maintain_range(0.75*range, deadband 2.0)`, `strafe` 0.5. Fires a
3-projectile fan; only when `dist <= 0.9*attack_range` since the fan diverges. Holds fire on
fast lateral movers (skip if perpendicular target speed `> 2.0` tiles/s and
`dist > 0.6*attack_range`). `flee` weight 1.0 below 30% HP. Like sniper, also requires
`raw_los` (Notice 4 names both "the sniper/rifle fire-gate in Steps 17/20" explicitly).
`strafe_sign` (the `(-1)**entity_index` pattern) moved from `melee.py` into
`bots/policy.py` as `policy.strafe_sign`, now that a second archetype needs it verbatim —
`melee.py` was updated to call the shared version instead of keeping its own copy.

**Discovered emergent dynamic, not a bug — read before changing any of these four numbers.**
`0.9*attack_range` (7.2, the fire-eligible distance) sits *inside*
`maintain_range(0.75*range, deadband 2.0)`'s settle band (`[4, 8]`), unlike sniper/artillery
where the whole settle band is fire-eligible. `strafe` (weight 0.5) is unconditional on
distance and is the *only* active force once inside the deadband, since `maintain_range`
itself is exactly zero there (Step 16's own spec). A bot approaching from outside the deadband
can settle anywhere in `[4, 8]`, including the `(7.2, 8]` sliver where it's holding range
correctly but still can't fire; from there, `strafe`'s straight-line tangential step per tick
causes a slow, fully deterministic outward orbital drift (radius grows by roughly
`speed^2 / (2*radius)` per tick — the standard error of approximating a circular arc with a
straight chord), which eventually pushes it back out past 8, re-engaging `maintain_range` and
pulling it back in past 7.2. The cycle is geometric, not noise-driven — reproducible
regardless of seed. Measured at 14 tiles' starting distance: first shot lands at tick 214, not
within the other three archetypes' 200-tick bar. Resolution: left the four numbers exactly as
specified (D05/D06: exact numbers don't matter, but this interaction would recur for any
choice where `fire_fraction * range < desired + deadband`) and gave `rifle.py`'s own
full-loop acceptance test a 300-tick budget instead, with the mechanism documented inline —
adjust `configs/brawlers.yaml`'s numbers later if faster initial engagement is wanted, now
that the relationship causing the delay is understood.

**Acceptance (each):** in `debug_tiny` against a stationary hero, the bot reaches and holds
roughly its `desired_range` within 100 ticks and lands a hit within 200; with
`aim_noise_std_rad = 0` it never misses a stationary target at half range; no NaN over 10k
ticks of randomized state.

---

## Step 21 — `bots/policy.py`

**Files:** `brawl_sim/bots/policy.py`, `tests/test_bot_dispatch.py`

```python
def all_bot_intents(state, vis, bank, params, cfg, gen) -> BotIntent
```
1. `perception.select_target`.
2. Call all four archetype policies (Python `for` over 4 constants is fine).
3. Select with a chain of `torch.where` on `ent_kind`.
4. **Reaction delay / decision period**, with no per-bot scratch memory:
   fire gate `((step_count + entity_index) % decision_period) == 0`; movement low-passed into
   `ent_move_smooth` at rate `clamp(dt/reaction_delay, 0, 1)`, and `ent_move_smooth` is the
   actual intent.
5. Zero the intent for entity 0 and all dead entities.

**Acceptance:** each entity's intent matches its archetype; changing
`params.aim_noise_std_rad` changes behavior with no structural change; no `(N,H,W)` allocation.

**Implementation notes:**
  - The four archetype modules are imported *inside* `all_bot_intents`, not at the top of
    `bots/policy.py` -- each of `sniper.py`/`artillery.py`/`melee.py`/`rifle.py` already does
    `from . import policy as shared` to reach Step 17's `BotIntent`/`fire_gate`/`aim`/
    `*_contribution`/`strafe_sign`, so an eager top-level import here would be circular. By the
    time `all_bot_intents` is actually called, `policy.py` has already finished initializing
    (it's what the archetype modules import first), so the lazy import resolves cleanly.
  - Decision period and reaction delay are genuinely different mechanisms, matching D19 listing
    them as two separate per-archetype difficulty knobs: `decision_period` **discretely gates
    `fire` only** (`(step_count + entity_index) % decision_period == 0` -- entities are
    staggered by slot, so e.g. all sniper-kind bots don't all decide on the same tick just
    because they share a `decision_period`); `reaction_delay` **continuously low-passes
    `move_dir` only**, via an EMA into the persistent `ent_move_smooth` state field at
    `rate = clamp(dt/reaction_delay, 0, 1)` -- `ent_move_smooth` (not the raw per-tick
    selection) is what the returned `BotIntent.move_dir` actually is, per the plan text.
  - Step 5's "zero the intent for entity 0 and all dead entities" is applied to the **output**
    only, after decision-period gating and move-smoothing (matching the plan's own item
    ordering, 5 after 4) -- `ent_move_smooth`'s underlying *state* for entity 0 / dead entities
    is left un-zeroed, since it keeps getting overwritten with whatever archetype intent was
    selected for that slot every tick regardless (harmless: `movement.apply_movement` already
    multiplicatively masks out `~alive` entities via its own gate, and the hero's movement
    comes from `decode_action`, Step 11, never from `ent_move_smooth`).
  - Verified the "no `(N,H,W)` allocation" bar directly, not just structurally: `n_envs=4096`
    on the `bushy` map, mixed archetypes across all enemy slots, 60 steady-state ticks (after a
    20-tick warmup) show exactly 0 bytes of net CUDA memory growth.

---

# PHASE G — World systems

## Step 22 — `core/boxes.py`

**Files:** `brawl_sim/core/boxes.py`, `tests/test_boxes.py`

```python
spawn_boxes(state, reset_mask, bank, params, cfg, gen) -> None
    """MUTATES box_*. n_boxes distinct spots without replacement: draw
    (N, MAX_BOX_SPOTS) random keys, set invalid spots to +inf, topk smallest n_boxes."""
damage_boxes(state, dmg_box) -> None
resolve_broken_boxes(state, params, cfg) -> newly_broken (N,B) bool
    """MUTATES box_alive, pku_*, boxes_broken. One pickup per broken box."""
```
Boxes are static, block nothing, take damage from projectiles, AoE, melee cones, and the dash.

**Acceptance:** boxes never spawn on a `blocks_unit` tile; no two share a spot;
`n_boxes > n_box_spots` clamps; breaking a box yields exactly `cubes_per_box`.

**Retroactive fix found here, applied in `core/combat.py` (Step 14):** this step's own text
says boxes already take damage "from projectiles, AoE, melee cones, and the dash" — true for
the dash (Step 11) and projectiles (Step 13), which both already computed `dmg_box`, but
`combat.melee_hitscan` (Step 14) never did; there was nothing to hit when Step 14 was written,
so the gap was invisible until boxes actually started spawning here. `melee_hitscan` now
additionally returns `dmg_box (N,B)`, computed with the same in-cone + physical-LOS test
already used for units. Its return signature changed from `(dmg_ent, dmg_by)` to
`(dmg_ent, dmg_by, dmg_box)`; every call site (tests, this step's own CUDA check) was updated.

**Implementation notes:**
  - `spawn_boxes` generalizes "topk smallest n_boxes" to a full sort (`topk` with
    `k = min(cfg.max_boxes, MAX_BOX_SPOTS)`, `largest=False`) followed by a `rank < n_boxes_env`
    mask, rather than calling `topk` with `k = n_boxes` directly — `n_boxes` is a `SimParams`
    field (N08: scalar-or-range, resampled per env) and can legitimately differ across envs in
    the same batch, but `torch.topk` takes one Python int `k` for the whole call. Also clamps
    `n_boxes_env` against `cfg.max_boxes` (`state.box_pos`'s actual slot count `B`), not just
    `n_box_spots` — the acceptance text only names the latter, but `B` is an equally real upper
    bound the spec's own pseudocode didn't need to mention because `max_boxes` and `n_boxes`
    are ordinarily configured consistently; asserted false here for a config where they aren't
    (`test_n_boxes_greater_than_max_boxes_clamps`).
  - `spawn_boxes` writes ranks 0..k-1 directly into box slots 0..k-1 rather than using
    `projectiles.alloc_slots`' incremental collision-free allocator — that scheme exists for
    slots claimed piecemeal, over many ticks, by many independent callers (shots landing,
    corpses dying); a box respawn is a single all-at-once full replace of every slot for a
    masked env, so a direct rank-to-slot write is both correct and simpler.
    `resolve_broken_boxes` DOES reuse `alloc_slots`/`_set_scalar`/`_set_vec2`, for the same
    reason `combat.drop_cubes_on_death` does: box-breaking happens piecemeal, one box at a
    time, against the same shared `pku_*` pool other systems claim from on the same tick.

---

## Step 23 — `core/zone.py`

**Files:** `brawl_sim/core/zone.py`, `tests/test_zone.py`

```python
init_zone(state, reset_mask, params, cfg) -> None
step_zone(state, params, cfg) -> None
    """Where time >= zone_next_t: lo += tiles_per_step, hi -= tiles_per_step (clamped so
    hi-lo >= 2), zone_next_t += zone_step_seconds, zone_step += 1. Exactly one shrink per
    tick; validation guarantees dt < step_seconds."""
zone_damage(state, params, cfg) -> (N,E) f32
zone_grid(state, cfg, out_h, out_w, origin) -> Tensor   # reused for view AND world grids
```
`zone_start_time = start_fraction * max_episode_steps * dt`, computed in `config.py`.

**Acceptance:** with `start_fraction=0.2`, 3000 steps, `dt=0.05`, first shrink at `t=30.0 s`;
outside entities lose `dps*dt` per tick; the rect never inverts; `zone_enabled: false`
produces zero damage and no shrink.

**Implementation notes:**
  - `_outside_rect` (the `perception.in_zone` predicate) is duplicated locally rather than
    imported from `bots/perception.py` -- `core/` never depends on `bots/` (the dependency
    only ever runs the other way; every archetype already imports `core/` freely), and
    relocating `perception.in_zone`'s existing, already-tested public call sites over a
    two-line comparison wasn't worth entangling the module graph for.
  - `init_zone` is unconditional on `cfg.zone_enabled` -- a disabled zone should still carry a
    sane, non-degenerate full-map rect (not `allocate()`'s zero-init `(0,0)-(0,0)`), so that
    `bots/policy.zone_contribution`'s degenerate-rect guard (Step 17) never has to trigger on a
    freshly-reset env regardless of whether the zone feature itself is on.
  - `step_zone`'s clamp keeps the shrink symmetric: `new_lo`/`new_hi` are clamped against the
    rect's OWN pre-shrink center (`min(lo + step, center - 1)` / `max(hi - step, center + 1)`),
    not against a running min/max, so a rect that's already at the 2-tile floor stays parked
    there exactly (never inverts) on every subsequent call rather than only the one that first
    hits it.
  - Verified "no shrink" under `zone_enabled: false` with `state.time` set 10000s past any
    conceivable shrink schedule, not just past the first one.

---

## Step 24 — `core/spawn.py`

**Files:** `brawl_sim/core/spawn.py`, `tests/test_spawn.py`

Batched **partial** reset with no Python loop and no `nonzero()`.

```python
sample_map_ids(reset_mask, bank, cfg, gen) -> (N,) i64
sample_enemy_kinds(reset_mask, cfg, gen) -> (N,E) i64
    # index 0 always HERO_MORTIS; multinomial over 4 archetypes, or fixed_enemy_types
sample_spawn_positions(reset_mask, map_ids, bank, cfg, gen) -> (N,E,2) f32
    # spawns are angle-sorted. Draw a random rotation r in [0, n_spawns), then
    #   idx_k = (r + floor(k * n_spawns / E)) % n_spawns     for k in 0..E-1
reset_envs(state, reset_mask, bank, params, cfg, gen, spec) -> None
    """MUTATES everything for masked rows via torch.where over the full batch.
    Order: zero_ -> resample_params -> map_id -> kinds -> positions ->
    hp/max_hp/ammo/facing -> clear projectiles & pickups -> spawn_boxes -> init_zone ->
    clear the latency ring buffer -> time/step_count/n_alive/counters."""
```
Entities spawn facing the map center, `hp = max_hp = effective_max_hp(kind, 0, params)`,
`ammo = max_ammo`, `cubes = 0`, `alive = True`, `death_step = -1`.

**Acceptance:** across 1000 seeds no two entities spawn within `2*unit_radius`; angular gaps
differ by at most one slot; every spawn is on a passable tile; a mask covering rows 2 and 5
leaves other rows bit-identical (checksum test); no sync warnings.

**Implementation notes:**
  - "No Python loop" (the step header) means no loop over N or E as a *dynamic* count -- every
    per-env quantity stays fully batched. `sample_spawn_positions`' `for k in range(E)` is the
    same "fixed compile-time constant" shape as `bots/policy.all_bot_intents`' loop over the 4
    archetypes or `steering.avoid_walls`' loop over 8 probe directions: `cfg.n_entities` is a
    static Python int off a frozen `EnvConfig`, not a tensor, so the loop is unrolled at trace
    time and never touches N. No `.nonzero()` and no host sync anywhere in the module.
  - `zero_(state, reset_mask)` (Step 9) already clears every resettable field for masked rows
    before `reset_envs` runs any of its own logic -- including `prj_alive`/`pku_alive` ("clear
    projectiles & pickups"), `act_buf`/`act_head` ("clear the latency ring buffer"), and
    `time`/`step_count`/`boxes_broken`/every `ent_kills`-style counter. Those pseudocode bullets
    are a **consequence** of `zero_`, not separate code in this module -- `reset_envs` only
    touches the fields whose 0/False default is semantically wrong for a fresh spawn: `map_id`,
    `ent_kind`, `ent_pos`, `ent_hp`/`max_hp`/`ammo`/`facing`/`alive`, and two sentinel fields
    (next bullet).
  - `ent_target = -1` on spawn was a pre-planted forward pointer: `bots/perception.select_target`
    (Step 15) already documents that `ent_target < 0` means "no target" and names this exact
    step as responsible for the sentinel, since `zero_`'s blanket 0-init would otherwise read as
    "targeting entity 0". `ent_death_step = -1` needs the same treatment for a parallel reason
    that isn't pre-documented anywhere else: `step_count` also starts at 0, so a freshly-spawned,
    never-died entity's `zero_`-inited `death_step` (0) would be indistinguishable from "died on
    step 0" -- both are now handled explicitly in `reset_envs`.
  - `core/spawn.py` is the first `core/` module to import from `config.py` directly
    (`resample_params`, `ARCHETYPE_SHORT_NAMES`) instead of only receiving `cfg`/`params` as
    plain call arguments. This doesn't create a cycle -- `config.py` sits below `core/` in the
    dependency graph (it imports only `.constants`, never anything under `core/`), unlike the
    deliberately-avoided `core/` -> `bots/` direction from Step 23.
  - `cfg.map_selection` supports exactly the two modes the config schema and every shipped
    config actually use, `"uniform"` (independent per-env draw) and `"fixed"` (every env pinned
    to `cfg.fixed_map`'s index in `cfg.map_names`) -- anything else raises a clear `ValueError`
    rather than silently defaulting, matching `config.validate`'s existing fail-fast style for
    unknown map names.
  - `sample_enemy_kinds` raises if `len(cfg.fixed_enemy_types) != cfg.n_enemies` when
    `randomize_enemy_types` is `False` -- every shipped config already satisfies this (e.g.
    `default.yaml`'s 6 fixed types for `n_enemies: 6`), so this is a fail-fast guard against a
    future config edit silently breaking the index-0-is-always-hero contract, not a new
    constraint on existing behavior.

---

# PHASE H — The full observation package

> The simulator is an **information source**, not a feature extractor. Nothing is hidden,
> sorted away, or truncated. Fairness gating happens later, in Phase J, where it's
> configurable and auditable.

## Step 25 — `core/observation.py`

**Files:** `brawl_sim/core/observation.py`, `tests/test_observation.py`

```python
def build_obs(state, bank, vis, raw_los, params, cfg) -> dict
```
A **nested dict of named tensors**, all leading `(N,)`. Positions in **raw tile coordinates**
*and* normalized form, since you parse this externally.

**`obs["hero"]`** — mirror of entity slot 0:
`pos(N,2) pos_norm vel facing facing_vec(N,2) hp max_hp hp_frac alive cubes ammo ammo_frac
ammo_whole attack_cd can_attack dashing dash_t dash_dir(N,2) invuln in_bush in_zone tile(N,2)
damage_dealt damage_taken kills shots_fired rank`

**`obs["entities"]`** — **all E entities, index-stable, never sorted, never masked.** Slot `k`
is always entity `k`; slot 0 is the hero.
- *Ground truth:* `alive kind kind_onehot(N,E,5) team pos pos_norm vel speed facing hp max_hp
  hp_frac cubes ammo ammo_frac attack_cd can_attack dashing dash_t dash_dir invuln in_bush
  in_zone tile`
- *Hero-relative:* `rel_pos dist bearing rel_vel closing_speed in_view dist_rank`
  (`dist_rank` is ordering **as a field**, not as a sort)
- *Visibility annotation — flags only:* `revealed_to_hero hero_revealed_to los_from_hero
  hidden_by_bush`
- *Outcome:* `death_step death_cause damage_dealt damage_taken kills last_hit_by`
- **`obs["entities"]["privileged"]`** (R04): `target_id react_t reveal_t decision_phase
  move_intent`. Namespaced so it cannot reach the agent by accident — Step 32 refuses to
  select from this subtree.

**`obs["projectiles"]`** — **all P slots**, index-stable:
`alive kind kind_onehot(N,P,4) pos pos_norm vel speed heading owner owner_kind damage radius
aoe lobbed dist_left age rel_pos dist time_to_closest closest_dist threatens_hero in_view`

**`obs["boxes"]`** — all B: `alive pos pos_norm hp max_hp hp_frac rel_pos dist in_view`
**`obs["pickups"]`** — all U: `alive pos pos_norm cubes age rel_pos dist in_view`

**`obs["zone"]`**: `lo hi lo_norm hi_norm active step dps next_shrink_in hero_margin(N,4)
safe_area_frac`

**`obs["visibility"]`**: `vis(N,E,E) los(N,E,E) dist_matrix(N,E,E)` — ~200 KB at
`n_envs=4096` and invaluable for reward attribution.

**`obs["view"]`** — egocentric crop `(N, 12, 20, 40)` `uint8`, out-of-bounds reads as WALL,
**full information**:
```
 0 blocks_unit   1 blocks_projectile   2 is_bush        3 is_water
 4 in_zone       5 enemy_any           6 enemy_revealed 7 enemy_hidden
 8 hero          9 box                10 pickup        11 projectile
```
Channels 6 and 7 partition channel 5 by `revealed_to_hero`, so Phase J can build a fair view
or a cheating view without the simulator deciding.

**`obs["world"]`** — full-map `(N, 12, 60, 60)` `uint8`, same channels, when
`obs_include_world_grid`. ~162 MB at `n_envs=4096`; first thing to disable under VRAM
pressure, and you lose nothing since `obs["entities"]["pos"]` carries exact float positions.

**`obs["action_mask"]`**: `{"move": (N,17) bool, "attack": (N,2) bool}`

**`obs["meta"]`**: `map_id time step_count time_frac n_alive n_enemies_alive
episode_step_limit map_h map_w view_h view_w`

**Implementation notes.** View crop: precompute `dy = arange(view_h) - view_h//2` and
`dx = arange(view_w) - view_w//2` at init; per step
`iy = hero_iy[:,None,None] + dy + pad_y`, `ix = hero_ix[:,None,None] + dx + pad_x`, gather
from the **padded** bank so indices are always in range with no clamping. Occupancy channels
built by `index_put_(accumulate=True)` into a zeroed buffer (keep the counts — more
information; document that values may exceed 1). **Preallocate every output buffer in
`__init__`; `build_obs` allocates nothing.** Nothing here reads `vis` to decide what to
include — `vis` only populates `revealed_*`/`hidden_*` and channels 6–7.

**Acceptance:** shapes match `obs_schema`; every float field finite; an enemy in a bush 5
tiles away is **still fully present** with correct `pos`, `revealed_to_hero = False`,
`hidden_by_bush = True`, in channels 5 and 7 but not 6; `dist_rank` is a valid permutation;
a corner view is roughly half wall; `build_obs` allocates nothing per call (memory-delta test).

**Implementation notes (decisions filled in beyond the plan's literal text):**
  - **Layering:** `core/observation.py` never imports `bots/` — same discipline as
    `core/zone.py` (Step 23) and `core/spawn.py` (Step 24). `build_obs`'s signature already
    takes `vis`/`raw_los` as precomputed arguments, so the caller (the future `env.py` step
    loop, which needs both for bot targeting earlier in the same tick anyway) supplies them.
    The only two bits of `bots/perception.py` this module would otherwise reach for —
    `in_bush` and `team_id` — are duplicated locally (5 lines and 1 line respectively) rather
    than imported, for the same reason `zone.py`'s `_outside_rect` is.
  - **`rank` is not a stored state field.** Section 4's phase-13 pseudocode ("death
    resolution ... record rank + cause") predates `core/combat.py` (Step 14), whose actual
    `resolve_deaths` never grew a rank field — there's no `ent_rank` in `SimState`. Rather than
    retrofit already-shipped, already-tested Step 14 for a value that's fully recoverable from
    fields it already writes, `compute_rank` derives it live from `ent_alive`/`ent_death_step`
    every call: alive entities all tie at `rank == n_alive` (final placement isn't decided
    until they die); a dead entity's rank is `1 + count(entities that outlived it)` — everyone
    still alive, plus anyone who died later.
  - **`build_obs` stayed a plain function**, not a class with buffers preallocated in
    `__init__` as this step's own text suggests. Every hot-path module through Step 24
    (`movement.apply_movement`, `combat.melee_hitscan`, `projectiles.step_projectiles`, ...) is
    a plain function that allocates fresh intermediates every call, and "allocates nothing" has
    consistently meant "steady-state resident memory doesn't grow" (verified via memory-delta,
    per Step 21's own stricter check) rather than literally zero `torch.zeros` calls. A
    buffer-owning class would matter for `torch.compile`'s cudagraph mode (Step 31) but nothing
    before that needs it; deferred to whichever step actually wires up `compile`.
  - **View/world grid unification:** one private `_build_grid(state, bank, vis, cfg, origin,
    out_h, out_w)` builds both `obs["view"]` and `obs["world"]` — exactly the "same
    rasterization, different window" pattern `zone.zone_grid` (Step 23) already established,
    reused directly for channel 4. Terrain channels 0-3 always index the **padded** bank
    tensors with a `+ pad_h`/`+ pad_w` shift regardless of which grid is being built; for
    `obs["world"]` (`origin=(0,0)`) that shift lands exactly back on the unpadded region (Step
    6's `test_build_map_bank_padded_shape_and_blocking` already proves
    `pad_tiles[:, pad_h:-pad_h, pad_w:-pad_w] == tiles`), so one indexing path serves both
    without a branch. Occupancy channels 5-11 accumulate into a separate **int32** buffer via
    `index_put_(accumulate=True)` — `uint8` has no atomic-add guarantee under CUDA — and are
    clamped to `[0, 255]` only at the very end, when cast down to the `uint8` the schema wants.
  - **`bearing`** (entities' hero-relative fields) is egocentric: `angle_diff(angle_of(rel_pos),
    hero_facing)`, not the raw world-frame angle (which `facing`/`facing_vec` already cover
    elsewhere) — "how far off-center is this from where the hero is looking" is the
    information a bearing field is for.
  - **`dist_rank`** is `argsort(argsort(dist_from_hero))` (0 = nearest, ties broken by entity
    index via a stable sort) — a valid permutation by construction, computed against **every**
    entity regardless of alive state (dead entities keep their frozen last position, which
    stays a meaningful distance, matching D08's "nothing hidden" over any cleverer scheme that
    pushes corpses to the back).
  - **`threatens_hero`** (projectiles): alive, not hero-owned, and its `closest_approach` to the
    hero's current position is within `unit_radius + proj_radius` — a real "will this actually
    hit" predictor via `geo.closest_approach`, the same primitive `bots/perception.
    incoming_threat` (Step 15) already uses for its own steering-facing threat vector.
  - **`can_attack`** (entities, and hero's own copy of the same field) generalizes
    `hero.action_mask`'s `fire_ok` formula (`alive & ammo>=1 & attack_cd<=0 & dash_t<=0`) from
    hero-only to all `E` entities, computed once and sliced for the hero rather than duplicated
    — bots have no `has_target`/range gate in this field on purpose; that's
    `bots/policy.fire_gate`'s job (Step 17), a separate, bot-specific question from "is this
    entity mechanically able to fire right now."
  - `core/zone.py` grew a small `current_dps(state, params)` helper, factored out of
    `zone_damage`'s existing two-line formula on its second use (`obs["zone"]["dps"]`) — same
    "promote to a shared function on second use" pattern as `projectiles.alloc_slots`.

---

## Step 26 — `core/obs_schema.py` and `docs/OBSERVATION.md`

**Files:** `brawl_sim/core/obs_schema.py`, `scripts/dump_obs_schema.py`, `docs/OBSERVATION.md`

```python
OBS_SCHEMA: dict     # dotted_name -> {shape, dtype, units, range, description, privileged}
obs_spec(cfg) -> dict
validate_obs(obs, cfg) -> None
describe_obs(obs, env_index=0) -> str      # pretty-print ONE env with names/values/units
to_numpy(obs) -> dict
```
`scripts/dump_obs_schema.py` regenerates `docs/OBSERVATION.md` from `OBS_SCHEMA`, so the docs
cannot drift.

**Acceptance:** regeneration is deterministic; `validate_obs` passes on a real observation and
fails informatively on a corrupted one; **every field in `build_obs` output appears in
`OBS_SCHEMA` and vice versa** (set-comparison test — this is the mechanism that keeps the docs
honest); `describe_obs` on `debug_tiny` fits on one screen.

**Implementation notes (decisions filled in beyond the plan's literal text):**
  - **`OBS_SCHEMA` is one declarative table** (`_ROWS`, ~130 rows, one per leaf field) rather
    than hand-written `FieldSpec` calls scattered around — the same "declarative table + a
    small builder" shape `config.py`'s `PER_KIND_FIELDS`/`PER_ENV_FIELDS` and `state.py`'s
    `_ENTITY_FIELDS` already use, for the same reason: a long, mechanically similar list that
    has to stay in sync with real code is much safer as one flat table than as scattered
    literals. `_ROWS`' order matches `observation.py`'s own dict-construction order exactly, so
    `describe_obs`/`dump_obs_schema.py` render top-to-bottom for free with no separate sort key.
  - **Shape dims are symbolic strings** (`"E"`, `"P"`, `"B"`, `"U"`, `"H"`, `"W"`, `"VH"`,
    `"VW"`, `"MOVE"`) resolved against a real `cfg` by `obs_spec(cfg)`, except the leading `"N"`
    (batch size), which stays unresolved even there — `n_envs` isn't part of `EnvConfig`, it's
    a separate argument to env construction, so there's no cfg value to resolve it against.
    `validate_obs` reads `"N"` off the actual tensor's own leading dim instead.
  - **A `conditional` field** (an `EnvConfig` boolean attribute name, or `None`) drives both
    `obs_spec`'s filtering (drops the field when the attribute is falsy) and
    `dump_obs_schema.py`'s doc annotation — used today for exactly two things:
    `obs["world"]` (`obs_include_world_grid`) and the five `entities.privileged.*` fields
    (`obs_include_privileged`). This is what makes the set-comparison test pass with either
    toggle in either state, not just the all-on case.
  - **`validate_obs` checks shape, dtype, and float-finiteness — not `range`.** `range` is
    documentation (rendered into `docs/OBSERVATION.md`), useful to a human or to a future
    normalization layer, but several fields' true bounds are cfg-dependent in ways not worth
    encoding structurally (e.g. `pos` is bounded by `map_w`/`map_h`, not a fixed constant) —
    `range: None` there means "unbounded or cfg-dependent," not "unchecked because untyped."
  - **`describe_obs` summarizes rather than dumps** the big per-slot/grid/matrix fields so a
    `debug_tiny` observation fits on one screen: `entities` prints one line per slot (small, E
    is always ≤ a handful), but `projectiles`/`boxes`/`pickups` print only alive slots (capped
    at 8, "... and N more" beyond that), `visibility.vis` collapses to a single "hero sees K
    other entities" count, and `view`/`world` print shape plus one summary stat instead of
    16,800+ raw grid values. Like `state.snapshot` and `to_numpy`, `describe_obs` is a host-
    transferring debug utility and must never be called from the hot path.
  - `scripts/dump_obs_schema.py`'s determinism comes for free from `OBS_SCHEMA` being built off
    one fixed-order tuple (`_ROWS`) — Python dicts preserve insertion order, so iterating
    `OBS_SCHEMA.items()` filtered by group is already a stable sub-sequence every run; no
    explicit sort was needed.

---

## Step 27 — `core/events.py`

**Files:** `brawl_sim/core/events.py`, `tests/test_events.py`

```python
compute_done(state, cfg) -> (terminated (N,), truncated (N,))
    # terminated = (~hero_alive) | (n_alive == 1 & hero_alive)
    # truncated  = step_count >= max_episode_steps
compute_info(state, dmg_by, newly_dead, newly_broken, cubes_gained, cfg) -> dict
```
`info` holds per-tick **events** (the observation holds per-tick *state*), all device tensors:
```
damage_matrix(N,E,E)       # who damaged whom this tick — key for reward attribution
damage_dealt_tick(N,E)  damage_taken_tick(N,E)
kills_tick(N,E)  deaths_tick(N,E)bool  death_cause_tick(N,E)
cubes_gained_tick(N,E)  boxes_broken_tick(N,)  shots_fired_tick(N,E)  dash_hits_tick(N,)
hero_rank(N,)  terminated  truncated  time  step_count
```
Episode-cumulative counters live in `SimState`, zeroed on reset, mirrored into `obs`.

**Acceptance:** `hero_rank` is 0 exactly when the hero is last alive; `damage_matrix` row sums
equal `damage_dealt_tick`; `kills_tick` matches a manual count over a scripted 500-tick
rollout; `terminated` and `truncated` never both true except on a final-tick death.

**Implementation notes (decisions filled in beyond the plan's literal text):**
  - **`compute_info`'s `dmg_by` contract:** the caller is expected to pass the SUM of every
    combat damage source's own `(N,E,E)` matrix this tick (`hero.advance_dash` +
    `combat.melee_hitscan` + `projectiles.step_projectiles`) — `combat.apply_damage` is invoked
    once per source and never mixes causes, but `compute_info` only takes one combined matrix.
    Zone damage has no attacker (`combat.py`'s `_NO_ATTACKER = -1` sentinel) and structurally
    cannot appear in an attacker x victim matrix, so `damage_matrix`/`damage_dealt_tick`/
    `damage_taken_tick` are combat-only — `obs["entities"]["damage_taken"]` (the cumulative
    counter, Step 14/25) is the complete record across every cause including zone; this
    module's per-tick fields are not. This is documented as the contract `env.py` (Step 29)
    must honor when it becomes the real caller, since nothing enforces it structurally today.
  - **`compute_rank` promoted from `observation.py` to public** (was `_compute_rank`) on its
    second use — same "promote on second use" pattern as `zone.current_dps`. `info["hero_rank"]`
    is **0-indexed** (0 = winner) per this step's own acceptance text, while
    `obs["hero"]["rank"]` (Step 25) is **1-indexed** (1 = winner) — both come from the exact
    same underlying computation, this one shifted by `-1`. Two different indexing conventions
    for the same number is a real footgun; both call sites' docstrings say so explicitly.
  - **`kills_tick`, not a diff of `ent_kills`:** `combat.resolve_deaths` already mutates the
    *cumulative* `ent_kills` counter in place before `compute_info` ever runs, and
    `compute_info` has no "before this tick" snapshot to subtract. Instead it replays
    `resolve_deaths`' own credit rule (`newly_dead & last_hit_by >= 0`, credited to
    `clamp(last_hit_by, min=0)`) against this tick's `newly_dead` directly — the same rule,
    recomputed fresh rather than diffed, which is why the acceptance test (summing `kills_tick`
    across a 500-tick scripted rollout and comparing to the final `ent_kills`) passes exactly.
  - **`death_cause_tick` is `ent_death_cause` masked to `newly_dead`.** The raw state field is
    a *permanent* historical value once an entity has ever died (Step 14) — reading it
    unmasked would make an entity that died 200 ticks ago show up as "dying" in every
    subsequent tick's info. Elsewhere (never-died) it reports `DeathCause.ALIVE` (0).
  - **`shots_fired_tick`/`dash_hits_tick` are best-effort, not exact**, given the parameters
    this step's signature actually provides (no fire-attempt or per-source-attribution
    parameter exists). `shots_fired_tick` is a **lower bound**: `(damage_dealt_tick > 0)` — a
    shot or dash that fires and connects with nothing is invisible, since `dmg_by` only records
    outcomes, not attempts. `dash_hits_tick` **is** exact, via a different route: dash and
    other attacks are mutually exclusive for the same entity on the same tick (N02/N03/R02,
    "dash fully replaces the walk/attack that tick"), so any `dmg_by` entry whose attacker is
    currently dashing (`ent_dash_t > 0`) is unambiguously a dash hit, even after dash/melee/
    projectile damage has already been summed into one matrix. **Forward pointer for Step
    29:** if reward shaping ever needs a true miss-inclusive fire-attempt count, `env.py`'s
    step loop is the natural place to widen `compute_info`'s signature with an explicit
    `fired_mask` from `all_bot_intents`/`decode_action`, since `compute_info` as specified here
    structurally cannot see a miss.

---

## Step 28 — `core/reward.py`

**Files:** `brawl_sim/core/reward.py`, `tests/test_reward.py`

The simulator returns zeros. Reward is applied at the SB3 boundary (Step 33).

```python
class RewardFn(Protocol):
    def __call__(self, obs: dict, info: dict, cfg: EnvConfig) -> Tensor: ...   # (N,) f32

class ZeroReward:
    """Default. Returns a preallocated zeros tensor; allocates nothing per step."""

class ExampleReward:
    """NOT tuned, NOT recommended — a syntactically valid worked example showing how to
    read obs + info. Terminal-only: +1 last alive, -1 dead, 0 otherwise.
    Delete or replace; it exists so the SB3 smoke test in Step 36 has something to run."""
```
Both `BrawlVecEnv` and `BrawlSB3VecEnv` accept `reward_fn`. The docstring must state that
everything needed for any reward is already in `obs` + `info`, so no reward requires touching
the simulator.

**Acceptance:** `ZeroReward` returns exactly zeros, `(N,)` float32, the same tensor object each
step; a custom `RewardFn` works without modifying `env.py`.

**Implementation notes (decisions filled in beyond the plan's literal text):**
  - **`RewardFn` is `@runtime_checkable`.** `typing.Protocol` gives structural typing either
    way (any callable with a matching signature satisfies it, no inheritance needed — the whole
    point of using a `Protocol` here instead of an ABC), but `runtime_checkable` additionally
    lets `isinstance(fn, RewardFn)` work, which is a cheap way for `env.py` (Step 29) to
    validate a caller-supplied `reward_fn` early with a clear error, and makes the "a plain
    function satisfies this" property directly testable rather than just asserted in prose.
  - **`ZeroReward`'s "allocates nothing per step"** is implemented as a `(n_envs, device)`-keyed
    cache: the zeros tensor is built once per distinct pair and the same object is returned on
    every later call for that pair, satisfying the acceptance criterion's "same tensor object
    each step" literally, not just "an equal-valued tensor." It reallocates correctly (and
    still returns a fresh, correctly-shaped, all-zero tensor) if `n_envs` or `device` ever
    changes mid-run — a real scenario for e.g. a smaller eval batch sharing one `ZeroReward`
    instance with training. Callers must treat the returned tensor as read-only, since it's
    shared across every future call for that shape/device; documented on the class itself.
  - **`ExampleReward` derives "dead" from `terminated & ~hero_alive`**, not a separate
    `info["hero_rank"]` or death-cause check — `info["terminated"]` (`core/events.py`, Step 27)
    is already exactly "hero dead OR hero last alive," and those two cases are mutually
    exclusive and exhaustive whenever `terminated` is True, so `hero_alive` alone disambiguates
    them with no extra state needed. A truncation that catches the hero alive but not alone
    (timeout mid-match) correctly falls through to the terminal `0`, since `terminated` is
    False there.

---

# PHASE I — Native environment

## Step 29 — `env.py`

**Files:** `brawl_sim/env.py`, `tests/test_env.py`

```python
class BrawlVecEnv:
    def __init__(self, cfg, n_envs, device=None, seed=0,
                 reward_fn=ZeroReward(), randomization=None): ...
    def reset(self, reset_mask=None) -> dict
    def step(self, action, override=None) -> (obs, reward, terminated, truncated, info)
        # action:   (N, 2) int64 -> the hero
        # override: (N, E, 2) int64 | None -> drives any entity externally (N09)
    @property observation_spec
    @property action_spec                       # {"nvec": (17, 2)}
    def snapshot(self, env_index: int) -> dict  # CPU, rendering only
```
`step()` runs the 17 phases from Section 4 as small named private methods
(`_pop_action_buffer`, `_tick_timers`, `_decode`, `_bot_phase`, `_override_phase`,
`_attack_phase`, `_movement_phase`, `_dash_phase`, `_projectile_phase`, `_zone_phase`,
`_box_phase`, `_pickup_phase`, `_death_phase`, `_zone_schedule`, `_bookkeeping`, `_observe`,
`_autoreset`), each under ~40 lines.

**Action latency:** `_pop_action_buffer` writes the incoming action into `act_buf[:, act_head]`
and reads from `act_head - action_latency_ticks` (mod `L`). At 0 ticks it's a pass-through.
Cleared on reset.

**Autoreset:** done envs reset within the same `step`. Returned `obs` is the **first
observation of the new episode**; the finished episode's final obs/info go in
`info["final_observation"]` / `info["final_info"]`. **Document loudly.**

**Acceptance:** `n_envs=4096` GPU: reset + 1000 random steps, no error, no NaN;
`debug_checks=True` passes `check_invariants` every tick of a 3000-step rollout; every episode
ends within `max_episode_steps + 1`; **zero host syncs** verified with
`torch.cuda.set_sync_debug_mode("error")` around a 100-step loop (hard criterion); a dashing
tick shows zero walk displacement; peak VRAM under ~2.5 GB at 4096 envs with the world grid on.

**This was the first step to actually exercise every earlier step together, end to end, and it
surfaced several real bugs in already-tested code that no single step's isolated tests could
have caught.** All are fixed and covered by this step's own tests; each is called out below
rather than folded silently into the diff.

**Retroactive fixes to already-shipped code, found while building this step's integration
tests:**
  - **`terrain.march` could miss a wall entirely** when a caller's `max_dist` was smaller than
    one `los_step_tiles` (0.5 by default) — every fixed sample landed beyond `max_dist` and got
    masked out by `in_range`, so `hit` was `False` no matter what was actually there. Any
    projectile slower than `los_step_tiles / dt` (10 tiles/s at the defaults — true of every
    archetype in `brawlers.yaml` except nothing, since the fastest, sniper's bolt, is 14 tiles/s
    but rifle's arrow at 9 tiles/s and artillery's shell at 6 tiles/s both qualify) could pass
    straight through a wall, undetected, forever. Fixed by adding one extra sample exactly at
    `max_dist` itself, on top of the regular fixed-spacing samples — provably harmless for the
    long-range LOS case (`line_of_sight`, `hero.start_dash`'s dash clip): it only ever wins the
    "first hit" pick when none of the regular samples did, i.e. exactly when it's needed.
  - **`hero.start_dash`'s dash-clip distance could land a dasher marginally past a wall's true
    boundary** — `march`'s `hit_t` is the distance to the first BLOCKED *sample*, not the true
    boundary, which can be up to one `los_step_tiles` closer to the attacker. Fixed by backing
    the clip off by `cfg.los_step_tiles`. `test_hero.py`'s existing dash-clip tests already
    tolerated exactly this magnitude of change (`abs(traveled - expected) < los_step_tiles +
    1e-3`), so this needed no test updates.
  - **The same class of bug in `projectiles.step_projectiles`' wall-kill position** — fixed the
    same way, backing `wall_hit_pos` off by `los_step_tiles` along the travel direction.
  - **Lobbed (artillery) shells had no bounds clamping at all** and could fly off the map:
    `bots/artillery.py`'s lead prediction plus its positional aim noise can push `aim_point`
    past the map edge, and a lobbed shell flies a straight ballistic line to `prj_target` with
    no wall/bounds check by design (Step 18: they arc *over* walls). `projectiles.spawn_volley`
    now clamps `aim_point` into a small inset of the map, and — since a lobbed shell's own
    detonation trigger *is* "have I passed `aim_point`" — recomputes `aim_dir` from the clamped
    point too, only for lobbed shots (non-lobbed `aim_dir` carries genuine angular firing noise,
    `bots/policy.aim`, that must not be overwritten).
  - **`ent_damage_dealt` was declared in `SimState` (Step 9), read by `observation.py` (Step
    25), and named as a maintained "cumulative counter" by `events.py`'s own docstring (Step
    27) — but nothing anywhere ever wrote to it.** `combat.apply_damage` only updates the
    *victim's* `damage_taken`; no lower-level function aggregates the *attacker's* side across
    melee/dash/projectile sources landing on possibly-different victims in the same tick. Fixed
    in `_bookkeeping`, from the same combined attacker-victim matrix `events.compute_info`
    already needs (combat damage only, matching `damage_dealt_tick`'s own definition — zone
    damage has no attacker to credit).
  - **`torch.tensor(python_list, device="cuda")` is a real host sync** — confirmed directly with
    `torch.cuda.set_sync_debug_mode("error")`, which is the first CUDA verification in this
    project to actually check for this rather than a memory-delta proxy. `torch.full(scalar,
    device=...)` is not a sync; combining several via `torch.stack`/`torch.cat` is the safe
    replacement. This pattern was already present in four places from earlier, already-tested
    steps — `core/observation.py` (`map_wh`), `core/spawn.py` (`map_center`, and the enemy-kind
    weights/fixed-list tensors), `core/zone.py` (`init_zone`'s full-map rect), and
    `bots/melee.py` (`_patrol_direction`'s map-center fallback) — all fixed via two new small
    helpers: `geometry.vec2(x, y, device, dtype)` for the 2-element case, and a local
    `spawn._const_tensor(values, device, dtype)` for the arbitrary-length enemy-weights case.

**New design decisions, filled in beyond the plan's literal text:**
  - **`__init__` loads `configs/default.yaml` + `configs/brawlers.yaml` from fixed,
    package-relative paths** (mirroring `maps/loader.py`'s `CSV_DIR` precedent) to build the
    `spec` dict `build_params`/`resample_params` need — `EnvConfig` structurally cannot carry
    brawler stats (Step 3), so there is no way to derive `spec` from `cfg` alone. **This is a
    documented sharp edge**: if a caller's `cfg` came from `load_config(..., overrides=...)`
    and any override touched a SimParams-only field (`base_hp`, `zone.dps`,
    `entities.enemy_hp_mult`, ...), that override is invisible here unless the same merged
    dict is *also* passed via the new `spec=` constructor argument (not in the plan's literal
    signature, added for exactly this reason).
  - **`reward_fn` defaults to `None`**, constructing a fresh `ZeroReward()` inside `__init__`,
    not the plan text's literal `reward_fn=ZeroReward()` — a mutable default argument would be
    shared across every `BrawlVecEnv` built without an explicit `reward_fn`, and `ZeroReward`
    is stateful (Step 28's `(n_envs, device)`-keyed cache). Standard Python correctness fix.
  - **Regen is folded into `_tick_timers`**, not its own numbered phase — Section 4 never lists
    one, but `combat.apply_regen` (Step 14) reads `ent_no_damage_t`, a timer, and running it
    right after timers advance (BEFORE this tick's own combat) means a fatal hit dealt later
    the same tick can never be "un-fatal-ed" by a regen tick that hasn't seen it yet. Running
    it later (e.g. at bookkeeping) would risk exactly that.
  - **Ammo/cooldown/`shots_fired` bookkeeping for ranged and melee attacks is done by hand in
    `_attack_phase`** — neither `projectiles.spawn_volley` nor `combat.melee_hitscan` mutates
    `ent_ammo`/`ent_attack_cd`/`ent_shots_fired` (only `hero.start_dash` does, for dashes).
    This also closes the gap `core/events.py` (Step 27) flagged explicitly in its own module
    docstring: `ent_shots_fired` previously only ever counted dashes, so ranged/melee bots
    never registered as having fired, in either the cumulative counter or the observation.
    They do now.
  - **`combat.dominant_attacker(dmg_by) -> (N,E) i64`** (new, public) resolves `apply_damage`'s
    one-attacker-per-victim `attacker` parameter against a same-tick multi-attacker matrix
    (a wide melee cone, converging projectiles from different owners, two simultaneous
    dashers): whoever dealt the most damage gets kill credit / `ent_last_hit_by`. Used
    identically by `_attack_phase`, `_dash_phase`, and `_projectile_phase`.
  - **`override`'s `-1` sentinel and scope**: `override[n,e,0] == -1` means "no override for
    this entity slot" (valid move bins are 0..`n_move_bins`, so `-1` is unambiguous); otherwise
    `override[n,e]` is decoded exactly like the hero's own action and REPLACES that entity's
    `move_dir`/`fire` for the tick. It does not touch `aim_dir`/`aim_point` — override drives
    movement/attack-timing, not targeting; an overridden bot still aims via its own archetype
    logic. A blanket `alive & ammo>=1 & attack_cd<=0 & dash_t<=0` gate is ANDed into the final
    `fire` mask right before attack resolution, specifically because override has no built-in
    gate the way `bots/policy.fire_gate` and `hero.action_mask` already do — this makes it
    structurally impossible for any source, including an external override, to force an
    illegal attack through.
  - **`observation.clone_obs(node)`** (new): recursively `.clone()`s every tensor leaf. Needed
    because `build_obs`/`compute_info` return views/direct references into `state`'s own
    tensors (Step 25/27's zero-copy style, fine for the routinely-returned per-step `obs`/
    `info`, which is now documented as valid only until the next `step()`/`reset()` call) —
    but `info["final_observation"]`/`info["final_info"]` specifically must survive the
    autoreset that happens moments later in the SAME `step()` call, which mutates that exact
    storage in place. **Without this, a real bug reproduced immediately**: a hero forced to 0
    HP showed up as alive, full-HP in its own "final" observation, because `spawn.reset_envs`
    had already overwritten the memory `final_observation`'s tensors were still viewing. Caught
    by this step's own test suite, not a defensive-programming guess. The clone (and dropping
    the pre-clone reference) happens in `step()` itself, not inside `_autoreset` — `step()`'s
    own local variable is a second, independent reference to the same large tensors
    (`obs["world"]` alone is ~177 MB at `n_envs=4096`) that would otherwise keep them alive
    through `_autoreset`'s post-reset rebuild regardless of anything `_autoreset` does
    internally; freeing them first measurably lowered peak VRAM and was the difference between
    passing and failing this step's own VRAM budget.
  - **Visibility/observation are computed three separate times per tick** (bot targeting,
    pre-reset "final" state, post-reset returned state) — each reflects genuinely different
    state and none can be cached across the others; documented prominently in the module
    docstring as the least obvious performance characteristic of this module.

---

## Step 30 — Autoreset + `wrappers/episode_stats.py`

**Files:** `brawl_sim/wrappers/episode_stats.py`, `tests/test_autoreset.py`

1. Autoreset uses a **boolean mask**, not `nonzero()`: pass `done` into `reset_envs` where
   every write is a `torch.where` over the full batch. Sync-free and compile-safe.
2. `EpisodeStats` tracks `episode_length` and `episode_return` per env, exposing
   `final_episode_length` / `final_episode_return` on done ticks. The SB3 adapter reads these
   to emit SB3's `info["episode"] = {"r": ..., "l": ...}` convention.

**Acceptance:** after a step where env 7 finished, `state.step_count[7] == 0` and
`state.step_count[8]` is unchanged; `info["final_observation"]["hero"]["hp"][7]` reflects the
*old* episode; over 10k steps at `n_envs=64`, `final_episode_length` fires exactly once per
completed episode; no sync warnings in the reset path.

**Implementation notes:**
- Point 1 (boolean-mask autoreset via `torch.where` inside `reset_envs`) was already true as of
  Step 29 -- `env.py`'s `_autoreset` was written that way from the start (`spawn.reset_envs`
  itself is `torch.where`-masked internally, called unconditionally every tick with no `.any()`
  host check first). Nothing needed to change in `env.py`/`spawn.py` for this step; only
  `EpisodeStats` (point 2) was net-new.
- `EpisodeStats` lives under `wrappers/` but does **not** transfer anything to host and does
  **not** do per-env Python loops -- it's a thin torch-native, sync-free, device-resident
  stateful wrapper, same discipline as `env.py` itself. This matters because `wrappers/` is
  also home to `sb3_vecenv.py` (Step 33), which the plan's own architecture section names as
  "the only place host transfers are legal" -- `EpisodeStats` living in the same directory
  doesn't grant it that exception. `BrawlSB3VecEnv` (Step 33) will wrap `EpisodeStats` (not the
  bare `BrawlVecEnv`) and is the thing that eventually turns
  `final_episode_length`/`final_episode_return` into SB3's host-side `info["episode"] = {"r":
  ..., "l": ...}` dicts -- `EpisodeStats` itself never touches numpy or `.item()`.
- `final_episode_length`/`final_episode_return` follow the exact `final_observation`/
  `final_info` shape established in Step 29: **dense** `(N,)` tensors, present every tick for
  every env, only *meaningful* where that tick's `terminated | truncated` was True. For a
  still-running env they're just its in-progress totals, not a sentinel -- callers must gate on
  `done`, same as `final_observation`.
- `EpisodeStats.step` captures its own `length`/`return` *before* zeroing them for done envs,
  mirroring why `env.py`'s `step()` clones `final_observation`/`final_info` before calling
  `_autoreset`: the "old episode's" values must be read out first, then the parallel counters
  reset for the next episode -- same ordering hazard, same fix shape, independently rediscovered
  here rather than copy-pasted, since `EpisodeStats` has no access to `env.py`'s internals to
  reuse the exact same clone.
- Cross-checked `EpisodeStats`' own counter against ground truth rather than trusting it in
  isolation: `env.py`'s `_bookkeeping` (phase 15) increments `state.step_count` before
  `_observe` (phase 16) builds `final_observation`, so `final_observation["meta"]["step_count"]`
  on a done tick is the finished episode's true length, independently of `EpisodeStats`. Both
  `tests/test_autoreset.py` and the CUDA verification script assert
  `info["final_episode_length"][done] == info["final_observation"]["meta"]["step_count"][done]`
  on every done tick over thousands of steps -- this is a much stronger check than testing
  `EpisodeStats` against its own logic restated in the test.
- "Fires exactly once per completed episode" (no double counting, no staleness) is tested by
  tracking, per env, the tick index of every `done` event and asserting the *gap* since the
  previous one equals `final_episode_length` exactly -- this would catch a counter that failed
  to reset (stale length carried into the next episode) or one that reset a tick early/late,
  neither of which the ground-truth cross-check above would necessarily catch on its own.
- CUDA-verified at `n_envs=4096` with `max_episode_steps=40` (short on purpose, to force heavy
  reset churn rather than mostly-continuing envs) under
  `torch.cuda.set_sync_debug_mode("error")`: zero host syncs over 200 steps. Correctness
  (`isfinite(reward)`, the ground-truth cross-check above) was asserted in a **separate** pass
  *outside* the strict sync-debug block -- `assert tensor.all()` itself forces a sync via
  Python's `bool()` coercion, so interleaving correctness asserts inside the strict block would
  produce a false-positive "sync in the reset path" failure that's actually just the test
  harness's own `assert`, not a real bug. (Same pitfall Step 29's CUDA script had to route
  around; documented here again since it's easy to reintroduce per-script.)

---

## Step 31 — `torch.compile` support

**Files:** `brawl_sim/env.py` (extend), `tests/test_compile.py`

Wrap the inner tick with `torch.compile(..., dynamic=False, fullgraph=False)` when
`cfg.compile`. Run with `TORCH_LOGS="graph_breaks"` and document every break. Test that
compiled and eager rollouts match from the same seed (1e-4 float tolerance, exact on
int/bool). Note whether `fullgraph=True` is achievable and what blocks it.

**Note:** Triton on Windows + Blackwell has been fragile. If `torch.compile` doesn't work
here, **that's an acceptable outcome** — record the failure mode in the README and leave the
flag off. Eager is the supported path.

**Acceptance:** `compile: false` unaffected; `compile: true` either matches eager or fails with
a clear documented message without breaking the default path.

**Implementation notes:**
- `env.py`'s `step()` was split: the old inline phase-1-through-15 sequence is now its own
  method, `_run_tick(action, override)` (pure tensor math, no dict-building, no `reward_fn`
  call), returning the same `(dmg_by_total, newly_dead, newly_broken, cubes_gained)` tuple
  `step()` already needed downstream. `_tick_fn` (built once in `__init__` via
  `_make_tick_fn`) is either `_run_tick` itself (`compile: false`, zero overhead, byte-for-byte
  the same code path as Step 29/30) or a `torch.compile`-wrapped version. Phases 16-17
  (`_observe`/`_autoreset`) stay inline in `step()`, untouched -- they build nested python
  `dict`s and phase 16 calls the arbitrary user-supplied `reward_fn`, neither of which
  `torch.compile` should be asked to trace through.
- **`torch.compile` cannot reach a running state on this dev machine, confirmed for BOTH
  backends by hand, not assumed:**
  - **CPU** (`device="cpu"`): inductor's C++ codegen backend needs an MSVC `cl.exe` on PATH.
    Not installed. Fails with `torch._inductor.exc.InductorError: RuntimeError: Compiler: cl
    is not found.` on the very first compiled call.
  - **CUDA** (`device="cuda"`, RTX 5070 Ti / Blackwell / sm_120): inductor's Triton codegen
    backend needs a working Triton install. Not installable here (`pip install triton` has no
    working Windows+Blackwell wheel as of this writing). Fails with
    `torch._inductor.exc.TritonMissing: Cannot find a working triton installation.` This is
    exactly the fragility the plan's own note anticipated.
  - Both are genuine environment/toolchain gaps, not a bug in this codebase -- and both fail
    with a clear, typed, first-call exception, not a hang or silent wrong-output.
- **One real graph break found and documented** (via `TORCH_LOGS="graph_breaks"`, run against
  both backends before either hit its fatal error): `bots/policy.py`'s `aim()` calls
  `torch.randn(shape, generator=gen, device=...)` with a `torch.Generator` object -- dynamo has
  no `as_proxy()` implementation for a bare `Generator` argument (`gb0055` in dynamo's own
  graph-break catalog), so it falls back to eager for that call and continues tracing.
  **This is not necessarily the only break in the tick** -- since neither backend gets past
  compiling the FIRST subgraph (the one ending at this break), any later breaks further into
  `_run_tick` are unobserved on this machine. `fullgraph=True` is very unlikely to be
  achievable while any bot policy still uses `generator=gen` sampling (nearly every archetype's
  aim noise / steering jitter does) without a deeper rewrite (e.g. a custom `torch.library` op
  or switching those call sites to a stateless Philox counter-based RNG); not attempted here
  since `fullgraph=False` is what the plan asks for.
- **`_make_tick_fn`'s failure-wrapping is deliberately narrow.** It only intercepts exceptions
  whose type lives under `torch._dynamo`/`torch._inductor` (checked via `__module__`, not
  message-sniffing) and re-raises them as a `RuntimeError` naming the cause and pointing back
  at `compile: false`. Anything else -- e.g. a real bug in `_run_tick` itself, which
  `fullgraph=False` would surface exactly as it would in eager, just interleaved with compiled
  segments -- propagates with its own original type and message. A masked domain bug reported
  as "torch.compile isn't supported here" would be strictly worse than today's clearly-typed
  compile failure.
- **A real, independent bug was found and fixed while diagnosing this**, unrelated to
  `torch.compile` itself: `step()` cast `action`/`override` to `torch.int64` but never moved
  them onto `self.device`. A CPU-authored action tensor against a CUDA env fails in EAGER mode
  too (`state.act_buf.scatter_(...)` inside `_pop_action_buffer` requires matching devices) --
  torch.compile's fake-tensor device propagation just happened to be the thing that surfaced it
  first, with a much clearer error than eager's own `RuntimeError: Expected all tensors to be
  on the same device...`. Confirmed the eager bug independently (`compile: false`, same
  CPU-tensor-into-CUDA-env repro) before fixing `step()` to `action.to(device=self.device,
  dtype=torch.int64)` (same for `override`). No test before Step 31 exercised a mismatched
  action/env device, so this had never been caught.
- CUDA-verified at `n_envs=4096`: (1) the `_run_tick` extraction + the device-move fix did not
  reintroduce any host sync in the `compile: false` path (`set_sync_debug_mode("error")` over
  100 steps, clean); (2) `compile: true` fails cleanly at this scale too, with the same
  documented `RuntimeError` / `TritonMissing` cause, not a hang or an out-of-memory crash from
  a half-finished compile.
- `tests/test_compile.py` is written to be portable rather than hard-coded to this machine's
  failure: it runs an eager rollout and a `compile: true` rollout from the same seed, and either
  (a) the compiled run raises our documented `RuntimeError` (asserted to name `torch.compile`,
  point at `compile: false`, and chain a `torch._dynamo`/`torch._inductor` cause) -- the actual
  outcome here -- or (b) if compile genuinely succeeds on some other machine, asserts the two
  trajectories match exactly on int/bool and within `1e-4` on float, per the plan's own
  acceptance wording.
- README (Step 42) should carry this same failure mode (MSVC `cl.exe` for CPU, Triton for CUDA
  on Windows+Blackwell) alongside the existing sm_120 install trap, so a future user setting
  `compile: true` on a similar machine isn't surprised.

---

# PHASE J — Gymnasium / Stable-Baselines3 integration

> This phase is where fairness gating, host transfer, and reward all happen. The simulator
> below it stays pure and full-information.

## Step 32 — `core/obs_select.py` and `configs/agent_obs.yaml`

**Files:** `brawl_sim/core/obs_select.py`, `configs/agent_obs.yaml`,
`scripts/dump_obs_schema.py` (extend), `docs/AGENT_OBS.md`, `tests/test_obs_select.py`

**Task:** A declarative spec that turns the full observation package into the **agent's**
observation: a **flat** `gymnasium.spaces.Dict` of `Box` subspaces.

```python
@dataclass(frozen=True)
class AgentObsSpec:
    fair: bool                      # if True, zero out fields the hero cannot legitimately see
    groups: tuple[GroupSpec, ...]   # each becomes ONE key in the SB3 Dict space
    normalize: bool

@dataclass(frozen=True)
class GroupSpec:
    name: str                       # SB3 dict key, e.g. "self", "enemies", "grid"
    fields: tuple[str, ...]         # dotted paths into the full obs
    shape: tuple[int, ...]
    dtype: str                      # "float32" | "uint8"
    per_entity: bool                # True -> (n_enemies, F); False -> (F,)

def load_agent_spec(path, cfg) -> AgentObsSpec
def agent_space(spec, cfg) -> gymnasium.spaces.Dict
def build_agent_obs(full_obs, spec, cfg, out_buffers) -> dict[str, Tensor]
    # GPU tensors, preallocated buffers, allocates nothing per call
def agent_obs_index_map(spec, cfg) -> dict     # {group: {field: (start, end)}}, for docs
```

**Hard constraints (SB3 compatibility):**
1. **The space must be a flat `Dict`.** SB3's `CombinedExtractor` cannot handle nested Dict
   subspaces — it flattens each subspace with `nn.Flatten`, which fails on a dict. Group names
   are top-level keys; no nesting.
2. **`build_agent_obs` must refuse to select anything under `entities.privileged`.** Raise at
   spec-load time, not at runtime. This is the guardrail that makes R04 mean something.
3. Float groups are `Box(low=-inf, high=inf, dtype=float32)`; the grid group is
   `Box(low=0, high=255, shape=(C, VH, VW), dtype=uint8)`.
4. Keep the group count small (≤6). Every group is a separate device→host transfer in Step 33.

**Fairness gating (`fair: true`).** This is where visibility restriction finally happens:
- Per-enemy fields are multiplied by `revealed_to_hero`, and a `revealed` flag is included so
  the policy can distinguish "at origin" from "not seen".
- Projectile fields are gated by `in_view`.
- Grid channels are restricted to the fair set (`enemy_revealed`, never `enemy_any` or
  `enemy_hidden`).
With `fair: false` you get a cheating agent — useful as a difficulty ceiling and for debugging
whether a failure is perception or policy.

`configs/agent_obs.yaml`:
```yaml
fair: true
normalize: true
groups:
  - name: self
    per_entity: false
    dtype: float32
    fields: [hero.pos_norm, hero.vel, hero.facing_vec, hero.hp_frac, hero.ammo_frac,
             hero.ammo_whole, hero.attack_cd, hero.can_attack, hero.dashing,
             hero.dash_t, hero.dash_dir, hero.invuln, hero.in_bush, hero.in_zone,
             hero.cubes, meta.time_frac, meta.n_enemies_alive]
  - name: enemies                       # slots 1..n_enemies, index-stable, gated by fair
    per_entity: true
    dtype: float32
    fields: [entities.alive, entities.revealed_to_hero, entities.kind_onehot,
             entities.rel_pos, entities.dist, entities.rel_vel, entities.hp_frac,
             entities.cubes, entities.can_attack, entities.dashing, entities.in_bush]
  - name: projectiles                   # nearest K by time_to_closest, K = 12
    per_entity: true
    max_slots: 12
    dtype: float32
    fields: [projectiles.alive, projectiles.kind_onehot, projectiles.rel_pos,
             projectiles.vel, projectiles.damage, projectiles.aoe,
             projectiles.time_to_closest, projectiles.threatens_hero]
  - name: zone
    per_entity: false
    dtype: float32
    fields: [zone.hero_margin, zone.active, zone.safe_area_frac, zone.next_shrink_in]
  - name: grid                          # (10, 20, 40) uint8 egocentric, fair channels only
    dtype: uint8
    view_channels: [blocks_unit, blocks_projectile, is_bush, is_water, in_zone,
                    enemy_revealed, hero, box, pickup, projectile]
```

The projectile group is the one place sorting is allowed, since `P = 128` slots is too wide
for a policy input: take the `max_slots` smallest `time_to_closest` among live projectiles via
`topk`, zero-pad the rest. Document that this group is **not** index-stable, unlike everything
else.

Extend `dump_obs_schema.py` to also emit `docs/AGENT_OBS.md`: every group, its shape, and the
exact field order inside it, generated from the spec.

**Acceptance:** `agent_space(spec, cfg)` is a flat `spaces.Dict` and
`gymnasium.spaces.utils.flatdim` succeeds on it; `build_agent_obs` output shapes match the
space exactly and every value is finite; a spec referencing `entities.privileged.target_id`
raises at load time with a clear message; with `fair: true`, a bush-hidden enemy contributes
all-zero per-enemy fields with `revealed = 0` while the full obs still has its true position;
`build_agent_obs` allocates nothing per call; `docs/AGENT_OBS.md` regenerates deterministically.

**Implementation notes:**
- **Shapes are computed, never authored.** `configs/agent_obs.yaml` states `fields`/
  `per_entity`/`max_slots`/`view_channels` only; `load_agent_spec(path, cfg)` resolves each
  `GroupSpec.shape` from `obs_schema.obs_spec(cfg)`'s already-resolved field shapes. This keeps
  the yaml honest -- there's no way for a hand-written shape to drift from the real field
  widths, the same "compute, don't restate" discipline `obs_schema.obs_spec` itself uses for
  symbolic dims.
- **Per-entity axis handling is generic, not hardcoded to "projectiles"/"entities" by name.**
  `_entity_prefix(fields)` requires every field in a `per_entity: true` group to share one
  dotted prefix (raises otherwise) and uses THAT to decide: (a) whether to drop hero slot 0
  (only the `"entities"` prefix has one), and (b) which field gates fairness (`_FAIRNESS_MASK_
  FIELD = {"entities": "...revealed_to_hero", "projectiles": "...in_view"}` -- any other prefix
  a spec author adds is silently ungated under `fair: true`, since neither Step 32's plan text
  nor the schema define what "fair" means for a `boxes.*`/`pickups.*` axis; extend the table
  when a concrete need shows up). Likewise `max_slots` (top-K by `<prefix>.time_to_closest`,
  zero-padded via the gathered `<prefix>.alive` mask) works for any per-entity group whose
  prefix has that field, not only the one group (`projectiles`) that uses it today.
- **Fairness gating turned out to need no field-level special-casing.** Multiplying an entire
  per-enemy row by its own `revealed_to_hero` value (broadcast across every output column)
  automatically produces the exact documented behavior for the `revealed_to_hero` *column
  itself* too: `revealed * revealed == revealed`, so no separate "except this one field" branch
  was needed to keep that column truthful while zeroing everything else.
- **`normalize: true` is real but intentionally narrow** (the plan's own text doesn't specify a
  concrete transform, and no acceptance criterion exercises it): fields whose `obs_schema`
  `units == "tiles"` are divided by `max(cfg.map_w, cfg.map_h)`; `units == "tiles/s"` fields are
  divided by a fixed `_NORM_SPEED_SCALE = 20.0`, grounded in `configs/brawlers.yaml`'s actual
  ranges (dash speed ~16.7 tiles/s, fastest projectile 14 tiles/s). Every other unit (seconds/
  hp/count/fraction/bool/onehot/unitless) passes through unnormalized -- most already have a
  bounded `_frac`/`_whole`/`onehot` alternative available in the schema for a spec author to
  pick instead of the raw field. Grid (`uint8`) groups are never normalized (Hard Constraint 3
  fixes their space at `[0, 255]` regardless of `normalize`). Forward pointer: extend
  `_NORM_DIVISOR_TILES`/`_NORM_SPEED_SCALE`/the units table if a concrete training run needs
  `"hp"` or `"seconds"` fields normalized too.
- **Both the grid-channel index and the per-group normalize-scale vector are resolved to plain
  Python tuples at spec-LOAD time** (`cfg` is available there) and only turned into device
  tensors lazily, memoized by `(key, device)` in a small module-level cache
  (`_cached_tensor`) -- the exact same shape as `core/reward.py`'s `ZeroReward` cache and
  `core/geometry.vec2`'s host-sync avoidance (Step 29). Building the index/scale tensor via
  `torch.tensor(python_list, device=...)` fresh every `build_agent_obs` call would be a real,
  avoidable host sync on the hot path; caching means it happens at most once per (spec, device)
  pair, not once per tick. CUDA-verified sync-free at `n_envs=4096` under
  `set_sync_debug_mode("error")` over 100 steps, after a warmup pass to populate the cache.
- `build_agent_obs` "allocates nothing per call" follows the same established reading as
  `observation.build_obs` (Step 25): steady-state resident memory doesn't grow (verified via
  `torch.cuda.memory_allocated()` delta across 100 calls at `n_envs=4096`, not literal
  zero-temporaries) -- `out_buffers[name].copy_(...)` writes into caller-owned, reused storage,
  while intermediate concatenation/gather/topk temporaries are allowed, same as every other
  hot-path module in this codebase.
- `docs/AGENT_OBS.md` is generated against the plain `configs/default.yaml` (no overrides),
  matching `docs/OBSERVATION.md`'s own "one reference cfg" convention -- `scripts/
  dump_obs_schema.py` now writes both files in one run.

---

## Step 33 — `wrappers/sb3_vecenv.py`

**Files:** `brawl_sim/wrappers/sb3_vecenv.py`, `tests/test_sb3_vecenv.py`

**Task:** Implement `stable_baselines3.common.vec_env.VecEnv` directly. This file is the
**only** place in the repo where host transfers and per-env Python loops are permitted; say so
in a header comment.

```python
class BrawlSB3VecEnv(VecEnv):
    def __init__(self, env: BrawlVecEnv, spec: AgentObsSpec,
                 reward_fn: RewardFn,
                 info_mode: str = "minimal",     # "minimal" | "episode" | "full"
                 info_every: int = 1):
        self.num_envs = env.n_envs
        self.observation_space = agent_space(spec, env.cfg)
        self.action_space = gymnasium.spaces.MultiDiscrete([17, 2])
        super().__init__(self.num_envs, self.observation_space, self.action_space)
```

Required `VecEnv` surface:
`reset() -> obs`, `step_async(actions)`, `step_wait() -> (obs, rewards, dones, infos)`,
`close()`, `get_attr`, `set_attr`, `env_method`, `env_is_wrapped`, `render`, `seed`.
For the last four, return sensible batch-wide answers or raise `NotImplementedError` with a
message explaining this is a single fused vector env, not `n` independent envs.

**SB3 contract details that are easy to get wrong — get all five right:**
1. **`dones` is a single boolean array**, `terminated | truncated`. SB3 has no separate
   truncated return.
2. **Bootstrapping flag:** `infos[i]["TimeLimit.truncated"] = bool(truncated[i] and not
   terminated[i])`. Without this, SB3 will treat timeouts as terminal states and your value
   function will be wrong.
3. **Terminal observation:** on done, `infos[i]["terminal_observation"]` must hold the agent
   observation of the *finished* episode — build it from `info["final_observation"]`, not from
   the post-reset obs.
4. **Episode stats:** on done, `infos[i]["episode"] = {"r": float, "l": int}` from
   `EpisodeStats`, so `VecMonitor` and the SB3 logger work.
5. **`reset()` returns observations only**, no info.

**Reward.** `step_wait` calls `reward_fn(full_obs, info, cfg) -> (N,) float32` on the **full**
observation package before transferring, then moves the result to host. This is the seam the
whole design points at: full information in, scalar reward out, simulator untouched.

**Action masking.** Expose
```python
def action_masks(self) -> np.ndarray     # (num_envs, 19) bool
```
concatenating `[move_mask (17), attack_mask (2)]` in `MultiDiscrete` order, which is what
`sb3_contrib.MaskablePPO` expects. **MaskablePPO is the recommended algorithm here** — you
already have a correct mask and it removes a whole class of wasted exploration.

**Host transfer discipline** (this determines your throughput):
- One `.to("cpu", non_blocking=True)` per observation group, then `.numpy()`. Keep the group
  count small.
- Reuse pinned host buffers allocated once in `__init__`; do not allocate numpy arrays per step.
- `infos` modes:
  - `"minimal"` (default): a list of `num_envs` empty dicts, reused, with entries added only
    for done envs. Cheapest and sufficient for training.
  - `"episode"`: adds `episode` stats on done.
  - `"full"`: adds per-env logging tensors every `info_every` steps. **Expect this to dominate
    step time at large `n_envs`** — use it for diagnostics, not for training runs.
- Never transfer `obs["world"]` or `obs["visibility"]` to the host in `"minimal"` mode.

**Acceptance:** `stable_baselines3.common.vec_env.VecEnv` subclass check passes;
`terminal_observation` matches the pre-reset agent obs for done envs and is absent otherwise;
`TimeLimit.truncated` is True exactly on timeout-without-death; `action_masks()` matches
`obs["action_mask"]` bit for bit; a 1000-step loop in `"minimal"` mode performs a bounded,
counted number of device→host transfers per step (assert the count); switching `info_mode` does
not change trajectories.

**Implementation notes:**
- **Reward ownership was a real design tension, not just an implementation detail.** Read
  literally, "`step_wait` calls `reward_fn(full_obs, info, cfg)`" would mean calling it a SECOND
  time, separately from `BrawlVecEnv.step()`'s own existing `self.reward_fn` call at phase 16
  (`core/reward.py`'s documented call site, Step 28). Doing that naively either double-computes
  reward for no reason, or -- worse -- if the wrapped `env` keeps its default `ZeroReward`, makes
  `EpisodeStats`' accumulated `final_episode_return` (which reads `env.step()`'s OWN first
  return value) silently diverge from the reward SB3 actually sees, so `infos[i]["episode"]["r"]`
  would always read 0. Resolved by having `BrawlSB3VecEnv.__init__` overwrite `env.reward_fn`
  with the `reward_fn` passed to it -- one call site, correct on both sides, and matches `core/
  reward.py`'s own stated intent ("anything real gets computed at the SB3 adapter boundary").
  Documented prominently since it means constructing `env` with a different `reward_fn` and
  expecting it to survive is a real, easy-to-hit footgun.
- **Three real bugs were found by actually running `sb3_contrib.MaskablePPO.learn()` against
  this wrapper** (a genuine integration smoke test, beyond what the written acceptance criteria
  above exercise) -- none were visible from the plan text or from the acceptance list alone:
  1. `self.spec = spec` (the `AgentObsSpec`) collided with `gymnasium`/SB3's OWN reserved
     `.spec` attribute (the registration `EnvSpec`). `VecMonitor.__init__` reads `venv.spec.id`
     when `spec` is present and non-`None`; ours doesn't have `.id`, so wrapping in `VecMonitor`
     (the plan's own Step 42 usage example!) crashed immediately. Renamed to `self.agent_spec`.
  2. `get_attr`/`set_attr` only ever looked at the wrapped `BrawlVecEnv`, never at `self`.
     `sb3_contrib.MaskablePPO` detects masking support via `VecEnv.has_attr("action_masks")`,
     which calls `get_attr` and checks for `AttributeError` -- since `action_masks` is a method
     on `BrawlSB3VecEnv` itself, not on `BrawlVecEnv`, this always failed, and MaskablePPO
     refused to train ("Environment does not support action masking"). Fixed by checking `self`
     first, falling back to the underlying env -- which in turn required making `reward_fn` a
     property proxying straight to `self.env.env.reward_fn` (rather than a second, inert copy
     on the wrapper), so `get_attr`/`set_attr("reward_fn", ...)` can't desync from the value
     `step_wait` actually uses.
  3. `env_method` unconditionally raised `NotImplementedError`. `sb3_contrib`'s
     `get_action_masks(env)` calls `env.env_method("action_masks")` (not `env.action_masks()`
     directly) and `np.stack()`s the result -- the "one result per sub-env" shape a real
     multi-process `VecEnv` would produce. Fixed by special-casing exactly this one method name
     (`list(self.action_masks())`, honoring `indices`); every other method name still correctly
     raises, since there is genuinely no single sub-env to call it on. This makes
     `env_method("action_masks")` an unavoidable per-`n_envs` Python loop on every rollout step
     under `MaskablePPO` specifically -- a real, load-bearing cost this file's own "per-env
     Python loops are permitted here" header comment already anticipated in spirit, just not by
     name until this was found.
  A full `MaskablePPO.learn()` + `.predict()` round trip (via a grid-free `AgentObsSpec`, see
  next note) passes after all three fixes; a regression test for each is in
  `tests/test_sb3_vecenv.py`.
- **The default `configs/agent_obs.yaml` (with its `grid` group) cannot train under
  `MaskablePPO`'s default `MultiInputPolicy` today -- confirmed by hand, and this is EXACTLY
  the gap Step 35 (`wrappers/sb3_features.py`) already exists to close**, not a Step 33 bug:
  SB3's `CombinedExtractor` routes any 3D `uint8` subspace to `NatureCNN`, whose fixed
  8/4-then-4/2-then-3/1 kernel/stride stack can't run on `debug_tiny`'s `(10, 10, 14)` grid (and
  wouldn't work on the default `(10, 20, 40)` grid either, per Step 35's own text). The
  integration smoke test above used a grid-free spec specifically to isolate this wrapper's own
  correctness from Step 35's not-yet-built feature extractor; re-confirmed separately that the
  full default spec fails with exactly this one, already-anticipated error and nothing else.
- Host-transfer discipline: `reset()`/`step_wait()` each issue one `non_blocking=True` `.copy_`
  per agent-obs group into a buffer pinned once in `__init__` (`_pinned = device.type ==
  "cuda"`), plus reward/terminated/truncated, batched behind a SINGLE `_sync()` call rather than
  one sync per tensor -- the whole point of `non_blocking`. `terminal_observation` needs a
  SEPARATE buffer set (`_terminal_device_buffers`/`_terminal_host_buffers`): it's built from
  `info["final_observation"]`, a genuinely different tensor than the regular `obs` `step_wait`
  already transferred that same call, and reusing the same persistent buffers would clobber it.
  That second pass only runs on ticks with `>= 1` done env (checked via the already-transferred
  `dones` array), so it doesn't add cost to the common case.
- `info_mode` gating, as actually implemented: `"minimal"` always includes
  `terminal_observation`/`TimeLimit.truncated` on done (these are SB3 correctness requirements,
  not optional diagnostics); `"episode"` additionally adds `episode: {r, l}` on done; `"full"`
  additionally adds a small, deliberately-not-schema-fixed per-env diagnostic dict
  (`hero_hp_frac`/`hero_pos`/`hero_alive`/`step_count`/`n_alive`) for every env every
  `info_every`-th tick. This reading isn't 100% forced by the plan's prose (which lists all 5
  "get them right" constraints together); it's the interpretation that makes `"minimal"`
  actually minimal while keeping every SB3-correctness-critical field unconditional.

---

## Step 34 — `wrappers/gym_single.py`

**Files:** `brawl_sim/wrappers/gym_single.py`, `tests/test_gym_single.py`

**Task:** A standard single-environment `gymnasium.Env` wrapping `BrawlVecEnv(n_envs=1)`, for
validation and debugging. Standard gymnasium 5-tuple: `reset(seed, options) -> (obs, info)`,
`step(action) -> (obs, reward, terminated, truncated, info)`. **No autoreset** — gymnasium
single envs don't autoreset, so this wrapper must disable the vec autoreset path and let the
caller reset.

Its main job is to pass `stable_baselines3.common.env_checker.check_env`, which catches
space/dtype/bounds mismatches far more cheaply than a failed training run.

**Acceptance:** `check_env(BrawlGymEnv(...), warn=True)` passes with no errors; a 300-step
random episode terminates correctly; obs dtypes exactly match the declared space.

**Implementation notes:**
- **"Disable the vec autoreset path" required a small, additive change to `env.py` itself**,
  not something achievable purely inside the wrapper: `BrawlVecEnv.__init__` gained an
  `autoreset: bool = True` parameter (default preserves every Step 29/30/31/32/33 behavior and
  test byte-for-byte -- confirmed by rerunning the full suite unchanged before adding anything
  new). With `autoreset=False`, `step()`'s phase-17 `_autoreset` call is skipped entirely:
  `state` is left exactly as phase 16 finished it, and the returned `obs` is that same
  (possibly terminal) tick's own observation -- not a fresh episode's first one, and not a
  second `_build_observation()` call (phase 16's `obs_before_reset` is reused directly instead
  of being cloned-then-deleted-then-rebuilt, which is actually cheaper than the
  `autoreset=True` path for this specific case). `info["final_observation"]`/`info[
  "final_info"]` are still populated for API-shape consistency across both modes, and under
  `autoreset=False` are simply equal to `obs`/`info` rather than a distinct pre-reset snapshot.
- **`BrawlGymEnv` forces `env.autoreset = False` and `env.reward_fn = reward_fn` on the
  `BrawlVecEnv` it's given, in `__init__`** -- the exact same "wrapper takes ownership of an
  attribute on construction" pattern `BrawlSB3VecEnv` already established for `reward_fn`
  (Step 33), applied a second time for a second attribute. This means the caller cannot
  accidentally construct a `BrawlGymEnv` that silently autoresets or ignores its `reward_fn`;
  both would otherwise be easy, hard-to-notice misconfigurations for a "debugging and
  validation" tool whose whole point is catching mistakes cheaply.
- **Reuses `core/obs_select.py` (Step 32) and the same host-transfer pattern as `wrappers/
  sb3_vecenv.py` (Step 33)**, at `n_envs=1` instead of batched -- `AgentObsSpec`/`agent_space`/
  `build_agent_obs` are shared verbatim, so `BrawlGymEnv` and `BrawlSB3VecEnv` are guaranteed to
  agree on what "the agent's observation" means; a spec that passes fairness/privileged-field
  validation for one automatically passes it for the other. `wrappers/sb3_vecenv.py`'s own
  docstring originally claimed to be "the ONLY place in the repo" host transfers are permitted
  -- corrected in place (not left stale) now that `gym_single.py` is a second, equally
  legitimate one, for the identical reason: `gymnasium.Env`'s contract also requires plain
  Python/numpy, not torch tensors.
- **A caller stepping again after `terminated`/`truncated` without calling `reset()` first
  raises `RuntimeError`**, rather than silently continuing to simulate a frozen/terminal state.
  This isn't required by `gymnasium`'s contract (many envs just let it happen) or tested by
  `check_env`, but matches this file's own stated purpose ("validation and debugging") -- a
  debugging tool should fail loudly on caller misuse, not produce quietly-meaningless
  trajectories that look valid but aren't.
- `check_env(BrawlGymEnv(...), warn=True)` passes cleanly against the real, full `configs/
  agent_obs.yaml` (including the `grid` group) on the first attempt, with exactly the two
  `UserWarning`s `warn=True` is supposed to surface: "unconventional shape" for the
  non-image/non-vector `enemies`/`projectiles` groups, and "minimal resolution... 36x36" for
  the grid -- both already-known, already-documented (Step 32/35's own text) characteristics of
  this observation, not new problems.
- CUDA-verified: `autoreset=False` is sync-free at `n_envs=4096` under
  `set_sync_debug_mode("error")` (the underlying `env.py` change, exercised at the scale that
  matters even though `gym_single.py` itself is single-env-only); a 300-step `BrawlGymEnv`
  rollout on a CUDA single env produces finite obs throughout (host transfers here are
  inherent/deliberate, not a sync-freedom target, per this file's own docstring).

---

## Step 35 — `wrappers/sb3_features.py`

**Files:** `brawl_sim/wrappers/sb3_features.py`, `tests/test_sb3_features.py`

**Task:** The minimum model-side code needed for SB3 to *accept* this observation. Not a
tuned architecture — a working default you will replace.

**Why this is necessary (concrete gotcha).** SB3's `CombinedExtractor` routes any 3D `uint8`
subspace to `NatureCNN`, whose kernel/stride stack (8/4, then 4/2, then 3/1) collapses a
20x40 input to an invalid size and raises. Your grid is 10x20x40, so the default path
**cannot** work.

Provide:
```python
class BrawlFeaturesExtractor(BaseFeaturesExtractor):
    """Small CNN sized for (C, view_h, view_w) + MLP for the vector groups, concatenated.
    Grid conv stack: 3x3 s1 -> 3x3 s2 -> 3x3 s2 -> flatten, ~32-64 channels.
    Vector groups: flatten -> Linear -> ReLU."""

def default_policy_kwargs(spec, cfg) -> dict
    # {"features_extractor_class": BrawlFeaturesExtractor,
    #  "features_extractor_kwargs": {...},
    #  "normalize_images": False}
```
**`normalize_images=False` is required** — SB3 otherwise divides `uint8` observations by 255,
and your grid channels are small occupancy counts, not 0–255 pixel intensities.

Also document in the module docstring: if you use `VecNormalize`, pass
`norm_obs_keys=[<float group names>]` to exclude the `uint8` grid, which must not be
normalized as a running statistic.

**Acceptance:** a forward pass through `BrawlFeaturesExtractor` on a real observation batch
produces the declared `features_dim` with no shape errors; `default_policy_kwargs` plugs into
`PPO("MultiInputPolicy", ...)` without modification; the module imports without
`stable_baselines3` installed only if guarded — otherwise it may hard-require it.

**Implementation notes:**
- **This closes the exact gap Step 33's own integration smoke test hit and documented**: a real
  `MaskablePPO.learn()` run against the FULL default `configs/agent_obs.yaml` (grid group
  included) previously failed with `NatureCNN`'s "Kernel size can't be greater than actual
  input size" on `debug_tiny`'s `(10, 14)` grid. Re-ran that exact same integration test with
  `policy_kwargs=default_policy_kwargs(spec, cfg)` -- it now completes end to end (`.learn()` +
  `.predict()`), with both `MaskablePPO` and plain `PPO`, on CPU and CUDA. This is the concrete
  evidence the plan's "why this is necessary" gotcha was real and is now actually fixed, not
  just theoretically addressed.
- **Padding is the load-bearing difference from `NatureCNN`, not just a smaller kernel.**
  `NatureCNN`'s 8/4-then-4/2-then-3/1 stack has NO padding and dies on small inputs by
  construction. This extractor's 3x3-stride-1/3x3-stride-2/3x3-stride-2 stack uses `padding=1`
  throughout, which keeps the spatial size positive even at `debug_tiny`'s `(10, 14)` (verified
  down to `(3, 4)` after the third conv) -- worth calling out since a smaller kernel ALONE,
  without padding, would still have broken on a small enough grid.
- **Generalized beyond the default 5-group spec on purpose**: `BrawlFeaturesExtractor` doesn't
  hardcode group names -- it classifies every `observation_space` key as "the one `uint8` group"
  (routed through the CNN) or "a float32 group" (flattened, concatenated, routed through the
  MLP), and correctly handles zero-or-one grid groups and zero-or-more vector groups (raising
  only on more than one `uint8` group, or exactly zero groups of any kind). This matters because
  `core/obs_select.py` (Step 32) is itself spec-driven and not hardcoded to the shipped
  `configs/agent_obs.yaml` -- a differently-shaped `AgentObsSpec` (e.g. `fair: false` with the
  grid group dropped for a pure-vector run) should still work with this extractor unmodified,
  and does (covered by `tests/test_sb3_features.py`).
- **`default_policy_kwargs(spec, cfg)` eagerly constructs a throwaway `BrawlFeaturesExtractor`**
  against `obs_select.agent_space(spec, cfg)` before returning, specifically so a
  shape-incompatible spec (two `uint8` groups, or an empty one) fails immediately with a clear
  message at policy-kwargs-construction time, rather than lazily and more confusingly inside
  SB3's own `MultiInputActorCriticPolicy.__init__` much later.
- **`normalize_images=False` cannot be enforced by `BrawlFeaturesExtractor` itself** -- it's
  consumed one level up, by the POLICY's own `preprocess_obs` call, before the extractor ever
  sees the data. `default_policy_kwargs` returns it as a top-level dict key (not nested under
  `features_extractor_kwargs`) specifically so it lands on the POLICY constructor when spread
  into `PPO(..., policy_kwargs=...)`. Confirmed by hand that omitting it silently corrupts the
  grid: `preprocess_obs` on a raw occupancy count of 200 returns `200.0` with
  `normalize_images=False` and `~0.78` (`200/255`) with the SB3 default (`True`) --
  regression-tested.
- **Chose to hard-require `stable_baselines3`** (no `try`/`except ImportError` guard), matching
  `wrappers/sb3_vecenv.py`'s (Step 33) existing choice, rather than introducing a second import
  convention within the same `wrappers/` package. Both are covered by the `sb3` optional
  dependency group in `pyproject.toml`; nothing under `core/`/`bots/`/`env.py` imports this
  module.
- Added `float_group_names(spec) -> list[str]` (not in the plan's own pseudocode, but directly
  serves the plan's own "document in the module docstring: pass `norm_obs_keys=[<float group
  names>]` to `VecNormalize`" instruction) -- computing that list by hand means re-deriving
  which groups are `float32` vs `uint8` from the spec every time; this makes the documented
  usage copy-pasteable (`VecNormalize(venv, norm_obs_keys=float_group_names(spec))`) instead of
  something the docstring describes but the caller has to re-implement.

---

## Step 36 — `scripts/sb3_smoke.py`

**Files:** `scripts/sb3_smoke.py`, `tests/test_sb3_integration.py`

**Task:** End-to-end plumbing validation. **This is not a training run** — it proves the
interfaces connect.

```
1. check_env(BrawlGymEnv(cfg))                                  -> no errors
2. venv = BrawlSB3VecEnv(BrawlVecEnv(cfg, n_envs=8), spec, ExampleReward())
3. venv = VecMonitor(venv)
4. model = MaskablePPO("MultiInputPolicy", venv,
                       policy_kwargs=default_policy_kwargs(spec, cfg),
                       n_steps=64, batch_size=128, verbose=1)
5. model.learn(total_timesteps=2048)                            -> completes
6. model.predict(obs, action_masks=venv.action_masks())         -> valid actions
7. model.save / load round-trip                                 -> identical predictions
8. repeat 4-5 with plain PPO (no masking) to confirm both paths work
```
Print observed steps/sec so the SB3 boundary cost is visible from step one.

**Acceptance:** the script runs start to finish on CPU and GPU; `learn` completes without
shape, dtype, or space errors; the mask path and the unmasked path both work; no step of the
script requires editing anything under `brawl_sim/core/`.

**Implementation notes:**
- Structured as a reusable `run_smoke(cfg, agent_spec, device, n_envs, n_steps, batch_size,
  total_timesteps, ...)` function plus a thin `main()` CLI, rather than a flat top-to-bottom
  script -- `tests/test_sb3_integration.py` calls `run_smoke` directly with small overrides
  (`n_envs=4`, `total_timesteps=128`) so the pytest run takes ~10s instead of the plan's own
  `n_envs=8`/`total_timesteps=2048` defaults (~8-9s alone on CPU, longer on this machine's GPU
  since CUDA per-step overhead dominates at these tiny batch sizes -- see the steps/sec numbers
  below). `main()` uses the plan's literal defaults and is what a human runs directly.
- `venv.action_masks()` (step 6) works through `VecMonitor` unmodified -- `VecEnvWrapper.
  __getattr__` proxies method calls, not just attributes, down to the wrapped `BrawlSB3VecEnv`.
  Confirmed by hand before relying on it (worth confirming explicitly: SB3's own `VecEnvWrapper`
  attribute-proxying conventions aren't `__getattr__` in every wrapper library).
- Verified with real numbers, not just "it completes": CPU, `n_envs=8` (plan defaults):
  `MaskablePPO` ~244 steps/sec, plain `PPO` ~250 steps/sec. CUDA, same `n_envs=8` but
  `total_timesteps=512` (small-batch CUDA overhead dominates at this scale): `MaskablePPO` ~83
  steps/sec, plain `PPO` ~86 steps/sec -- CPU is faster than GPU here, expected and fine: `n_envs
  =8` is far below the batch size needed to amortize CUDA kernel-launch/host-transfer overhead;
  this script exists to prove the plumbing connects, not to benchmark throughput (that's what
  n_envs=4096-scale profiling, not built yet, is for).
- Also ran the FULL default (non-`debug_tiny`) config end to end on CUDA -- the real `60x60` map
  / `(10, 20, 40)` grid, not just the small `debug_tiny` shapes exercised by the automated
  tests -- to confirm nothing in this pipeline is accidentally tuned to the tiny test config.
  Passed cleanly; not wired into the automated test suite (too slow to run on every `pytest`
  invocation) but confirmed by hand for this report.
- `save`/`load` round trip (step 7) loads with `env=venv` (the SAME venv instance, not a fresh
  one) and asserts the reloaded model's `predict()` output is `np.array_equal` to the
  pre-save one from the identical `obs`/`action_masks` -- a real bit-for-bit determinism check,
  not just "loading doesn't raise."
- Deliberately did not touch anything under `brawl_sim/core/` for this step (confirmed by
  recollection of the session's own edits, matching the acceptance criterion) -- every piece
  this script exercises (`BrawlGymEnv`, `BrawlSB3VecEnv`, `BrawlFeaturesExtractor`/
  `default_policy_kwargs`) already existed from Steps 32-35.

---

# PHASE K — Tooling & validation

## Step 37 — `render/ascii.py` + `scripts/record_rollout.py`

**Files:** `brawl_sim/render/ascii.py`, `scripts/record_rollout.py`

`render_ascii(snapshot, bank, cfg, mode="world"|"view"|"agent") -> str`:
`#` wall, `,` bush, `~` water, `=` fence, `H` hero, `s`/`a`/`m`/`r` bots (uppercase when
revealed to the hero, lowercase when hidden), `o` box, `*` pickup, `'` projectile, `:` zone,
`.` floor. Status line: `t`, hero HP/max, ammo pips, cubes, `n_alive`, zone step, dash state.
`mode="agent"` renders exactly what `AgentObsSpec(fair=True)` exposes — hidden enemies simply
absent — so you can see the fair view and the true view side by side.

`scripts/record_rollout.py --config ... --steps 600 --out rollout.npz`: `n_envs=1`, CPU
snapshot per tick, stacked arrays saved.

**Acceptance:** frame 0 shows a sane map with `1 + n_enemies` units; `view` mode outputs
exactly `view_h` lines of `view_w` characters; `agent` mode never shows a bush-hidden bot.

**Implementation notes:**
- **A real, previously-undiscovered bug was found in `core/state.py`'s `snapshot()` (Step 9,
  long-shipped) while writing `record_rollout.py`.** On a CPU-device `SimState`, `.cpu()` is a
  no-op (already CPU) and `.numpy()` shares memory with the live tensor -- `snapshot()` was
  returning numpy VIEWS, not copies, on CPU (it only happened to be safe on CUDA, where `.cpu()`
  itself forces a real device-to-host copy). Every earlier caller only ever held one snapshot at
  a time, so this never surfaced; `record_rollout.py` is the first caller to hold `steps` worth
  of snapshots alive simultaneously (to `np.stack` them all at the end) across intervening
  `step()` calls, on CPU by default -- every held snapshot silently aliased to whatever the
  LAST `step()` call left behind (concretely: `frame[0]["step_count"]` read `4` instead of `0`
  after a 5-frame recording with 4 steps in between). Fixed with one added `.copy()` after
  `.numpy()` in `snapshot()`; regression-tested in both `tests/test_state.py` (holding two
  snapshots across a mutation) and `tests/test_record_rollout.py` (frame 0 really is the
  post-reset state). This is the same "zero-copy view valid only until the next call" footgun
  Steps 25/27/29 documented for `build_obs`/`final_observation` -- `snapshot()`'s own docstring
  didn't carry the same warning before this fix, which is exactly why it went unnoticed for so
  long despite being a known *class* of bug in this codebase.
- **`render_ascii`'s signature stays exactly `(snapshot, bank, cfg, mode)`, per the plan** --
  `revealed_to_hero` (a `bots/perception.visibility` OUTPUT) and `max_ammo` (a `SimParams`
  field) are both needed by the char legend / `mode="agent"` but aren't part of raw `SimState`.
  Rather than add parameters or import `bots/` into `render/` (a boundary this package never
  had before), both are OPTIONAL EXTRA KEYS a caller adds to the plain `state.snapshot()` dict;
  `record_rollout.py` computes and adds them every frame. A bare `state.snapshot()` dict still
  renders in `"world"`/`"view"` mode (bots read as revealed, ammo falls back to a plain number);
  `mode="agent"` raises a clear `ValueError` naming the missing key.
- **A second real bug, found while writing this module's own tests**: `prio`'s "out-of-map"
  sentinel was initially set to the OBSTACLE priority, which is HIGHER than floor's -- meaning
  ordinary floor tiles (priority 0) could never overwrite it, and every in-map floor cell
  rendered as `#` instead of `.`. The fix is a sentinel strictly BELOW every real priority
  (`_P_FLOOR - 1`), relying on the terrain-paint loop's own bounds (not the sentinel's value) to
  keep genuinely out-of-map cells walled. Caught immediately by eyeballing a rendered frame
  (an entirely wall-covered "blank" map is obviously wrong) before it reached a test.
- **`mode="agent"` is a deliberately scoped-down reading of "exactly what `AgentObsSpec(fair=
  True)` exposes"**: it reuses `"view"`'s egocentric crop and omits bush-hidden bots (the one
  behavior the acceptance criterion actually tests), but does not reproduce `AgentObsSpec`'s
  finer-grained per-field gating (e.g. `in_view` for projectiles) -- doing that exactly would
  mean rebuilding the full `obs_select` pipeline (needs live GPU tensors) inside a numpy-only
  debug renderer. Documented as a forward pointer, not silently glossed over.
- **"Revealed" and "inside the crop window" are different gates**, also found while testing:
  a bot can be `revealed_to_hero` (bush/targeting visibility) while standing outside the
  `view_h x view_w` crop in `"view"`/`"agent"` mode -- it still won't be drawn, matching the real
  `obs["view"]` grid's own independent spatial cropping. An early version of this step's test
  suite assumed "revealed implies visible in agent mode" and failed until the test was fixed to
  place the bot within crop range.
- Overlay z-order (documented in the module docstring, not literally specified by the plan
  text): hero > bot > box > projectile > pickup > obstacle terrain > zone marker > floor. Zone
  only ever replaces floor, never an obstacle -- walls don't move because the zone shrank.
- No test file was named in this step's own `Files:` line (only the two source files were);
  written anyway (`tests/test_ascii.py`, `tests/test_record_rollout.py`) to keep the same
  automated-coverage bar every prior step has had, and because the acceptance criteria as
  written clearly call for it.
- CUDA-verified: `record_rollout` + `render_ascii` produce the same correct frame-0/frame-`t`
  progression on a CUDA-device env as on CPU (the snapshot-copy bug was CPU-only by nature, but
  worth confirming the fix didn't regress the CUDA path, which was already correct).

---

## Step 38 — `render/viewer.py`

**Files:** `brawl_sim/render/viewer.py`

**Matplotlib** replay viewer for `rollout.npz` (see Notice 3 — swapped from pygame, same
controls and acceptance bar). Single `Figure`/`Axes`, `origin="upper"` to match the CSV
row-major / y-down convention.

- Tiles: `ax.imshow` of the tile-id grid through a fixed `ListedColormap` keyed by `Tile`.
- Entities: one `ax.scatter` per kind (color by kind, consistent with the ASCII legend), sized
  by `unit_radius`; a small `Rectangle` HP bar above each; a short `Line2D` segment for facing;
  ammo pips as a tiny text annotation; dashing entities get a bright `edgecolor` ring.
- Projectiles: one `ax.scatter` per `Proj` kind, colored by type.
- Boxes: `ax.scatter` with `marker="s"`. Pickups: `marker="D"`.
- Unsafe zone: a semi-transparent red `Rectangle` (or four rectangles framing the safe rect)
  redrawn from `zone_lo`/`zone_hi` each frame.
- View bounds: an unfilled `Rectangle`, `view_h x view_w` centered on the hero, toggled by `c`.
- Reveal overlay: a hatch/alpha applied to entities where `raw_los` is True but `vis` is False
  (bush-hidden from the hero), toggled by `v`.

**Performance:** update artist data in place (`set_offsets`, `set_data`, `set_xy`) inside a
`FuncAnimation` callback with `blit=True` — do not call `ax.clear()`/re-plot every frame, that
will not hit 20 fps. Bind `key_press_event` for: space pause, left/right arrows scrub, `+`/`-`
speed, `v` reveal overlay, `c` view rect, `g` world/view toggle.

**Acceptance:** plays at 20 fps with blitting on this machine's GPU+CPU; matches the ASCII
renderer frame for frame; the reveal overlay correctly marks bush-hidden bots.

**Implementation notes (filled in during Step 38):**
- `render/ascii.py`'s status-line formatter was renamed `_status_line` → `status_line` (dropped
  the leading underscore, no behavior change) and is now imported directly by
  `render/viewer.py` for the animation title, instead of a second hand-written copy. This is
  what makes "matches the ASCII renderer frame for frame" true by construction for the HP/ammo/
  cubes/dash text, rather than something that can silently drift between the two renderers —
  verified directly: `viewer.title.get_text() == ascii.status_line(frame)` for every frame
  checked, both in `tests/test_viewer.py` and by hand across a 30-step rollout.
- **`scripts/record_rollout.py` (Step 37) gained two more optional extra keys this step:**
  `los_to_hero` (`bots.perception.raw_los`'s hero row — physical wall-only LOS) and
  `unit_radius` (a `SimParams` PER_ENV scalar, shared by every entity in an env, used to size
  the entity circles). Both follow the same "optional extra key" contract `revealed_to_hero`/
  `max_ammo` established in Step 37 — a bare `core.state.snapshot()`-only frame still plays in
  `ReplayViewer`, just without the reveal-ring overlay (falls back to never showing it) and with
  a fixed default radius (`_DEFAULT_RADIUS = 0.4`) instead of the config-accurate one.
- **Reveal overlay is `los_to_hero AND NOT revealed_to_hero`, not just `NOT revealed_to_hero`.**
  The plan's wording ("`raw_los` is True but `vis` is False") already specifies this, but it's
  worth calling out why it matters: per Notice 4, walls never hide anything from this game's
  fixed bird's-eye camera, so an entity that's merely not `los_to_hero` (a wall is between it
  and the hero) isn't "hidden" in any sense this game models — only bush-hiding is. Ringing
  every non-`revealed_to_hero` entity would incorrectly flag wall-blocked-but-not-bush-hidden
  entities too. Regression-tested directly: `test_reveal_ring_stays_off_when_los_is_blocked_by_a_wall`
  forces `revealed_to_hero=False, los_to_hero=False` and asserts no ring.
- **Per-slot entity artists (Circle/Rectangle-pair/Text/Circle) vs shared scatter for boxes/
  pickups/projectiles** was a deliberate split, not an inconsistency: entity annotations
  (HP bar, ammo text, dash edge, reveal ring) are inherently per-instance state that a shared
  `PathCollection` can't carry per-point as easily (color/size arrays yes, but not a whole
  second Rectangle+Text pair), and `n_entities` (`1 + n_enemies`) is small enough that one
  artist set per slot has no measurable perf cost. Boxes/pickups/projectiles are larger,
  visually-homogeneous populations exactly where the plan calls for shared scatter.
- **Blitting and the `g` world/view pan toggle are in tension, and this is a documented
  tradeoff, not a bug:** `FuncAnimation(blit=True)` caches one static background and only
  redraws the returned artists on top of it — correct and fast for the default static
  full-map camera (the "20 fps with blitting" acceptance target), but `g` mode re-centers the
  camera on the hero every frame, which blit's static-background model can't express. Cropped/
  panned mode instead calls `self.fig.canvas.draw()` (a full redraw, which also resyncs
  `FuncAnimation`'s blit background cache via the `draw_event` it emits) every frame it's
  active — correct, but not blit-fast. Noted in the module docstring as a forward pointer:
  a future pass could maintain a separate `FuncAnimation` per camera mode, or accept blit-drawn
  artists at a fixed viewport and pan the DATA instead of the axes limits (translate every
  artist's coordinates by `-hero_pos` each frame) to stay inside blit's model; out of scope for
  what a debug tool needs today.
- No test file was named in Step 38's own `Files:` line either (same as Step 37); written
  anyway (`tests/test_viewer.py`, 16 tests) for the same reason.
- Verified by hand, not just by pytest: rendered several frames to PNG (Agg backend, no
  display) across a plain rollout and a constructed bush/zone-shrink scenario, confirmed
  visually that zone bands correctly frame the safe rect with no overlap, the reveal ring
  circles exactly the bush-hidden-but-LOS-clear bot and no one else, and the `c`-toggled view
  rect is correctly centered on and sized around the hero.
- This machine's actual FPS/blit throughput was not benchmarked in this step (no display
  attached in this environment) — the artist-update-in-place / blit-eligible structure is in
  place per the plan's performance requirement, but the literal "20 fps" number is unverified
  here. Flagged as a blocker-adjacent note, not blocking: Step 39/40 give more opportunities to
  check real playback feel once there's an interactive session to check it in.

---

## Step 39 — `scripts/play_manual.py`

**Files:** `scripts/play_manual.py`

Live keyboard play, `n_envs=1` at 20 Hz. WASD / arrows select the nearest of the 16 move bins;
space or left-click fires. Display the action mask and current dash state.

**This is your feel check.** If Mortis's dash feels wrong, `brawlers.yaml` is wrong — fix the
numbers before any training. Watch specifically whether "dash in the move direction, replacing
the walk" reads as a short committed hop; `dash_duration` and `dash_distance` are the knobs.

**Acceptance:** playable; the dash covers `dash_distance` tiles in `dash_duration` seconds and
visibly cancels that tick's walk; snipers punish standing in the open; the attack indicator
greys out during dash and while reloading.

**Implementation notes (filled in during Step 39):**
- `ManualPlaySession` reuses `render/viewer.py`'s `ReplayViewer` PURELY as an artist/drawing
  toolkit -- it never calls `ReplayViewer.show()` (that method's own `FuncAnimation` and key
  bindings are built for scrubbing a pre-recorded rollout, e.g. space=pause, and would collide
  with this step's WASD/space controls, where space means fire). Each tick it replaces
  `viewer.frames` with a fresh one-frame dict and calls `viewer._draw_frame(0)` directly -- the
  exact same shape trick `tests/test_viewer.py` already used to feed `ReplayViewer` hand-built
  scenarios. No changes to `render/viewer.py` itself were needed; its "optional extra keys, one
  frame at a time" design (Step 38) turned out to already support live driving for free.
- **`scripts/record_rollout.py`'s `_extra_fields` was renamed to public `extra_fields`** (Step
  39, no behavior change) so this step can import it directly instead of writing a third copy
  of the `revealed_to_hero`/`los_to_hero`/`max_ammo`/`unit_radius` computation -- same treatment
  `render/ascii.py`'s `status_line` got in Step 38 for the identical reason (one renderer-facing
  computation, three consumers, not three copies that can drift).
- **WASD/arrow direction is converted to a move bin via `core/geometry.py`'s existing
  `bin_from_dir`**, previously unused outside its own module/tests -- it matches `decode_action`'s
  own action-shape contract exactly (`move_bin=0` is idle, `bin_from_dir(...) + 1` otherwise),
  and because it already existed, tested, and unused outside its own module, writing a second
  angle-to-bin table by hand would have been a real (if small) risk of silently drifting from
  `decode_action`'s convention.
- **Held-key state (a `set`, updated by `key_press_event`/`key_release_event`), not discrete
  key events, drives both movement and fire.** This is what makes "hold W to walk" or "hold
  space through a reload, fire the instant it's off cooldown" work the way a human playtester
  expects, and it leans directly on `core/hero.py.decode_action`'s own documented contract that
  illegal fire is a silent no-op, never an error -- holding space while ammo/cooldown/dash gate
  it closed is expected, harmless input, not something this module needs to pre-filter.
- **The attack indicator is exactly `hero.action_mask(...)["attack"][:, 1]`** -- the SAME
  boolean gate `decode_action` itself applies before any fire reaches the sim, not a
  hand-rolled approximation of it. Since that gate is `alive & ammo>=1 & cooldown<=0 &
  dash_t<=0`, "greys out during dash and while reloading" (the plan's own acceptance wording)
  falls out of one shared boolean, not two separately-coded cases that could disagree with the
  sim's own rule about what counts as legal fire.
- **The dash feel-check itself was verified programmatically, not just by eye** (Step 38
  already flagged that this environment has no attached display to confirm real interactive
  feel): `tests/test_play_manual.py::test_dash_covers_dash_distance_in_dash_duration_and_cancels_the_walk`
  drives the exact same held-key -> `action_tensor` -> `env.step` pipeline a human would use,
  against the REAL `configs/brawlers.yaml` Mortis numbers (not a tiny debug preset), and asserts
  net displacement over `dash_duration` is within 0.05 tiles of `dash_distance` (5.0) -- tight
  enough that a normal walk contribution during those same ticks (which would add ~0.08 tiles at
  `move_speed=2.56`) would fail it, directly confirming `movement.apply_movement`'s dash-exclusion
  (Step 12's phase-7 gate) still cancels the walk exactly as designed.
- **Two acceptance-bar items are inherently human-judgment calls this automated pass can't
  close, same honest-gap category as Step 38's unverified literal FPS number:** whether the
  dash "reads as a short committed hop" and whether "snipers punish standing in the open" are
  both about how play FEELS, not a checkable invariant -- the numbers back them up (dash covers
  5 tiles in 0.3s vs. a 2.56 tiles/s walk; `bot_sniper` has `attack_range: 11.0`, tested
  separately in earlier steps' bot-policy suites) but actually pressing WASD and watching it is
  still the real acceptance test here, and remains a manual step for whenever there's a display
  attached to this machine.
- No test file was named in Step 39's own `Files:` line either (same pattern as Steps 37/38);
  written anyway (`tests/test_play_manual.py`, 17 tests, headless Agg backend).
- Added `tests/conftest.py` this step: `test_viewer.py` (Step 38) and `test_play_manual.py`
  each construct a real matplotlib `Figure` per test and never closed it, which was harmless
  individually but tripped matplotlib's "more than 20 figures open" `RuntimeWarning` partway
  through a full suite run -- a real, if minor, resource leak, not just log noise (each open
  `Figure` holds its full artist tree in memory). One autouse `plt.close("all")` fixture fixes
  it for every test file, not just these two.
- CUDA-verified: built a `cuda`-device env, drove `ManualPlaySession.tick()` through several
  ticks (movement + a full dash), confirmed no crashes and the title/indicator overlay reads
  correctly (`fire=blocked  dash=ACTIVE(...)` mid-dash) -- the per-tick `.item()`-style host
  syncs this module does (`bool(mask[...])`, `float(ent_dash_t[...])`) are fine here since this
  is a 20 Hz human-input loop, not the simulation's hot path CONVENTIONS.md's no-sync rule
  actually scopes to.

---

## Step 40 — `scripts/benchmark.py`

**Files:** `scripts/benchmark.py`

Sweep `n_envs` in `[1, 64, 256, 1024, 4096, 16384]`. Report env-steps/sec, ms/step,
`torch.cuda.max_memory_allocated()`. **Benchmark three paths separately:**
1. native `BrawlVecEnv.step` (zero-copy, the ceiling),
2. `BrawlSB3VecEnv.step` with `info_mode="minimal"`,
3. `BrawlSB3VecEnv.step` with `info_mode="full"`.

Break down per tick phase with `torch.cuda.Event` timers (not `time.time()`). Report the
observation footprint with and without `include_world_grid`, and the per-step host transfer
byte count for each SB3 mode. Warn loudly on CPU or if `sm_120` is missing.

**Native-path target on a 5070 Ti (16 GB):** past 100k env-steps/sec at `n_envs >= 1024`. If
an order of magnitude below, check in order: (1) a hidden `.item()`/`.cpu()` sync — run with
`set_sync_debug_mode("warn")`; (2) an accidental `(N,H,W)` gather in terrain; (3) boolean-mask
indexing in a hot path; (4) float64 via a Python float in a `torch.where`; (5) the world grid
rebuilt by gather instead of scatter.

**Expect the SB3 path to be substantially slower and to diverge from the native path as
`n_envs` grows** — that's the cost of the contract, not a bug. Use this benchmark to pick your
SB3 `n_envs`.

**Memory at `n_envs=4096`:** state ~250 MB, view grid ~36 MB (uint8), world grid ~162 MB
(uint8), entity/projectile tables ~30 MB; LOS ray sampling is the spiky one. If you OOM: turn
off `include_world_grid`, then coarsen `los_step_tiles`, then chunk the entity dimension in
`perception.raw_los` (the physical-LOS ray sampler — `perception.visibility` is bush-only and
has no ray sampling to chunk, Notice 4) — reduce `n_envs` last.

**Implementation notes (filled in during Step 40):**

- **`scripts/benchmark.py` measures three genuinely different things through three separate
  functions, not one shared "runner"**: `throughput_native`/`throughput_sb3` (wall-clock
  env-steps/sec + peak VRAM via `time.perf_counter()` with a `torch.cuda.synchronize()` boundary
  — the plan's "not `time.time()`" instruction scopes to the PHASE breakdown, this is the
  standard correct pattern for a top-level GPU throughput number), `phase_breakdown`
  (per-Section-4-phase device-side timing via real `torch.cuda.Event`s), and
  `observation_footprint` (purely analytical byte arithmetic from `obs_schema.obs_spec` — no
  env construction at all, so it works for an `n_envs` no machine could actually allocate).
- **`phase_breakdown` times the REAL, unmodified production code path**, not a hand-copied
  mirror of `_run_tick`'s phase sequence: `_instrumented_phases` monkeypatches each of `env.py`'s
  17 phase methods (`_run_tick`'s 15 + `step()`'s own `_observe`/`_autoreset`) onto the env
  INSTANCE with a timing wrapper, restored on exit via a context manager. Since `self._phase(...)`
  calls resolve by attribute lookup at CALL time, not at `_tick_fn`-construction time, this is
  picked up transparently — no risk of the mirror silently drifting from a future `_run_tick`
  reorder, the risk a first draft of this step (hand-copying the phase sequence into the
  benchmark script) would have carried. Only valid for `cfg.compile=False` (the default, and
  `main()` never exposes a `--compile` flag) — a compiled `_run_tick` traces its phase calls
  away, and instance-attribute monkeypatching after that point means nothing.
- **`host_bytes_per_step` is computed analytically from `BrawlSB3VecEnv`'s own persistent host
  buffer tensors** (`._host_buffers`, `._reward_host`, etc.), not measured -- it's a fixed
  function of `agent_spec`/`cfg`/`n_envs`/`info_mode`. For `info_mode="full"`, adds
  `_FULL_DIAG_BYTES_PER_ENV * n_envs` for `_maybe_attach_full_diagnostics`'s own extra
  `.to("cpu")` transfers (hero hp_frac/pos/alive + meta step_count/n_alive), also derived
  analytically rather than measured, and assumes the default `info_every=1` (this script never
  overrides it). Separately reports a "worst case" number (`+ obs_bytes` again) for the
  CONDITIONAL `terminal_observation` transfer that only fires on ticks with >= 1 done env.
- **Added `verbose: bool = True` to `BrawlVecEnv.__init__`** (threads through to
  `core.state.allocate`'s own pre-existing `verbose` param, which `env.py` previously never
  exposed at all): purely additive, default preserves every existing caller's behavior exactly.
  `scripts/benchmark.py` constructs 12+ envs per full sweep run and had no way to suppress
  `allocate`'s per-construction memory-report printout before this -- it was burying the actual
  benchmark results under walls of per-field byte tables. `benchmark.py` passes `verbose=False`
  at all three construction sites; every other caller is unaffected.

**Real numbers from this machine (RTX 5070 Ti, sm_120, Windows), `--steps 20 --warmup 5`
(reduced from the default 30/10 to keep the verification pass quick — re-run with the defaults,
or higher, for a tighter final read):**

| n_envs | native steps/sec | native ms/step | SB3 minimal steps/sec | SB3 full steps/sec |
|---|---|---|---|---|
| 1 | 13 | 78.3 | 12 | 12 |
| 64 | 779 | 82.2 | 773 | 759 |
| 256 | 3,371 | 75.9 | 3,126 | 3,082 |
| 1024 | 12,470 | 82.1 | 10,875 | 12,167 |
| 4096 | 34,929 | 117.3 | 35,007 | 33,886 |
| 16384 | 51,767 | 316.5 | 51,445 | 49,505 |

**The native path does NOT hit the plan's 100k+ steps/sec target at any `n_envs` tested,
including 16384 (~52k, roughly half target) — this is the honest, load-bearing finding this
step exists to surface, not a benchmark-script bug.** Root-caused by hand, per the plan's own
troubleshooting list, going further than list items (1)-(5) required:

- `torch.cuda.set_sync_debug_mode("warn")` around one `env.step()` call raised no warnings —
  ruling out item (1) (a stray `.item()`/`.cpu()` forcing a sync) as the primary cause, though
  the mode's own doc string admits it "does not yet detect all synchronizing operations."
- `torch.profiler` around one `env.step()` call at `n_envs=256` found **7,457 individual
  `cudaLaunchKernel` calls in a single tick**, costing ~46 ms of pure CPU-side kernel-dispatch
  overhead alone (out of ~83-97 ms total step time) — this is why ms/step is nearly FLAT across
  `n_envs=1` through `1024` (78-86 ms regardless of batch size, confirmed with 60 warmup + 100
  measured steps, ruling out a warmup artifact): at these batch sizes the sim is
  **launch-count-bound, not compute-bound** — each of the ~7,500 tiny kernels finishes almost
  instantly on-device, but Windows' `cudaLaunchKernel` dispatch cost (~6 µs/call here, measured)
  dominates when there's this many of them per tick. Throughput only starts scaling with
  `n_envs` once per-kernel compute time grows enough to matter (1024 -> 16384 does show real,
  if sublinear, improvement).
- **This is a SIXTH root cause the plan's own 5-item list didn't anticipate**, and it's not
  one any of items (2)-(5) describe either (those are compute/memory-efficiency bugs; this is
  pure per-call dispatch overhead from a highly decomposed, many-small-ops eager implementation
  across 17 phases and 4 bot archetypes). The per-phase breakdown (below, `n_envs=1024`)
  localizes WHERE most of those ~7,500 launches live: `_bot_phase` (34%) and `_autoreset`+
  `_observe` (23%+19% = 42% combined, large dict/obs-building work) dominate; the physical-sim
  phases (`_movement_phase`, `_attack_phase`, `_projectile_phase`, ...) are comparatively cheap.
- **The only real fix for launch-count-bound eager overhead is kernel fusion — `torch.compile`
  or CUDA graph capture — and Step 31 already found both broken on this specific machine**
  (Windows CPU inductor needs `cl.exe`, not installed; Windows+Blackwell CUDA inductor needs a
  working Triton install, also not installed). This benchmark step is what turns that from "a
  nice-to-have that happens to be unavailable" into "the concrete, measured reason native
  throughput can't hit the plan's own target on this machine" — **flagged to the user as the
  actual decision point Step 40 exists to surface**, not solved here: a deep hand-fusion
  refactor of `_bot_phase`/`_observe`/`_autoreset` is a large, separate undertaking well outside
  "write a benchmark script," and is not attempted in this step.

Per-tick phase breakdown at `n_envs=1024` (17 phases, `sum ≈ 83.6 ms` vs. this same
n_envs's own measured `82.1 ms/step` -- the two agree to within noise):

| phase | ms | % |
|---|---|---|
| `_bot_phase` | 28.32 | 33.9% |
| `_autoreset` | 19.50 | 23.3% |
| `_observe` | 15.48 | 18.5% |
| `_movement_phase` | 7.22 | 8.6% |
| `_attack_phase` | 5.44 | 6.5% |
| `_projectile_phase` | 3.77 | 4.5% |
| (remaining 11 phases) | 3.45 | 4.1% combined |

Observation footprint at `n_envs=4096` (matches the plan's own quoted ~162 MB world-grid figure
closely: 42.2 KB/env × 4096 ≈ 173 MB, the small delta being this run's actual `n_enemies`):
per-env 68.6 KB with the world grid / 26.4 KB without (world grid alone: 42.2 KB/env); total
274.4 MB with / 105.7 MB without.

SB3 host-transfer bytes/step scale linearly with `n_envs` as expected (e.g. at 4096: minimal
35.7 MB/step, full adds a small, fixed `21 B × n_envs` on top for the diagnostic snapshot) —
`info_mode="full"`'s steps/sec is statistically indistinguishable from `"minimal"`'s in this
run, meaning at these `n_envs` the diagnostic dict-building isn't yet the bottleneck the launch-
count problem already is; a machine without the launch-overhead issue would likely show a
clearer minimal-vs-full gap.

CPU/non-sm_120 warning path verified by hand (monkeypatched `torch.cuda.is_available` to
`False`; confirmed the loud warning banner + CPU fallback, no crash).

No new test file was named in Step 40's own `Files:` line; written anyway
(`tests/test_benchmark.py`, 14 tests) for the same reason as Steps 37-39. All small/fast/CPU —
the real hardware sweep above was a separate, one-off manual verification run, not part of the
automated suite (a 16384-env sweep across 3 paths has no place in a fast test suite).

**Decision (post-Step-40): accept the launch-overhead ceiling as a known platform limit (Option
A from the three options above), not a blocker.** 12k-52k env-steps/sec (depending on `n_envs`)
is well within usable range for training; the hand-fusion refactor (Option B) and the
WSL2/`torch.compile`-unblock investigation (Option C) are both explicitly deferred, not pursued
in this pass.

**Resource-sizing follow-up (this machine: Ryzen 9800X3D, 8 cores; RTX 5070 Ti, 16 GB VRAM; 32
GB system RAM) -- answers "what does this actually cost to run," not part of `benchmark.py`
itself, just recorded here since it came up immediately after this step and future-you (or
whoever tunes training hyperparameters later) shouldn't have to re-derive it:**

- **VRAM is not the constraint at any `n_envs` worth training at.** Even `n_envs=16384` peaks
  under 8 GB (measured, both paths) -- half the card. The practical `n_envs` range (1024-4096,
  balancing throughput against the launch-overhead ceiling above) uses well under 2 GB.
- **PPO's rollout buffer (host RAM, not VRAM) scales as `n_steps * n_envs *
  per_env_agent_obs_bytes`.** Using this step's measured ~8.9 KB/env agent observation at
  typical settings (`n_steps=64`, `n_envs=1024`, matching `sb3_smoke.py`'s own defaults): ~580
  MB -- trivial against 32 GB. Recompute this product before pushing `n_steps` much higher;
  it's the one number here that scales with a training hyperparameter, not just `n_envs`.
- **The bottleneck (per-tick kernel-dispatch overhead) is single-process/single-CPU-thread by
  nature** -- Python's GIL and one sequential CUDA-launch stream mean threading within one
  process can't parallelize it, and the 17 phases are sequentially dependent on each other
  besides. This CPU/GPU combination's real leverage point is therefore **running several
  independent training processes concurrently** (different seeds/hyperparameters), not making
  one run faster -- VRAM/RAM headroom above comfortably supports 4-6 concurrent runs at
  `n_envs` in the low thousands. A harder, unproven alternative -- sharding ONE run's `n_envs`
  across multiple processes/CUDA contexts to parallelize the dispatch cost itself -- is
  plausible in theory (the bottleneck is CPU-side, not GPU-compute-side) but would need a new
  multi-process `VecEnv` wrapper this codebase doesn't have yet (nothing like SB3's
  `SubprocVecEnv` exists for a GPU-BATCHED sub-env), and consumer GPUs don't guarantee clean
  concurrent kernel execution across contexts without NVIDIA MPS. Deferred, not attempted here
  -- noted as a possible future step if a single run's wall-clock speed specifically becomes
  the blocker, not before.

---

## Step 41 — `scripts/smoke_test.py` + integration tests

**Files:** `scripts/smoke_test.py`, `tests/test_integration.py`

`smoke_test.py`: for every config in `configs/` and `configs/presets/`, build at `n_envs=8`,
reset, run 500 random steps with `debug_checks=True`, asserting: no NaN/Inf; positions in
bounds; `alive` never flips False→True mid-episode; `0 <= hp <= max_hp`; live projectiles
`<= max_projectiles`; `cubes <= max_cubes`; `validate_obs` passes every tick; every episode
ends within `max_episode_steps + 1`.

`tests/test_integration.py`:
- **Determinism:** same seed, same `n_envs`, same actions → bit-identical trajectories (holds
  only for fixed `n_envs`, per N07).
- **Isolation:** changing env 0's actions leaves envs 1..N-1 identical to a control run.
- **Autoreset boundary** (Step 30).
- **Randomization:** a ranged param differs across envs and resamples on reset (N08).
- **Full-information obs:** a bush-hidden bot is present in `obs["entities"]` with correct
  position and `revealed_to_hero = False`.
- **Fairness gating:** the same bot contributes all-zero fields in the agent obs under
  `fair: true`, and non-zero under `fair: false`.
- **Privileged isolation:** no field under `entities.privileged` appears anywhere in the agent
  observation, for any spec (walk the built obs and assert).
- **Dash-overrides-walk:** a dashing tick produces no walk displacement.
- **SB3 contract:** `terminal_observation`, `TimeLimit.truncated`, and `episode` all appear
  exactly when they should.
- **Config overrides:** `n_enemies`, `view_h/w`, `map_h/w`, `zone.enabled`,
  `action_latency_seconds`, `include_world_grid`, `agent_obs.fair` each changeable with
  everything still running.
- **Sync-free (native path):** `set_sync_debug_mode("error")` around 100 `BrawlVecEnv.step`
  calls.

**Acceptance:** `pytest -x` green on CPU; `smoke_test.py` exits 0 on CPU and GPU.

**Implementation notes (filled in during Step 41):**

- **"every config in `configs/` and `configs/presets/`" is read as "every world-config
  variant," not literally every `.yaml` file in those two directories.** `configs/
  brawlers.yaml`/`agent_obs.yaml`/`randomization.yaml` aren't `EnvConfig` sources at all --
  they're auxiliary specs merged into or layered alongside a real config; pointing `load_config`
  at one directly wouldn't error (any dotted key it doesn't recognize is silently skipped), it'd
  just silently test nothing new while looking like extra coverage. `smoke_test.py` builds:
  `configs/default.yaml` alone, plus `default.yaml` + each `configs/presets/*.yaml` overlay --
  exactly how every preset file's own docstring already says it's meant to be used.
- **`core.state.check_invariants` already runs automatically inside every `BrawlVecEnv.step()`/
  `reset()` once `debug_checks=True`**, and already covers 8 of the plan's acceptance items
  (NaN/Inf, position bounds, `0<=hp<=max_hp`, no dead-with-nonzero-hp, `cubes<=max_cubes`,
  `dash_t<=dash_duration`, live-projectile-count, `act_head`/`env_idx` bookkeeping) --
  `smoke_test.py` does not reimplement any of them, just sets `debug_checks=True` on every
  config it builds. What it adds on top (things `check_invariants` structurally can't check):
  alive-never-resurrects-mid-episode and the `max_episode_steps+1` episode-length bound, both
  tracked as BLACK-BOX checks against `env.step()`'s own public `terminated`/`truncated`
  return values (not read off `SimState`'s internal `step_count`/bookkeeping) -- a deliberate
  choice, since reading internal state would only prove the sim is self-consistent, not that
  its public autoreset CONTRACT is honored; plus `validate_obs` every tick.
- **`tests/test_integration.py` does NOT give all eleven plan bullets a brand-new test.**
  Researched existing coverage first (rather than assume): autoreset boundary
  (`tests/test_autoreset.py`), dash-overrides-walk (`test_env.py`/`test_hero.py`/
  `test_play_manual.py`), and the SB3 `terminal_observation`/`TimeLimit.truncated`/`episode`
  contract (`test_sb3_vecenv.py`) already had thorough, dedicated coverage -- duplicating them
  here would be a second copy that could drift, not more safety. Added one light autoreset
  check anyway (the property the other 8 new tests in this file implicitly lean on), but didn't
  re-derive `test_autoreset.py`'s full per-env-isolation matrix. The other 7 items had genuine,
  confirmed gaps against true multi-module/end-to-end coverage:
  - **Determinism/isolation** existed only through `scripts/record_rollout.py`'s single-env
    numpy wrapper (`test_record_rollout.py`) -- nothing constructed two independent
    multi-env `BrawlVecEnv`s and diffed full `SimState` (every field, via `torch.equal`) across
    20 real steps. Now does.
  - **Randomization** was tested at the `build_params`/`resample_params` layer directly
    (`test_config.py`), never through `BrawlVecEnv`'s own public `randomization=` constructor
    arg + `reset(mask)`. Now does -- and had to hand-build a `RandomizationSpec` dict rather
    than load the real `configs/randomization.yaml`, since that file **ships with every line
    commented out** (a template, not a default) -- `load_randomization` on it returns `{}`,
    asserted explicitly so this doesn't silently regress into a vacuous test later.
  - **Fairness gating** was tested with a hand-built `AgentObsSpec`, never the actual shipped
    `configs/agent_obs.yaml`. Now loads it for real and flips `fair` via `dataclasses.replace`
    (the field's a frozen dataclass member) to get the `fair=false` variant from the SAME spec.
  - **Privileged isolation** only had one hardcoded field name tested
    (`entities.privileged.target_id`). Now enumerates every field `obs_schema.OBS_SCHEMA` marks
    `privileged=True` and confirms `load_agent_spec` rejects each individually, plus confirms
    the real shipped `agent_obs.yaml` exposes none of them.
  - **Combined config overrides** existed only individually, scattered across files -- no test
    changed `n_enemies`/`view_h/w`/`map_h/w`/`zone.enabled`/`action_latency_seconds`/
    `include_world_grid`/`agent_obs.fair` TOGETHER and ran `reset()`+`step()` end to end. Now
    does (had to swap a mismatched custom `map_h/w` for the `blank` map's real 20x20 size --
    `maps.loader.load_map_csv` enforces the CSV's actual dimensions match `cfg.map_h/map_w`,
    same combo `debug_tiny.yaml` already uses for exactly this reason).
  - **Sync-free native stepping** had no test at all (only a docstring comment in
    `core/geometry.py` mentioning the technique was used once by hand while building `env.py`).
    Now a real CUDA-only test (skipped cleanly on CPU, matching `test_sb3_integration.py`'s own
    established CUDA-skip pattern): `set_sync_debug_mode("error")` around 100 real
    `BrawlVecEnv.step()` calls (after a few untimed warmup steps, since first-call lazy
    allocation is expected to sync and isn't what this checks) -- **passed on this machine**,
    confirming by construction what Step 40's `set_sync_debug_mode("warn")` probe already
    suggested: the native path's throughput ceiling (Step 40) is launch-count overhead, not a
    hidden host sync.
- **Verified for real, not just via the automated suite:** `smoke_test.py` run at full scale
  (`n_envs=8`, `steps=500`, all 4 config variants) on BOTH CPU and CUDA on this machine -- all
  8 runs (`4 variants x 2 devices`) passed clean.

---

## Step 42 — README

**Files:** `README.md`

Document: the sm_120 install trap and the `get_arch_list()` check; the config schema and
scalar-or-range syntax; pointers to `docs/OBSERVATION.md` (full package) and
`docs/AGENT_OBS.md` (SB3-facing); the action space and masking; the tick order, emphasizing
that attack resolves before movement; autoreset semantics; the map CSV legend; the two-path
architecture (native torch vs SB3) and when to use each; and Appendix A.

Usage example:
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

**Implementation notes (filled in during Step 42):**

- **Found and fixed a real bug in this very usage example, not just in README.md:** the
  `overrides={"entities.n_enemies": 6}` line above used a flat dotted-string key, but
  `config.py`'s `_deep_merge` is a plain recursive DICT merge with no dotted-path flattening --
  it only matches keys structurally (`out["entities"]["n_enemies"]`, not a literal
  `out["entities.n_enemies"]` key). Passing the dotted form silently adds an inert, never-read
  top-level key and the override never applies -- `n_enemies` would silently stay at its
  `EnvConfig` default. This went unnoticed here specifically because the default (6) and the
  example's intended override (6) happen to be the same number, so the example "looks like it
  works" even though the override line does nothing. Fixed to the nested form
  (`{"entities": {"n_enemies": 6}}`) that every script and test in this repo has used
  consistently since Step 3 -- verified by actually running the corrected example end to end
  (`MaskablePPO.learn` + a native `sim.step()` call, both completing cleanly) before treating
  it as fixed, not just eyeballing the diff.
- **README.md was expanded, not replaced** -- Step 1's original Setup/sm_120-install-trap
  section is untouched; everything else in this step's list (config schema, action space/
  masking, tick order, autoreset, `docs/OBSERVATION.md`/`docs/AGENT_OBS.md` pointers, map CSV
  legend, native-vs-SB3 guidance, Appendix A) is new. Regenerated `docs/OBSERVATION.md`/`docs/
  AGENT_OBS.md` via `scripts/dump_obs_schema.py` before writing pointers to them and confirmed
  (via `git status`) they were already current -- no drift to catch this time, but worth
  checking rather than assuming.
- **Added a "Scripts" table** (not explicitly asked for by this step's text, but a natural
  extension of "usage example" for a README's actual job — orientation): every script under
  `scripts/` built across Steps 36-41, one line each, so a new reader doesn't have to grep the
  directory to find out `benchmark.py`/`smoke_test.py`/`play_manual.py`/`record_rollout.py`
  exist at all.
- **Appendix A is summarized in the README, not duplicated verbatim** — one link back to this
  file's own Appendix A for the full list/rationale, since a second full copy is exactly the
  kind of thing that silently drifts (same reasoning already applied to `render/ascii.py`'s
  `status_line` and `record_rollout.py`'s `extra_fields` reuse in Steps 38-39, just at the
  documentation layer instead of the code layer this time).
- Appendices B (delegation prompt template) and C (parallel build order) are deliberately NOT
  referenced from the README -- both are about how THIS plan was meant to be delegated/built in
  parallel, not about how to use the finished sim; out of a user-facing README's scope.

---

# Appendix A — Known divergences from the real game

1. **No Supers, Gadgets, Star Powers, or Hypercharges.** Mortis's Super and Star Powers change
   his kit materially; this is his primary attack only.
2. **Health regeneration off by default.** Real Showdown regenerates HP a few seconds after
   last damage, which changes kiting a lot. `combat.apply_regen` and the config keys exist and
   work — turn them on early.
3. **16-bin movement, attack has no independent aim.** The dash inherits the move direction.
4. **Soft unit collision.** Real units block each other.
5. **No knockback, stuns, or slows.**
6. **Simplified bushes.** Real mechanics involve bush clusters and reveal-on-damage.
7. **20 Hz tick.** `action_latency_seconds = 0.001` rounds to 0 ticks; the smallest
   representable non-zero latency is 50 ms, which would need a higher tick rate to refine.
8. **No cube redistribution or timed cube spawns.**
9. **Bot policies are hand-written heuristics**, not players. Randomize their difficulty
   parameters (N08) or the agent will overfit to them hard.
10. **No per-env RNG determinism** (N07). Trajectories reproduce for a fixed `n_envs`.
11. **The simulator does not restrict the hero's information.** Fairness lives in
    `AgentObsSpec.fair`, one config flag away from being switched off. That's deliberate — it
    makes the cheating-vs-fair comparison a one-line experiment — but it also means a
    misconfigured spec silently trains a cheating agent. The privileged-isolation test in
    Step 41 is the guard; keep it.

---

# Appendix B — Delegation prompt template

```
You are implementing one step of a batched PyTorch Brawl Stars simulator.

Read CONVENTIONS.md in the repo root first and follow every rule, especially:
- everything is a tensor with a leading (N,) env dimension; NEVER loop over envs
  (the ONLY exception is wrappers/sb3_vecenv.py, which SB3's contract forces)
- no host syncs: no .item(), .cpu(), .tolist(), bool(t), if t:, t[bool_mask], nonzero()
  (again, sb3_vecenv.py is the one sanctioned exception)
- no data-dependent shapes; use torch.where and masked arithmetic
- float32 for reals, int64 for indices, bool for masks, uint8 for observation grids
- distance in tiles, time in seconds, angles in radians, y increases downward
- entity index 0 is the hero; entity slot order is stable for the whole episode
- functions mutate state in place and must declare "MUTATES: <fields>" in the docstring
- read every number from EnvConfig / SimParams / AgentObsSpec; hardcode nothing
- the full observation package hides nothing: visibility is reported as flags, never used
  to mask, truncate, or sort data out. Fairness gating happens ONLY in obs_select.py.

Implement EXACTLY the following step and nothing more. Do not modify files outside the
"Files" list. Do not add dependencies. Do not write a reward function or training code
beyond what the step explicitly asks for.

<PASTE THE STEP HERE>

Output: (1) complete contents of each file, (2) the test file, (3) the command to run the
tests, (4) any assumptions you had to make, (5) any config key, OBS_SCHEMA entry, or
AgentObsSpec field you needed that does not already exist.
```

---

# Appendix C — Build order for parallel delegation

- **Critical path (serial):**
  1 → 2 → 3 → 4 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 21 → 24 → 25 → 26 → 27 → 29 →
  32 → 33
- **Parallel once Step 9 lands:** 5, 15, 16, 22, 23, 28
- **Parallel once Step 16 lands:** 17, 18, 19, 20 (four independent archetypes)
- **Parallel once Step 29 lands:** 30, 31, 37, 38, 39, 40
- **Parallel once Step 33 lands:** 34, 35, 36, 41, 42

**Checkpoint 1 — after Step 29.** Run one env end to end and watch it in ASCII with
`debug_checks = True`. Don't scale `n_envs` until a single 3000-step rollout is clean; batched
bugs are far harder to diagnose than single-env ones, and `check_invariants` plus
`validate_obs` are your only safety nets now that there's no functional purity to lean on.

**Checkpoint 2 — after Step 36.** `sb3_smoke.py` must complete `learn(2048)` before you write
a real reward function. Debugging a reward against a broken observation pipeline is the worst
possible ordering, and the smoke test is cheap.