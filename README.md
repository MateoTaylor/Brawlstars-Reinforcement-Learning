# brawlstars-rl

Two halves of one goal: train an agent to play Brawl Stars Solo Showdown, then let it see the
real game.

- **`brawl_sim`** — a batched, GPU-resident approximation of Solo Showdown. Thousands of
  independent matches step in lockstep as one set of tensors, with no Python loop over
  environments and no host round trip in the hot path. A native PyTorch `BrawlVecEnv` plus a
  Stable-Baselines3 `VecEnv` adapter, and a PPO training stack on top with shaped rewards, a
  difficulty curriculum, and fixed-seed evaluation.
- **`brawl_vision`** — the perception side. Takes a screen capture or an mp4 of a real match and
  reconstructs the map: calibrated homography, inverse-perspective rectification, camera
  odometry, per-cell terrain classification, and an accumulated world occupancy grid. The point
  is that a policy trained in `brawl_sim` should eventually be handed the same view from the
  real game.

The sim half is built and trains. The vision half runs end to end — recorded match in,
reconstructed map video out — at 88% per-cell accuracy with zero measured odometry drift; walls
are the outstanding classifier failure. Nothing yet connects a trained policy to live frames.

## What's in the box

| | |
|---|---|
| `brawl_sim/core/` | the simulation: movement, combat, projectiles, zone, boxes, spawn, observation |
| `brawl_sim/bots/` | bot opponents — one data-driven fire rule, five movement personalities |
| `brawl_sim/training/` | PPO stack: shaped reward, curriculum, schedules, stationary eval |
| `brawl_sim/wrappers/` | SB3 `VecEnv`, single-env `gymnasium.Env`, feature extractor |
| `brawl_sim/render/` | ASCII renderer and a scrubbable matplotlib replay viewer |
| `brawl_sim/maps/` | eight CSV maps (seven 60x60, one tiny for tests), six in the training rotation |
| `brawl_vision/` | frame sources, camera calibration, odometry, terrain, occupancy |
| `brawl_vision/object_detection/` | third-party YOLO entity detector over the raw frame |
| `configs/` | every number in the project; nothing is hardcoded |
| `scripts/` | the runnable entry points below |
| `tests/` | ~60 test modules, CPU by default |

A match is 10 players on a 60x60 map, 150 seconds, shrinking zone, loot boxes that drop power
cubes. The hero is Mortis (dash assassin); the seven bot brawlers are Brock, Grom, Buzz, Shelly,
Edgar, Spike and Bull. The world ticks at 20 Hz and the agent decides at 4 Hz. See
`CHARACTER_DETAILS.md` for the roster and `brawl_sim/maps/README.md` for the maps.

## Setup

```
pip install -e ".[dev,sb3,render]"
python scripts/check_install.py
```

Extras: `dev` (pytest), `sb3` (Stable-Baselines3 + TensorBoard), `render` (matplotlib),
`video` (mp4 export), `vision` (OpenCV + screen capture — only needed for `brawl_vision`).

### Install trap: Blackwell (sm_120) GPUs

On an RTX 50-series card, a plain `pip install torch` can pull a build whose kernels stop at
`sm_90`. It installs fine and then fails at *runtime* with `no kernel image is available for
execution on the device`. Install from the CUDA wheel index instead:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

(or a newer `cu12x`/`cu13x` channel — check https://pytorch.org/get-started/locally). The
definitive check is `torch.cuda.get_arch_list()` containing `'sm_120'`, **not**
`torch.cuda.is_available()`, which returns `True` even on a build that cannot launch a kernel.
`scripts/check_install.py` checks this for you. Windows-native is fine; no WSL needed.

## Things to run on a fresh clone

Everything here works with no trained model and no game footage:

```bash
python scripts/check_install.py           # CUDA, sm_120, and whether sb3/vision import
pytest tests/ -m "not slow"               # ~1140 tests, CPU (see "Running the tests")
python scripts/smoke_test.py              # every config variant, 500 random steps, invariants on
python scripts/benchmark.py               # env-steps/sec, per-phase timing, VRAM, obs footprint
python scripts/dump_obs_schema.py         # writes docs/OBSERVATION.md and docs/AGENT_OBS.md
python scripts/sb3_smoke.py               # proves the SB3 plumbing connects (not a training run)
```

`dump_obs_schema.py` is worth running first: both docs are generated and gitignored, so a fresh
clone doesn't have them. They are the field-by-field reference for what the sim observes
(`docs/OBSERVATION.md`, everything) and what a policy actually receives (`docs/AGENT_OBS.md`).

**Play it yourself** — WASD to move, space to fire, at the real 20 Hz. The human feel-check for
Mortis's dash, and the fastest way to understand what the bots do:

```bash
python scripts/play_manual.py
```

**Watch a random rollout** in the scrubbable viewer or in the terminal:

```bash
python scripts/record_rollout.py                       # -> rollout.npz
python -m brawl_sim.render.viewer rollout.npz          # scrub it: space pauses, arrows step
python -m brawl_sim.render.viewer rollout.npz --video replay.gif  # or render it headless
```

Sprite PNGs are extracted game assets and aren't in the repo — the viewer falls back to circles
and diamonds, which is fully functional. `.gif` export needs nothing extra; `.mp4` needs the
`video` extra for its bundled ffmpeg.

**Train something:**

```bash
python scripts/train.py --smoke     # ~1.5 min CPU wiring check, proves the loop end to end
python scripts/train.py             # the real thing, configs/train.yaml as-is
tensorboard --logdir runs
```

Then watch what it learned. `runs/` is gitignored, so this needs a checkpoint you trained
yourself:

```bash
python scripts/watch.py runs/mortis_ppo-.../best_model.zip --tier hard
python scripts/watch.py runs/.../best_model.zip --ascii        # terminal, no GUI
```

See **`TRAINING.md`** for the reward, the curriculum, the evaluation protocol, and what every
logged scalar means.

### The vision scripts

These need recorded match footage, which is Supercell IP and therefore gitignored — a fresh
clone has the calibration *outputs* but not the *inputs*, so you'll need to capture your own.
With a clip in hand:

```bash
python scripts/vision_watch.py clip.mp4        # every pipeline stage, side by side. Start here
python scripts/vision_watch.py --live          # or grab the screen instead of a file
python scripts/vision_label.py <clip> 700      # click-to-label terrain cells for training
python scripts/vision_train_terrain.py         # train the per-cell classifier
python scripts/vision_evaluate.py clip.mp4 -o map.mp4   # gameplay in, reconstructed map out
python scripts/vision_truth.py clip.mp4 --check         # score against a hand-verified grid
```

`vision_watch.py` is the tool the rest of the pipeline was built with — camera calibration,
rectification, odometry, zone detection and classification are all far faster to check by
looking than by reading numbers.

### Entity detection (third-party)

A separate, parallel chunk: `brawl_vision/object_detection/` runs **PylaEntityDetectorV2** — a
YOLOv11n trained by the PylaAI community, classes `enemy / teammate / player` — over the **raw**
frame and returns boxes in screen pixels. It is not built on the terrain pipeline and does not
feed it. Terrain rectifies and accumulates a world in tile space; this reads one frame and
remembers nothing. Putting them in the same output frame is so you can *look at both* before
deciding whether they should ever talk to each other.

The weights are fetched rather than committed (10.6 MB, AGPL, someone else's repo — see
`brawl_vision/object_detection/weights/NOTICE.md`):

```bash
pip install -e ".[detect]"                     # onnxruntime; use onnxruntime-gpu for CUDA
python scripts/vision_fetch_detector.py        # ~10 MB, digest-checked

python scripts/vision_detect.py clip.mp4 -o boxes.mp4    # detector alone. Start here
python scripts/vision_evaluate.py clip.mp4 -o both.mp4 --detect   # boxes AND the map
```

`--detect` implies `--layout side-by-side`: annotated frame left, reconstructed map right.
Measured at 12.9 ms/frame on CPU against the 250 ms decision budget, and the combined render
still runs faster than real time.

`--detect-on-map` additionally projects each box's bottom edge through the same homography the
terrain pipeline uses, and marks the resulting tile on the map. Paired with `--layout view` you
get the agent's own window twice — rectified frame left, predicted map right, same tiles at the
same scale, markers on both — which is where you can check whether a detected brawler lands on
the cell the map claims:

```bash
python scripts/vision_evaluate.py clip.mp4 -o view.mp4 --detect-on-map --layout view
```

**Where in the box you project from is the whole game.** A YOLO box here encloses the nameplate,
health bar, ammo pips, sprite *and* any active aura — so its bottom edge is the bottom of the
aura, measured ~68 px (about a tile) below the brawler's feet, and it breathes as effects come
and go (player box height swings 264→380 px within one clip).

`detector.anchor_frac` therefore projects from **30% of the box height up from the bottom**, not
from the bottom edge. It was measured against the one accuracy test available without hand
labels: a brawler cannot stand in a wall or in water, and the occupancy map knows where those
are. Share of frames placing the player in an impossible cell:

| anchor | 0% | 20% | 30% | 40% | 50% | base rate |
|---|---|---|---|---|---|---|
| `showdown_alternate_map` | 23.7% | **11.3%** | 11.9% | 14.9% | 23.8% | 26.5% |
| `showdown_alternate_map2` | 15.6% | 6.7% | 5.4% | **1.1%** | 0.8% | 19.1% |

The raw bottom edge is barely better than chance. The optimum is a *band* — the wall-heavy clip
prefers 0.20 and degrades past 0.40, the water-heavy one keeps improving — and they disagree
partly because the terrain classifier's own weak spot is walls, so some "impossible" cells are the
map being wrong rather than the anchor. 0.30 is the middle and near-best on both.

Still not a calibration: read a marker as "a brawler is about here", not as a tile index.
`--detect-anchor` overrides it. See `brawl_vision/object_detection/project.py`.

Two things worth knowing before trusting it. The model wants **RGB** — feeding it OpenCV's BGR
does not error, it produces *more* boxes: on one Solo Showdown clip, 44 "teammate" detections
where correct input yields none, plus duplicate players and lower confidence throughout. And
`teammate` is impossible in Solo Showdown, so `configs/vision.yaml` drops it by default; it fires
once in 101 frames at the shipped threshold, and a burst of them means something upstream is
wrong.

## Using the sim in code

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

Two things that trip people up:

- **`overrides` is a nested dict matching the YAML's structure** (`{"entities": {"n_enemies":
  6}}`), not a flat dotted key. It's deep-merged section by section.
- **`MyReward` is a placeholder.** `BrawlVecEnv`'s default reward is `ZeroReward` — the sim
  ships no opinion on what "good" means. Write your own `__call__(self, obs, info, cfg) ->
  torch.Tensor` (see `core/reward.py`'s `Reward` protocol, or use `training/reward.py`'s
  `ShapedReward`).

### Native or SB3?

| | native `BrawlVecEnv` | `BrawlSB3VecEnv` |
|---|---|---|
| Observation | full-information, device-resident torch tensors | the agent's fair, `AgentObsSpec`-selected view, numpy on host |
| Host transfers | none — zero-copy the whole way | every step, by SB3's contract |
| Scales to | `n_envs` in the thousands to tens of thousands | hundreds to low thousands before the host round trip dominates |
| Use it for | a custom GPU-native training loop, or max throughput | SB3's algorithms and ecosystem |

Both wrap the same `BrawlVecEnv` — the adapter never re-simulates anything. `scripts/benchmark.py`
finds the actual knee on your hardware. `wrappers/gym_single.py` is a third, narrower option: a
single-env `gymnasium.Env` for `check_env` and interactive play.

## Documentation map

| file | what's in it |
|---|---|
| `CONVENTIONS.md` | **read first if you're writing code here** — the batched-tensor rule, the no-host-sync rule, dtypes, shapes, and the canonical 17-phase tick order |
| `TRAINING.md` | reward design, curriculum, stationary evaluation, run layout, logging |
| `CHARACTER_DETAILS.md` | the eight brawlers and how their real kits were converted |
| `BRAWL_SIM_BUILD_PLAN.md` | the full sim design and build plan; Appendix A is the known divergences, Appendix D the runtime contract (config schema, action space, decision rate, autoreset) |
| `bot_overhaul.md` | the bot/hero overhaul; §8 is how the shipped bots behave and why |
| `Terrain_Perception_Build_Plan.md` | the vision pipeline, phase by phase, with what each measurement actually found |
| `brawl_sim/maps/README.md` | map CSV format, the tile legend, and how to design a map for this sim |
| `docs/OBSERVATION.md`, `docs/AGENT_OBS.md` | generated — run `scripts/dump_obs_schema.py` |

## Known divergences from the real game

The short version, from `BRAWL_SIM_BUILD_PLAN.md` Appendix A: no Supers, Gadgets, Star Powers or
Hypercharges; 16-bin movement with no independent aim; soft unit collision, no knockback, stuns
or slows; simplified bushes; 20 Hz tick; no cube redistribution or timed spawns; bot policies are
hand-written heuristics rather than real players (randomize their difficulty or an agent will
overfit to them); trajectories are deterministic only for a fixed `n_envs`.

One is worth stating outright: **the simulator does not restrict the hero's information by
itself.** Fairness is a single config flag (`AgentObsSpec.fair`) away from being switched off.
That's deliberate — it makes a cheating-vs-fair experiment a one-line change — but it means a
misconfigured spec can silently train a cheating agent. `tests/test_integration.py`'s
privileged-isolation tests are the guard; keep them green.

The egocentric view is `21x13` tiles, **measured** from the real Solo Showdown camera rather than
guessed (it was `20x40` before). Two camera divergences remain, both measured and both left in
place; `Terrain_Perception_Build_Plan.md` Phase K has the derivation and the numbers.

## Running the tests

```bash
pytest tests/ -n auto             # the whole suite in parallel (pytest-xdist ships in [dev])
pytest tests/ -m "not slow"       # ~1140 tests, CPU; skips the real-training-loop tests
pytest tests/test_vision_*.py     # vision only — a few seconds, and siloed from the sim
```

Serially the suite takes ~13 minutes, almost all of it `brawl_sim` — `-n auto` is worth it. Markers: `gpu` (needs CUDA —
these skip rather than fail without one), `slow` (runs a real if tiny training loop), `vision`
(needs footage under `tests/fixtures/vision/`, so it skips on a fresh clone).
