---
name: brawl-deployment
description: Rules for the brawl_deployment package — the live BlueStacks control loop (screen capture → brawl_vision → policy → touch input). Use when working in brawl_deployment/, configs/deployment.yaml, or brawl_deployment/data/, and whenever a task touches emulator input injection, the floating movement joystick, touch/ADB/sendevent/keymap backends, match-over or in-match detection, the live perception loop, or assembling a real-game observation vector for the trained policy.
---

# brawl_deployment

The live loop: BlueStacks screen → `brawl_vision` → trained policy → touch input → BlueStacks.
Full design and rationale in `BRAWL_DEPLOYMENT_DESIGN.md`. This file is the short form — the
rules that are load-bearing and expensive to rediscover.

## The boundary with CONVENTIONS.md

`CONVENTIONS.md` governs `brawl_sim`. It does **not** govern this package, exactly as it does not
govern `brawl_vision` (see that package's `__init__.py`).

**Does not apply here:** the leading-`(N,)` batch rule, the no-host-sync rule. There is one
screen, one match, one agent. `.item()`, `.cpu()`, `.numpy()`, plain Python loops, and OpenCV's
numpy-in/numpy-out contract are all fine. **Do not contort this code to look batched.**

**Still applies, because it is what lets the halves speak:** units (tiles, seconds, radians),
coordinates (`pos = (x, y)`, x = column, y = row, **y increasing downward**), and
`brawl_sim.constants.Tile` imported rather than redeclared.

## Coordinates: there is no y flip

`brawl_sim` world y increases downward. Screen y increases downward. They agree.

`dir_from_bin(k, 16)` returns `(cos(k·2π/16), sin(k·2π/16))`, so a screen-space joystick offset
is `radius · (cos θ, sin θ)` **with no negation on either axis**. Negating y here is a silent
100% failure that presents as "the agent walks into walls." If you find yourself adding a minus
sign to make something look right, stop and check the tile-space transform instead.

Confirmed live, not just derived: commanding bin 2 deflected the knob to (+64.4, **+65.6**), i.e.
45° **down**-right, which is what the unnegated formula predicts.

## The two-rate loop: capture 20 Hz, decide 4 Hz

Not a performance hedge — forced by `configs/vision.yaml`'s `odometry.max_shift_tiles: 2.0`,
which its own comment states is **per frame, not per second**, sized at 20 Hz. Capture slower and
every dash reads as a cut.

- **20 Hz (every tick):** capture, `normalize_viewport`, odometry, occupancy deposit, in-match
  ring score.
- **4 Hz (every 5th tick):** YOLO, HP read, projection, tracking, obs assembly, policy.

This mirrors the sim's own `dt=0.05` / `action_repeat=5` / `agent_dt=0.25` split. If perception
ever exceeds the 250 ms budget, lower the *decision* rate and retrain at a matching
`action_repeat` — do not skip perception ticks; odometry's per-frame shift bound breaks first.

## Movement: one held contact, idle is a position

The joystick floats — it teleports to touch-down, tracks the drag, homes to bottom-left on
release.

- **Touch down once per match at a chosen anchor. Never release mid-match.** Each decision is one
  MOVE event.
- **`move_bin == 0` (idle) moves the contact back to the anchor. It does NOT release.** Releasing
  homes the stick and forces a re-acquisition on the next non-idle decision. This is the single
  most tempting wrong simplification in the package.
- Anchor goes in the *centre* of the left region with clearance > saturation radius on all sides
  — not at the bottom-left rest position, which has no clearance down or left.
- One scalar `radius_px` is sufficient: the policy has no partial-speed action, so the stick only
  ever sits past saturation. Do not model the deadzone ramp. **Measured: the game clamps at 92 px**
  (commanding 140 gave 91.7 / 92.6 / 92.0 across three directions), so `radius_px = 110` sits 20%
  past the clamp — drift-proof without leaving the play area.

## Fire semantics: once per decision

`attack ∈ {0, 1, 2}` — nothing / attack / super. The tap fires on the **first** perception tick of
the decision window and does **not** repeat for the remaining four. This mirrors `env._held`,
which zeroes the fire column on sub-ticks 2..K. A held fire bit is a bug, not an optimization.

Apply action masking at inference the way `MaskablePPO` saw it in training — an uncharged super
means bin 2 is masked, not merely ignored.

Gadgets are not in the action space. Never touch the gadget button.

## Fail closed

Every failure — match over, focus lost, capture stall, odometry lost, policy exception — routes
to the same response: **release all contacts, stop emitting input, keep capturing.**

The interlock lives in `loop.py` and the control layer must *refuse* to emit when
`match_state.in_match` is false. It is not a politeness the caller is trusted to observe. A
stopped agent is recoverable by hand; an agent mashing a menu is not.

## Match detection: reuse gameplay.py, don't reinvent

`brawl_vision/gameplay.py` already solves this with measured margins (gameplay 0.649–0.790,
results screen 0.002–0.121, threshold 0.45). Its signal is the on-screen controls, which exist iff
you can act.

Live adaptation is only two changes:
- Calibrate once at startup from a temporal median, then cache. Per-frame is just `ring_score_at`
  at 1–3 anchors.
- Replace `longest_span` (which needs future frames) with **asymmetric hysteresis**. Exit must not
  be instant: `gameplay.py` documents that a Super detonating over the button drops the score for
  a few mid-match frames, and those frames are gameplay.

## Calibration is measured, never hardcoded

Same split `configs/vision.yaml` documents:
- **Settings** (numbers you argue about) → `configs/deployment.yaml`.
- **Measurements** (calibration outputs) → `brawl_deployment/data/*.json`, regenerated by a
  script, not hand-edited.

**The HUD layout is fixed for this project** (the BlueStacks/Mortis setup, matching
`tests/fixtures/vision/bluestacks-example.mp4`), so the two-layout hazard `gameplay.py` documents
— attack at (1487, 813) r≈65 vs (1752, 781) r≈47 across recordings — does not bite at runtime.
It still bites when reusing the *older iOS* fixtures for anything, and `data/hud_mask.json`
covers only one of those. Calibrate per setup anyway; reuse `gameplay.calibrate_buttons`.

## Two coordinate spaces, and they are not the same

- **Device pixels** — the Android screen's own 1920x1080. What `adb sendevent` addresses, fixed by
  the emulator, unrelated to the monitor. Tap targets and the joystick anchor live here.
- **Viewport pixels** — 2002x1126, what `brawl_vision` was calibrated at. Every ring score, every
  projection, every homography lives here.

`Calibration.button()` gives the first, `Calibration.viewport_button()` the second. **They
coincided exactly while the fixture happened to be 1080p** — a coincidence that passes every test
and then breaks the moment the display changes. Never pass one where the other is meant.

## Capture resizes to a VIEWPORT, never by a factor

`capture.DeployCapture` normalizes then resizes to `load_camera_model().viewport`. Two rules:

- **The target is 2002x1126, not the emulator's 1080p.** Both are downscales from a 1440p grab, so
  neither invents pixels, but only 2002x1126 lets the shipped homography, HUD mask and detector
  apply unchanged. MEASURED: the iOS calibration transfers to BlueStacks *unmodified* -- wall
  blocks land exactly on the 48 px tile grid across the whole frame, so the game's virtual camera
  does not depend on the device.
- **Resize the normalized frame to a viewport; do not multiply by a constant.** `detect_content_box`
  trims whatever bars the capture actually has (a fullscreen 1440p grab normalizes to 2558x1439,
  not 2560x1440), and a windowed emulator gives something else again. A viewport target absorbs
  all of it; a ratio does not.

## The world frame is `camera_relative + odometry.position_tiles`

Projection gives tile offsets **relative to the camera**, which follows the hero. Odometry gives
where the camera is. Every persistent quantity -- a track's position, a track's velocity, an
occupancy deposit -- must live in the sum, or the whole world appears to move whenever the hero
walks. MEASURED on live footage: hero screen-position regresses on odometry with slope +0.889
(the camera follows the hero, so relative position barely moves), an enemy with slope +0.165
(independent motion). A sign error would give ~+2.

**Copy the interlock from `occupancy.update`, do not invent a new one:**

- Deposit/track **only** when `odometry.status == "ok"`. An untrusted pose poisons the map and
  every track in it; coasting is cheap, a corrupted world model is not.
- **Reset on segment change.** `position_tiles` restarts at a cut, so tracks carried across one are
  tracks at fabricated positions.
- Hold the segment as `None` until the first tick, then *adopt* it. Seeding it to a sentinel like
  `-1` makes the first tick of every match a spurious reset -- it looks harmless and silently
  discards the opening frame's state.

## Tracking: associate in tiles, gate by roster speed

`perception/tracker.py`. Detections are per-frame and identity-free; `entities.rel_vel` and the
per-enemy slots the policy was trained on both need identity across frames.

- **Associate in world tiles, not pixels.** A pixel gate is a different physical distance at the
  top and bottom of a perspective frame, and cannot be set from anything the sim knows.
- **Gate = `GATE_NOISE_TILES + MAX_WALK_TILES_S * dt`** = `1.0 + 2.73 * dt` -- noise floor plus the
  roster's fastest walk (`configs/brawlers.yaml`), widened by the *actual* elapsed time so a late
  tick does not shed every track. Keep the two terms separate: at 4 Hz the noise term is 6x the
  motion term, and one combined constant hides which half is the problem.
- **Globally greedy, not per-track greedy.** Sort every (track, detection) pair inside the gate by
  distance and consume them in that order. Per-track order lets whichever track is enumerated first
  steal a detection that belongs to a nearer one, which is an identity swap and therefore a
  velocity spike.
- **No Kalman filter.** Constant-velocity predict, then gate. Position comes from the detection, not
  a blend; smoothing position is how a tracker starts lagging a dashing Mortis.
- **Slots are sticky.** The policy reads enemy k from a fixed column, so a track keeps its slot for
  life and only a retired track's slot is reused.
- `hp_detection/smooth.py` says in its own docstring that it is not a tracker and that `associate`
  is the function to replace. This is that replacement -- do not add a second one.

## Centres transfer across frame sources; radii do not

`ring_score_at` is sharply
radius-sensitive. Measured on the same button, same screen resolution, same HUD: the gadget fits
r=33.2 on an OBS recording's temporal median and r=39.9 on a raw ADB framebuffer grab, and scoring
one at the other's radius gives **0.083 instead of 0.984** — a 6.7 px error that flips the match
gate. Centres agree to 1–3 px.

So treat a stored calibration as authoritative for *where* and advisory for *how big*. Call
`match_state.MatchState.refine(frame)` once at startup on a frame known to show gameplay; it
rescans ±12 px for the score peak and raises if the peak is still sub-threshold, which means the
"known gameplay" assumption was wrong. **Never skip it because the stored numbers "were measured
properly"** — they were, on a different source.

**`normalize_viewport` crops and trims to 16:9 — it never rescales.** Every measured constant
downstream (`RADIUS_PX = (40, 115)`, `hud_mask.json` coordinates, the homography, ~77 px/tile) was
fit against ~1126 px-tall frames. Footage at a lower resolution does not degrade gracefully into
those constants; it misses them by the scale ratio. Check frame height before trusting any
calibrated stage, and reject rather than upscale — the function already raises for exactly this
reason.

## The observation spec the live loop targets is `configs/agent_obs_deploy.yaml`

Three sibling specs, each a narrowing of the last, for three different reasons:

| file | narrows | because |
|---|---|---|
| `agent_obs.yaml` | — | full information |
| `agent_obs_lowinfo.yaml` | enemy archetype, projectile type | a **type leak** — CV cannot name a silhouette |
| `agent_obs_deploy.yaml` | `entities.cubes`, `entities.can_attack`, `entities.dashing`, `hero.cubes` | nothing will **supply** them |

**Recoverability is not the criterion; intent to supply is.** `hero.cubes` is in that list even
though the pip row sits in the hero's own box stack next to the HP bar — the operator decided no
reader gets built for it, and a field nobody intends to fill is in the same position as one nobody
can. The one thing that is never an option is leaving it in the spec and feeding it zero.

**`entities.rel_vel` stays**, and the line is marked in the yaml. It is recoverable
(`perception/tracker.py`), the odometry error cancels in `ent_vel - hero_vel`, and an
inference-time ablation could not detect the trained policy using it at all — 1.3 pp against a
2.6 pp standard error. It is kept because it is free, not because it was shown to matter.

**Deploy also SWAPS two fields rather than dropping them: `hp_frac` → `hp`, on both sides.** A
fraction needs max_hp, and `cubes.hp_per_cube` adds a flat +400 to it for the hero and for every
enemy, with nothing on screen saying who holds how many. The kit constant in `configs/brawlers.yaml`
is right only until the first cube. `hp_detection/read.py` produces a NUMERAL and deliberately
ships no fill ratio, so absolute HP is what the pipeline already supplies — the swap needs no new
reader and does not change the spec's width.

**If you add a field to any spec, check its units have a divisor.** `obs_select._norm_divisor`
covers `tiles`, `tiles/s`, `hp` and `count`; everything else passes through `normalize: true`
untouched. That is fine for `fraction`/`bool`, and was NOT fine for `hp` (up to 20500) or `count`
(`meta.n_enemies_alive`, up to 9) — both were sitting raw in a normalized vector until the hp_frac
swap drew attention to the table. `tests/test_configs_files.py` now asserts the property against
the schema, so a new raw field fails rather than trains.

**Each of these is a from-scratch retrain, not a checkpoint swap.** Narrowing changes the
observation width and with it `BrawlFeaturesExtractor`'s first linear layer. And when a spec must
change, **add a file — never edit one in place**: finished runs name their spec by path in their
own `train.yaml`, so an in-place edit silently makes an existing checkpoint unreproducible.
`tests/test_configs_files.py` pins each file's relationship to its parent in both directions.

## `meta.time_frac` is wall-clock ÷ 150 s, clamped to [0, 1]

`core/observation.py:434` defines it as `step_count / max_episode_steps`, and
`configs/default.yaml` sets `max_episode_steps: 3000` at `dt: 0.05` — 150 seconds. So the
deployed value is seconds since the match gate went true, over 150.

**The clamp is the part that matters.** Running short is harmless: the policy just never sees the
top of the range, exactly as when a sim episode ends early. Running LONG is out-of-distribution —
the sim truncates at `time_frac == 1`, so the column's entire training range is bounded by 1 and a
200-second match would feed it 1.33.

## The `self` group is proprioception, not perception

`configs/agent_obs_lowinfo.yaml`'s rule — a field is admitted only if CV can recover it — was
written for `enemies` and `projectiles`, where the agent is a spectator. **It does not bind the
`self` group, because the deployed loop issues the actions.**

`hero.facing_vec`, `dashing`, `dash_t`, `dash_dir`, `long_dash_ready`, `long_dash_frac`,
`attack_cd` and `invuln` are all deterministic functions of our own action history plus the kit
constants in `configs/brawlers.yaml`. Verified: `movement.py:54` and `hero.py:276` set
`ent_facing` from `move_dir`/`dash_dir`; `hero.py:124` defines `long_dash_ready` as a stopwatch
since our last attack; `hero.py:114-115` tick `attack_cd`/`invuln_t` as plain `-= dt` countdowns.

So maintain a **shadow hero state** — the hero's own timers, ticked at `cfg.dt`, advanced by the
actions we emit, loading the same brawler params the sim loads. No CV, no retrain, exact values.

**This is load-bearing for Mortis specifically** (the MVP brawler): dash and long-dash are his
entire kit, so getting this wrong guts the columns his policy leans on hardest.

**The risk is drift, not absence.** The shadow desyncs when an action does not land as modelled —
a dropped input, or a dash cut short by a wall. Guard it:

- **`ammo` is the canary** — readable from the pip bar *and* dead-reckonable. One comparison per
  decision, and a mismatch is the earliest available desync signal.
- **Position** is likewise both CV-tracked and predicted; divergence means movement is not landing.
- On desync: resync observables from CV, reseed unobservable timers **conservatively** (assume not
  ready), log it. Sustained desync is a fail-closed condition.

**Never feed a constant or zero for a field you cannot supply.** A policy trained on a real value
and deployed against a fabricated one is being lied to in a column it learned to trust; the
failure is silent and looks like "the policy is bad at real Brawl Stars."

## Resources: hunt unbounded growth, not microseconds

Target is one agent on 16 GB VRAM / 32 GB RAM. Steady state is comfortable; growth is the risk.

- **Never retain frame history.** 1920×1080 BGR at 20 Hz is ~124 MB/s — a three-minute match is
  ~22 GB. Frame dumping is opt-in, capped, off by default. A ring buffer of ≤40 frames is the only
  standing retention.
- **Bounded telemetry.** Scalars to a fixed-size ring or append-only CSV. Never per-frame images
  or tensors.
- **No parallelism.** One capture, one model set, one policy. No vectorised env, no worker pool,
  no async prefetch. Single-agent is the requirement, not a limitation to design around.
- Measure memory process-scoped. Do not use `nvidia-smi` for this.

## Testing

The vision suite is siloed and fast — `pytest tests/test_vision_*.py -m "not vision"` runs in
~11 s, against ~13 min for the full suite (95% of which is `brawl_sim`). Deployment tests should
join that siloed set: no live emulator, no GPU, fixtures over captures.

Anything needing a real BlueStacks belongs behind a marker and is run by hand.
