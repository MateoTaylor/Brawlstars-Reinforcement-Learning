# brawl_deployment — closing the loop into BlueStacks

Design document. **§4 (control), §5 (match gate), the capture path, §6.1 (entity AND projectile
tracking), §6.2's grid group, all three readers of §6.4 and §6.3's shadow hero state are built and
tested** — `brawl_deployment/{capture,match_state}.py`, `control/`, `perception/`,
`brawl_vision/object_detection/hp_detection/hero_bars.py`, `brawl_vision/hud.py`,
`tests/test_deployment_{control,tracker,projectiles,shadow,grid}.py` +
`tests/test_vision_{hero_bars,hud}.py` (222 passed).
Injection is confirmed live in the training ground (§4.1); the vision stack is confirmed to
transfer to BlueStacks unmodified (§9.4); both detectors run on the GPU and a 20 Hz tick measures
36.2 ms against its 50 ms budget (§7.1). **Every field in the observation spec now has a supplier** — every CV one has a reader and
every proprioceptive one has the shadow, which is checked field-for-field against a live
`BrawlVecEnv`. The policy that will consume them exists too: `runs/mortis_deploy-20260907-041522`
trained 300M steps against `configs/agent_obs_deploy.yaml`. **`assemble.py` is BUILT and
proven against the sim** (§6.6) — `tests/test_deployment_assemble.py` runs a real
`BrawlVecEnv`, feeds the assembler the same frame through the deployed suppliers' interfaces,
and asserts the result equals `obs_select.build_agent_obs`'s column-for-column, on both deploy
specs. **`policy.py` is BUILT too** (§6.7): the checkpoint loads behind four guards against
being paired with the wrong observation, and runs on the CPU at 0.96 ms/decision. **Still
design: `loop.py`, `window.py` and `config.py` — the loop has no driver.**
**The `zone` group's supplier is decided (§9.14, §9.15, §9.16).** Two ablation passes price the
whole group at 10.4 pp and every realistic degradation of it — a countdown pinned to zero, margins
stale by six seconds, margins clamped to what the camera reaches — at 0–4 pp against a 1.5 pp
standard error. **Operator's call: deploy on the current checkpoint**, supplying `hero_margin` as
local directional distances, `next_shrink_in` pinned, `safe_area_frac` lived with. The retrain
stays a lever rather than a prerequisite, and the sim side of it is now built and waiting:
`zone.hero_margin_local` plus `configs/agent_obs_deploy2.yaml` (§9.16) train the next model on the
shape the live loop can actually produce, so pulling that lever is a run, not a design task.
The GPU blocker is gone: ONNX Runtime now runs both detectors on CUDA at 3.6–4.4× the CPU
numbers, after four separate faults (§9.12). Sibling of
`BRAWL_SIM_BUILD_PLAN.md` and
`Terrain_Perception_Build_Plan.md`; read `CONVENTIONS.md` first for the rules that govern
`brawl_sim` and, explicitly, **do not** govern this package (see §1.3).

**Scope.** One agent, one BlueStacks instance, Nulls Brawl friendly battles against bots.
Capture the screen, perceive, decide, inject touch input, detect when the match ends and stop.
**Out of scope for the MVP:** queueing matches, menu navigation, brawler selection, gadget use,
multi-instance, any form of parallelism.

---

## 0. The finding that should shape the schedule

The wiring described in the prompt — "check BlueStacks, screenshot into vision, vision into
agent, actions into BlueStacks" — is the *easy half*. Two things are harder than they look, and
one thing is already done.

**Already done (at design time):** match-over detection. `brawl_vision/gameplay.py` exists and
solves exactly this problem, with measured margins. It scores a ring at the attack-button
anchor; gameplay measures 0.649–0.790 and the results screen measures 0.002–0.121. That is a
2.3× gap either side of the 0.45 threshold. §5 is mostly "call the thing that already works,
live instead of offline."

**Harder than it looks #1 — multi-touch.** The agent must hold a movement contact *and* tap
attack in the same decision. A mouse has one contact point. This single constraint eliminates the
most obvious input backend and forces the choice in §4. It was the highest-risk unknown in the
design, was spiked first, and is now **closed**: the emulator's touch device takes 16 concurrent
slots without root, and the game acts on injected events — verified live, movement contact held
through four attack taps (§4.1).

**Harder than it looks #2 — entity tracking, not the hero's own state.** The `self` group looks
alarming at first read: it asks for `hero.facing_vec`, `dash_t`, `long_dash_frac`, `invuln` —
none of which a screenshot shows. But the deployed loop *issues the actions*, and every one of
those fields is a deterministic function of our own action history plus the kit constants already
in `configs/brawlers.yaml`. A shadow hero state supplies them exactly, with no CV. See §6.3,
which corrects an earlier and more pessimistic reading of this.

What is genuinely hard is the other direction: **enemy and projectile tracking**, where we have
no privileged information and must recover identity and velocity from anonymous per-frame
detections. §6.1.

Recommended order: ~~§4 spike~~ **done** → ~~§5~~ **done** → ~~§6.1~~ **done** → ~~§6.3 shadow
state~~ **done** → §6.2 grid → `loop.py`.

---

## 1. Architecture

### 1.1 The loop

```
                       BlueStacks (fullscreen, one monitor)
                                    │
                    ┌───────────────┴───────────────┐
                    │  20 Hz                        │  4 Hz
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │ PERCEPTION TICK       │       │ DECISION              │
        │                       │       │                       │
        │ ScreenCapture.grab    │       │ YOLO detector         │
        │ normalize_viewport    │       │ hp_detection          │
        │ terrain odometry      │       │ project.to_tiles      │
        │ occupancy deposit     │       │ tracker.update        │
        │ grid.observe_zone     │       │ shadow.check_ammo     │
        │ projectiles.update    │       │ grid.build            │
        │ ring_score (in-match) │       │ assemble obs          │
        │ shadow.advance        │       │ policy.predict        │
        └───────────┬───────────┘       │ shadow.act            │
                    │                   └───────────┬───────────┘
                    │ camera pose,                  │ action (move_bin, attack)
                    │ occupancy grid,               ▼
                    │ in_match flag         ┌───────────────────┐
                    └──────────────────────▶│ CONTROL           │
                                            │ joystick offset   │
                                            │ button taps       │
                                            │ input backend     │
                                            └─────────┬─────────┘
                                                      │ touch events
                                                      ▼
                                              BlueStacks / Android
```

### 1.2 Two rates, and why it is not one

**Capture and odometry run at 20 Hz. Decisions happen at 4 Hz, on every 5th frame.**

This is not a performance hedge, it is forced by an existing measured constant.
`configs/vision.yaml` sets `odometry.max_shift_tiles: 2.0` and its own comment states the bound
is **per frame, not per second**, sized against Mortis's charged dash at 0.89 tiles/frame *at 20
Hz*. Capture at 4 Hz and the same dash moves 4.45 tiles between frames, blowing through the gate;
every frame of a dash reads as a cut and camera tracking drops. The comment says so directly:
"raise it if the capture rate drops."

So the loop captures at the sim's *tick* rate and decides at the sim's *decision* rate, which is
the same 20 Hz / 4 Hz split `brawl_sim` already runs internally (`dt=0.05`, `action_repeat=5`,
`agent_dt=0.25`). The deployed loop mirrors the trained one rather than inventing a new cadence.

The cheap stages run at 20 Hz (odometry ~7.5 ms/frame measured, ring score is a few hundred
microseconds at 1–3 anchors). The expensive stages — YOLO, HP reading, projection, assembly,
policy — run at 4 Hz. Held actions between decisions are exactly the sim's `_held` semantics:
movement persists, the fire bit does not repeat.

`shadow.advance` sits on the 20 Hz side because it is the thing that *keeps* those two rates
honest: it takes the real elapsed seconds and spends them as whole `dt` sub-ticks, so a decision
that ran long is a decision that consumed more of them rather than one that quietly desynced the
hero's cooldowns from the game's. §6.3.

`grid` is split across both rates for a reason worth naming. `grid.observe_zone` is a 20 Hz
deposit because the gas map accumulates and a skipped frame is evidence thrown away, while
`grid.build` is a 4 Hz read because the grid is only ever consumed by a decision. That is the same
split `occupancy` already uses, and it is why `GasMap` holds state and `GridBuilder.build` is a
pure function of it. §6.2.

### 1.3 What `CONVENTIONS.md` does and does not govern

`brawl_vision/__init__.py` already draws this boundary and `brawl_deployment` inherits its side
of it verbatim: **the leading-`(N,)` batch rule and the no-host-sync rule do not apply here.**
There is one screen, one match, one agent. `.item()`, `.cpu()`, numpy, and OpenCV's numpy-in /
numpy-out contract are all fine. Do not contort this code to look batched.

What *does* carry over, because it is what lets the halves speak: **units** (tiles, seconds,
radians), **coordinates** (`pos = (x, y)`, x = column, y = row, **y increasing downward**), and
`brawl_sim.constants.Tile` imported rather than redeclared.

The y-down convention is load-bearing in §4.3 and is the single easiest thing to get silently
backwards.

### 1.4 Package layout

```
brawl_deployment/
  __init__.py           # the boundary docstring, as brawl_vision has
  config.py             # DeploymentConfig <- configs/deployment.yaml
  window.py             # locate BlueStacks, verify geometry, resolve monitor index
  loop.py               # the 20 Hz / 4 Hz driver; owns the safety gate
  match_state.py        # in-match detection + hysteresis (wraps gameplay.py)
  policy.py             # checkpoint load, action masking, predict                   BUILT
  perception/
    __init__.py
    tracker.py          # detections -> index-stable slots + velocity            BUILT
    projectiles.py      # projectile boxes -> velocity + time-to-closest, 20 Hz   BUILT
    shadow.py           # dead-reckoned own-state, checked against the CV readers      BUILT
    grid.py             # occupancy + tracks -> the (8, 13, 21) view grid              BUILT
    assemble.py         # -> obs dict matching the deployed spec, via obs_select      BUILT
  control/
    __init__.py
    backend.py          # InputBackend protocol + a NullBackend for dry runs
    adb.py              # persistent-contact sendevent backend
    keymap.py           # Win32 SendInput keystroke backend
    joystick.py         # move_bin -> anchor-relative contact point
    buttons.py          # attack / super tap geometry
    calibration.py      # anchor, saturation radius, button coords -> data/*.json
  data/
    control_calibration.json    # MEASURED, regenerated not edited
```

Same settings-vs-measurements split `configs/vision.yaml` documents: tunables you argue about go
in `configs/deployment.yaml`, outputs of a calibration script go in `brawl_deployment/data/*.json`.

---

## 2. What already exists and translates directly

Verified present in the tree, not assumed:

| Need | Module | Status |
|---|---|---|
| Live screen capture | `brawl_vision/capture.ScreenCapture` (mss) | **Done.** `open_source("screen:N")` already returns it. |
| Viewport normalization | `capture.detect_content_box` / `normalize_viewport` | **Done.** Measured, handles the Showdown 16:9 constriction. |
| Camera tracking | `terrain/odometry.py` | **Done.** ~7.5 ms/frame. |
| Tile map | `terrain/occupancy.OccupancyMap`, `classifier.py` | **Done.** Feeds the `grid` obs group. |
| Gas / zone | `terrain/zone.py` | **Done.** |
| Entity boxes | `object_detection/detector.ObjectDetector` (YOLO) | **Done.** |
| Screen box → tile | `object_detection/project.to_tiles` | **Done**, with a documented anchor offset. |
| HP bars | `object_detection/hp_detection/` | **Done.** |
| Projectiles | `object_detection/projectile_detection/` | **Done.** |
| **Match-over detection** | `gameplay.ring_score_at` | **Done offline**, needs a live adapter (§5). |

That is a genuinely large amount of the perception stack. The gap is not perception primitives —
it is **fusion into the exact obs vector the policy expects**, plus everything in `control/`.

---

## 3. The in-match gate (§5 detail deferred here for flow)

Covered in §5. Summarised: it works, it is cheap, and it is the safety interlock for everything
in §4.

---

## 4. Control — the hard part

### 4.1 The multi-touch constraint kills the obvious backend

The agent must move *and* attack simultaneously. That is two concurrent touch contacts. Ranking
the candidates against it:

| Backend | Concurrent contacts | Held contact | Latency | Verdict |
|---|---|---|---|---|
| Win32 `SendInput` **mouse** | **1** | yes | lowest | **Rejected.** Cannot move and shoot. |
| ADB `input tap` / `input swipe` | 1 per subprocess | **no** | 50–200 ms/call | **Rejected.** A subprocess per action at 4 Hz, and `swipe` cannot hold. |
| ADB `sendevent` over a **persistent** `adb shell` | yes (multi-slot) | **yes** | low once open | **Candidate A.** Full 16-direction resolution. |
| BlueStacks keymapping + `SendInput` **keystrokes** | yes (BlueStacks synthesises slots) | yes | lowest | **Candidate B.** But see the resolution problem below. |

**Candidate B's resolution problem.** BlueStacks keymapping binds keys to a synthesised joystick,
and a WASD-style binding yields **8 directions**. The policy emits `n_move_bins: 16`. Deploying
on B means either halving movement resolution or retraining at `action.n_move_bins: 8` — the
latter is a clean config change but **breaks checkpoint compatibility** (the action space
changes), so it is a decision to take before the next training run, not after.

**RESOLVED — Candidate A, and the hybrid turned out to be unnecessary.** Measured on this
emulator:

```
touch device        /dev/input/event4   (the only one exposing ABS_MT_SLOT + ABS_MT_POSITION_X)
concurrent slots    16                  (ABS_MT_SLOT max 15)
axis range          0..32767            (NOT pixels -- scaled from a 1920x1080 screen)
permissions         writable by uid 2000(shell), group 1004(input) -- NO ROOT NEEDED
android screen      1920x1080           (matches the capture plan exactly)

persistent adb shell, bare round trip      0.51 ms   (p95 0.65)
four real process spawns on device         5.77 ms   (p95 5.94)   <- one joystick MOVE
adb exec-out screencap                   ~290 ms                  <- rejected as a frame source
mss grab, 2560x1440                       18.1 ms    (p95 19.3)   <- the capture path
```

The device natively supports 16 concurrent contacts, so **movement and firing coexist without any
keymap involvement at all**. Candidate C's open question is moot; there is no second input path to
reconcile. 16-direction movement is preserved, which means **no `n_move_bins` change and no
retrain**.

**One implementation detail that the measurements forced.** The natural fast path — writing packed
24-byte `input_event` structs straight into the device — does not work here: BlueStacks ships
**adb 1.0.36**, whose Windows stdin forwarding fails on binary payloads (`OSError 22` on first
flush, device-side shell still alive). Text `sendevent` commands through the same persistent pipe
work fine, and at 5.77 ms per joystick update against a 250 ms decision budget (2.3%) the binary
path would buy nothing worth a platform-tools dependency. Revisit only if the budget tightens.

**VERIFIED: the game honours injected touches.** Everything above is transport — it says the
events reach the kernel, not that Nulls Brawl acts on them, and some titles ignore synthetic
input. Checked live in a training ground, reading the result back off the framebuffer:

```
the floating stick teleported to OUR anchor (345.6, 669.6), not to the thumb rest
Mortis walked on the commanded bins, and reversed when the bin reversed
four attack taps landed: "Damage per second" went 0 -> 2000 and a dummy died
THE MOVEMENT CONTACT STAYED DOWN THROUGH ALL FOUR TAPS   <- the multi-touch premise, live
```

The last line is the one that mattered. §4.1's whole backend argument rests on movement and
firing coexisting; the slot count proved the device allows it, this proved the *game* allows it.

### 4.2 The floating joystick: persistent contact, not per-decision gestures

The stick teleports to wherever the contact lands on the left side, tracks the drag, and homes to
bottom-left on release. Two ways to drive it:

**Rejected — tap / drag / release per decision.** At 4 Hz that is a fresh gesture every 250 ms.
Each one pays a touch-down, and the game's own stick-acquisition behaviour sits inside the
decision budget. A brief release also homes the stick, which means the *next* touch-down is
re-acquiring from scratch. Worse, a short contact risks being read as a tap rather than a drag.

**Chosen — one contact, held for the whole match.** Touch down once at a chosen anchor when the
match starts. From then on, every decision is a single MOVE event to a new position. Never
release until the match ends.

Three properties make this work, and they follow from the floating behaviour rather than fighting
it:

1. **We choose the anchor.** Because the stick teleports to touch-down, the resting geometry is
   ours to pick. Put the anchor in the *centre* of the left region with clearance greater than
   the saturation radius in all four directions — not at the bottom-left rest position, which has
   no clearance down or left. Proposed starting point: `(0.18·W, 0.62·H)` = **(345.6, 669.6)**,
   which worked live, unmodified — the stick teleported there and stayed inside the play area
   through every direction.
2. **Direction changes cost one event.** No release, no re-acquire, no homing.
3. **Idle is a position, not a release.** `move_bin == 0` (the sim's explicit idle bin) maps to
   *move the contact back to the anchor*, inside the deadzone. Releasing to express idle would
   home the stick and force a re-acquisition on the next non-idle decision — the exact cost this
   design exists to avoid.

**Recovery.** If the contact is lost (backend hiccup, focus change, the game rejecting it), the
loop must detect it and re-acquire rather than silently walking nowhere. Cheapest detector: the
resting-position stick is visible at bottom-left when *no* contact is held, so a ring/blob score
at the bottom-left rest anchor is a direct "did my contact drop" signal, reusing the same
`ring_score_at` machinery as §5. Worth building; not MVP-blocking.

### 4.3 Bin → screen offset

The agent's action is `(move, attack)` where `move ∈ [0, 16]` and `attack ∈ {0, 1, 2}`
(`brawl_sim/config.py:108` → `action_nvec = (n_move_bins + 1, 3)`).

From `hero.decode_action` and `geometry.dir_from_bin`:

```
move == 0                  ->  idle
move >= 1  ->  bin = move - 1
               theta = bin * 2*pi / 16
               dir   = (cos(theta), sin(theta))
```

**The world's y increases downward. Screen y increases downward. They agree — there is no flip.**

```python
# control/joystick.py
def contact_point(move: int, anchor: tuple[float, float], radius_px: float, n_bins: int = 16):
    """Screen-pixel contact position for one action's move component.

    move == 0 returns the anchor itself: idle is the contact resting in the deadzone, NOT a
    release. See BRAWL_DEPLOYMENT_DESIGN.md 4.2 for why releasing to idle is a trap.
    """
    if move == 0:
        return anchor
    theta = (move - 1) * (2.0 * math.pi / n_bins)
    # No y negation. brawl_sim's world y and the screen's y both increase downward
    # (brawl_vision/__init__.py's coordinate note). Negating here is a silent 100% failure
    # that looks like "the agent walks into walls".
    return (anchor[0] + radius_px * math.cos(theta),
            anchor[1] + radius_px * math.sin(theta))
```

**Why a single radius is enough.** The policy has no partial-speed action — every non-idle bin is
full speed. So the stick only ever needs to sit *past saturation*. We do not need to model the
deadzone-to-saturation ramp at all; we need one number, `radius_px`, comfortably above saturation
and within screen bounds. That collapses joystick calibration from a curve to a scalar.

**MEASURED: saturation is 92 px, and the directions came out exactly as derived above.** Commanding
a 140 px deflection and reading the knob's own position back off the framebuffer:

```
action 1  -> bin 0  -> ( +91.7,  -0.1 )   right
action 9  -> bin 8  -> ( -92.6,  -1.6 )   left
action 3  -> bin 2  -> ( +64.4, +65.6 )   45 deg DOWN-right
action 0  -> idle   -> ( +1.1,   +0.3 )   back at the anchor, CONTACT STILL DOWN
```

Two things fall out. The game clamps at ~92 px, so `radius_px` is set to **110** — 20% past
saturation, enough that anchor or UI-scale drift cannot fall short of full speed, tight enough
that the reachable circle clears the bottom UI strip. And bin 2 landing *down*-right rather than
up-right is **the no-y-flip convention confirmed empirically**, not just derived. Had the negation
been there, this is the measurement that would have caught it.

### 4.4 Buttons

Attack and super are fixed-position binary taps in the bottom right, and the prompt is explicit
that no aiming is needed — a tap fires in the game's default direction. `attack == 1` taps
attack; `attack == 2` taps super. **Gadgets are not emitted** (the policy has no gadget action)
and their button is never touched.

The sim's fire semantics carry over exactly: **one decision means at most one attack attempt.**
The tap fires on the first perception tick of the decision window and does not repeat for the
remaining four — mirroring `env._held`, which zeroes the fire column on sub-ticks 2..K.

Action masking must be applied at inference the same way `MaskablePPO` saw it in training: if
super is not charged, bin 2 is masked out. `hero.action_mask` is the sim-side reference; the
deployed equivalent reads super readiness off the HUD (§6.3).

### 4.5 Calibration

`control/calibration.py`, run once per resolution/UI-scale, output to
`brawl_deployment/data/control_calibration.json`, regenerated rather than hand-edited:

- `anchor` — chosen joystick touch-down point.
- `radius_px` — saturation radius (§4.3).
- `attack`, `super` — button centres and radii. **Reuse `gameplay.calibrate_buttons`**, which
  already finds these from a temporal median via `HoughCircles` + `ring_score_at` refinement, and
  already handles the fact that this game ships two different HUD layouts.
- `rest_anchor` — bottom-left resting stick position, for the contact-loss detector (§4.2).

**Radii do not transfer across frame sources; centres do.** This was found the hard way, live.
The same gadget button fits `r = 33.2` on the OBS recording's temporal median and `r = 39.9` on a
raw ADB framebuffer grab — and `ring_score_at` is sharp enough that scoring one at the other's
radius gives **0.083 instead of 0.984**. That is a 6.7 px error turning "clearly in a match" into
"clearly not". The centres agree to 1–3 px across both sources.

The encoded video's temporal median softens the button edge and pulls the best-fit annulus inward;
the raw framebuffer keeps it crisp. `gameplay.py` already noted the sensitivity in passing
("Hough's radius estimate is loose enough to cost 0.3 of score") — this is the same effect, larger,
and it bites across *sources* rather than across Hough candidates.

So the stored calibration is authoritative for **where** a button is and advisory for **how big**
it looks. `match_state.refine_radius()` rescans ±12 px for the score peak, and
`MatchState.refine(frame)` applies it once at startup on a known-gameplay frame, raising if the
peak is still sub-threshold (wrong resolution, a menu frame, a changed HUD). Verified: an ADB grab
goes 0.083 → 0.920, and the OBS median goes 0.818 → 0.818, i.e. a safe no-op on the source the
numbers were measured on.

**The two-HUD-layout hazard applies here directly.** `brawl_vision/gameplay.py` documents it: the
2026-08 clips put attack at (1487, 813) r≈65, while four of sixteen recordings put it at
(1752, 781) r≈47 — the game's own UI-scale setting. `data/hud_mask.json` records only the first.
A button table calibrated on one layout **taps empty screen on the other**, silently. This is why
calibration is measured per-setup and cached, never hardcoded.

---

## 5. Match-over detection

**Reuse `brawl_vision/gameplay.py`.** Its signal is the on-screen controls, which exist if and
only if you can act — absent on the loading screen, in the lobby, during the intro swoop, the
instant "Defeated" appears, and on the results screen.

**Validated on the real deployment footage** (`tests/fixtures/vision/bluestacks-example-new.mp4`,
1920×1080 @ 30 fps, BlueStacks fullscreen, Mortis, sampled every 5th frame):

```
                        attack anchor      gadget anchor
  pre-match / loading   0.000 - 0.198      0.001 - 0.078
  gameplay              0.582 - 0.593      0.776 - 0.823
  post-match            0.000 - 0.198      0.001 - 0.078
                        ---- 3.0x gap      ---- 10x gap
GAMEPLAY_THRESHOLD = 0.45  (unchanged; sits cleanly inside both gaps)
```

**Anchor the live gate on the GADGET (skull) button, not the attack button.** Both separate
cleanly, but the gadget's margin is far better balanced: 5.8× below threshold and 1.8× above,
against attack's 2.3×/1.3×. The attack button reads lower here than in the iOS footage
(0.585 vs 0.649–0.790) because Nulls Brawl draws a bright filled disc where the iOS layout drew a
dark annulus, and `ring_score_at` is measuring radial gradient either way. Still well clear of
0.45, just with less headroom than the gadget.

Convenient side effect: the gadget button is the one the policy never presses, so nothing the
agent does can perturb its own in-match signal.

**One constant blocks auto-calibration: `RADIUS_PX = (40, 115)`.** `calibrate_buttons` returns
**zero anchors** on this footage — not a scoring failure but a search-range one. The measured
radii are attack ≈ 35, gadget ≈ 33, super ≈ 46, so the two best buttons are never proposed to
`HoughCircles` in the first place. Re-running the identical search at `minRadius=20` finds them
and they score 0.587 and 0.818 against `MIN_CALIBRATION_SCORE = 0.60`. Lowering the bound to ~25
fixes it.

That said — **the HUD is fixed for this project, so runtime calibration is unnecessary.** Measure
the anchors once, store them in `brawl_deployment/data/control_calibration.json`, and skip Hough
entirely in the loop. Per-frame cost is then three `ring_score_at` evaluations. Measured anchors,
1920×1080:

```
attack  (1676.5,  999.5)  r 35.1
gadget  (1560.5,  901.5)  r 33.2      <- in-match gate anchor
super   (1462.5, 1000.5)  r 46.5
```

**Offline → live adaptation.** `scan_gameplay` walks a whole recording and takes the longest
gap-filled run; live has no future frames. The adaptation:

- **Calibrate once at startup** from `median_frame` over N frames captured while the user is in a
  match, then cache. Per-frame cost afterwards is `ring_score_at` at 1–3 anchors — microseconds,
  affordable at 20 Hz.
- **Replace "longest run" with hysteresis.** `longest_span`'s job was rejecting flicker in both
  directions; live, the two directions have very different costs. Entering gameplay wrongly means
  spraying inputs at a menu; exiting wrongly means standing still for a beat. So: **enter** needs
  a sustained run above threshold, **exit** is fast but not instant.
- The reason exit cannot be instant is documented in `gameplay.py` itself: *"a Super detonating
  over the button drops the score for a few frames in the middle of a match, and those frames are
  gameplay."* Require K consecutive sub-threshold samples. Suggested starting point: enter on 10
  consecutive (0.5 s at 20 Hz), exit on 6 (0.3 s), both tuned against a real clip before trusting.

**The safety gate.** On exit: release every contact, stop emitting input, keep capturing. This is
what stops the agent hammering the end-of-round menu. It should be a hard interlock in `loop.py`
— the control layer refuses to emit unless `match_state.in_match` is true — not a politeness the
caller is trusted to observe. It should also fire on: window focus lost, capture stall, and
policy exception. **Fail closed.**

---

## 6. Perception → observation

### 6.1 Tracking — BUILT

`brawl_deployment/perception/tracker.py` (entities, 4 Hz) and `perception/projectiles.py`
(projectiles, 20 Hz), with `tests/test_deployment_tracker.py` (19 tests) and
`tests/test_deployment_projectiles.py` (32 tests) — no emulator and no model weights, synthetic
detections through a stub plan.

Detections are per-frame and anonymous; the obs spec needs identity. `enemies` slots are
**index-stable within an episode**, and `entities.rel_vel` requires association across frames.

Nearest-neighbour association in *tile* space (not pixels — tile space is camera-motion-corrected
by odometry, so a moving camera does not look like moving enemies) with a gating radius, plus a
short coast for missed detections. Constant-velocity predict-then-gate, globally greedy rather
than per-track greedy, no Kalman filter: with ≤10 entities and a gate that must be wider than the
step anyway, a filter's covariance would carry less information than its parameters imply.

**The world frame is `camera_relative + odometry.position_tiles`** — taken from
`occupancy.update`, which is tested and does exactly this, rather than re-derived. Confirmed
independently by regressing each track's world displacement on the camera's: the flipped sign
would give a slope of +2, and the measurement gives **+0.889 for the player** (the camera follows
the hero, so ~1:1, exactly as it should) and **+0.165 for enemies** (independent motion, ~0).

The tracker inherits odometry's own interlock for the same reason `occupancy.update` has it: a
tick admitted on an `uncertain` estimate would displace every track by the odometry error at
once, which reads as every entity accelerating together. `lost`/segment change drops all tracks,
since a new segment has no defined offset to the old one.

#### Measured

**Precision floor, against real ground truth.** The training-ground dummies never move. Tracked
over 8 s: position sd **0.00–0.05 tiles in x, 0.01–0.19 in y**; speed median **0.01–0.03
tiles/s**. So the detector-plus-projection chain is far more precise than
`detector.anchor_frac`'s impossible-cell table suggested — that table was measuring the *hard*
cases. (The training ground's camera is clamped — it panned 0.01 tiles in x across 8 s of
driving — which makes it an excellent precision rig and a useless one for camera-motion tests.)

**On the moving-camera fixture** the absolute speeds look much worse — median ~2.4 tiles/s, but
34% (hero) and 38% (enemies) past 2.73 tiles/s. Two things make that number smaller than it
first appears, and both were established by measurement rather than argument.

**The bound itself is wrong for some of those samples.** 2.73 is the roster's fastest *walk*;
Mortis dashes at 8.9 tiles/s and long-dashes at 17.8. The 9.57 tiles/s maximum in that data is
squarely in dash range, so an unknown share of the exceedances are correct readings of a real
movement ability, not error.

**And what error there is largely does not reach the policy.** `entities.rel_vel` is
`ent_vel - hero_vel`, and `update` builds both from `camera_relative + position_tiles` using the
*same* odometry sample at the same tick — so the odometry term cancels exactly in the difference.
Measured on `bluestacks-example-new.mp4`, 92 enemy observations over 209 decision ticks:

| quantity | median | p90 | past its physical bound |
|---|---|---|---|
| hero absolute speed | 2.40 | 3.22 | 34% (of 2.73) |
| enemy absolute speed | 2.45 | 3.51 | 38% (of 2.73) |
| **`rel_vel`** | 2.41 | 4.43 | **8%** (of 5.46) |

The 8% residual is projection-anchor noise, which is per-entity and genuinely does not cancel.
The rest was common-mode camera error that the subtraction removes.

**Where the odometry error does land is `hero.vel`** — an absolute velocity, and one §6.3 already
has a better source for: dead-reckoning from the movement we command. Take it from the shadow
state, not from the tracker.

**And the policy turns out not to care about `rel_vel` at all.** `TierEvaluator`, 300 episodes ×
4 tiers on identical reseeded scenarios, perturbing only the column the policy reads:

| condition | overall win rate | Δ |
|---|---|---|
| baseline | 0.680 | — |
| zeroed | 0.667 | −0.013 |
| gaussian sd 0.5 tiles/s | 0.672 | −0.008 |
| sd 1.0 | 0.659 | −0.021 |
| sd 2.0 | 0.677 | −0.003 |
| sd 4.0 | 0.678 | −0.002 |

Every delta is inside the ±2.6 pp standard error, there is no monotone trend, and **zeroing the
field entirely is as cheap as leaving it alone**. Two conclusions: deployment carries no risk from
`rel_vel` noise even at several times the measured level, and the column is a free candidate to
drop alongside the three in §9.8. What this does *not* establish is that a run trained without it
would score the same — ablating at inference and never training on the field are different
experiments.

`n_enemies_alive` is available independently from the "Brawlers left: N" HUD text, which remains
a useful cross-check on the tracker.

#### The projectile tracker is a separate class, and the reasons are measurements

`projectiles.vel` and `projectiles.time_to_closest` have no supplier without one —
`projectile_detection/detect.py` says so itself, that turning a sequence of boxes into a velocity
is *"a layer above that does not exist yet"*. It now exists, and it is not `EntityTracker` with
different constants: the rate differs (20 Hz vs 4 Hz, §9.5), there are no index-stable slots to
protect so none of the promotion machinery applies, and a coasted projectile is a phantom threat
where a coasted entity is merely stale. Only `_greedy_match` is shared.

Three constants were measured rather than argued, all on the real pipeline (rectify → odometry →
detect → `to_tiles`), and two of them overturned the obvious guess.

**The gate.** 101 consecutive pairs on `showdown_alternate_map.mp4`, restricted to frames with
exactly one detection before and after so the association is unambiguous rather than assumed:
median step **0.55 tiles** at 20 Hz, max **0.92**, and **zero of 101 above 1.0**. Implied speeds
run 10.9–18.4 tiles/s — *faster than anything in the simulator*, whose roster tops out at
`super_proj_speed: 12.0`. So `projectiles.vel` will carry values the policy never saw in training,
which is worth knowing and cannot be fixed from this side.

**Velocity is fitted over the whole span, not the last two samples.** A brawler changes direction,
which is why `EntityTracker` runs an EMA over finite differences. A projectile does not, so every
sample since the first is evidence about one velocity, and `(pos_now − pos_first) / (t_now − t_0)`
divides the anchor noise by a baseline that grows each tick instead of decaying it. No tuning
constant, and strictly better by the third sample.

**Duplicate boxes are real, and distance separates them where IoU does not.** The NMS-free head
is documented as emitting one box per object; across 269 simultaneous pairs from six clips, 17 of
them are the same projectile detected twice. Bucketing those pairs by box IoU and scoring the tile
separation the tracker actually associates in:

| IoU band | n | min | p50 | max (tiles apart) |
|---|---|---|---|---|
| > 0.9 — duplicates | 17 | 0.004 | 0.020 | **0.061** |
| 0.7–0.9 | 5 | 0.057 | 0.077 | 0.238 |
| 0.1–0.7 | 18 | 0.172 | 1.323 | 1.425 |
| ≤ 0.1 — distinct | 229 | **0.910** | 3.894 | 21.179 |

Every duplicate is inside 0.061 tiles and every unmistakably distinct pair is beyond 0.910 — a 15×
gap with nothing in it — while the IoU bands overlap heavily in distance, because these boxes are
large (median 114 px) and perspective turns a few pixels near the top of the frame into more than
a tile. Dedupe is therefore by tile distance at 0.3, and it lives in the tracker rather than in
`_decode_end2end` because that function's refusal to run NMS is *right*: distinct projectiles
genuinely do overlap, and ordinary NMS really would eat a volley.

**One field is still a guess: the anchor.** Projectiles are projected from the box CENTRE, not
`to_tiles`'s 0.30 brawler default, which was fitted to find feet inside a box containing a
nameplate and an aura. But a projectile flies *above* the ground plane the homography maps, so any
choice places it further from the camera than it is. There is a cheap accuracy test available and
it has not been run: a projectile travels in a straight line in world space, so a wrong assumed
height makes the projected track **curve** as it moves toward or away from the camera, and
curvature needs no hand labels. That is the projectile equivalent of `Detection.anchor`'s
"a brawler cannot stand in a wall" test.

**What `MIN_SAMPLES = 2` costs.** A one-sample projectile is dropped, not reported with a zero
velocity — `obs_select` sorts this group by `time_to_closest` *ascending* and `closest_approach`
returns 0 for a stationary point, so an unknown-direction projectile would outrank every real
incoming one and evict them from the twelve slots. The cost is smaller than §9.5's table suggests,
and the difference is worth understanding: that table counts *tracks*, of which only 44% reach two
samples at 20 Hz, but the observation is read at decision instants, which weights by lifetime.
Sampling at the 4 Hz decision ticks instead, **85% of projectiles visible at that moment already
carry a velocity** (28 of 33). The short-lived majority is largely gone before the policy looks.

### 6.2 Grid group — BUILT

`view_channels: [blocks_unit, blocks_projectile, is_bush, is_water, in_zone, enemy_revealed,
hero, ~~box~~, ~~pickup~~, projectile]`. Terrain supplies the static channels from `OccupancyMap`;
detections supply the dynamic ones. Both already exist in tile space, so this group is close to
free once §6.1 lands.

> **BLOCKED, and the sentence above is wrong. Two of these ten channels have no supplier.**
> Audited against what is actually on disk: the entity detector's classes are exactly
> `{0: enemy, 1: teammate, 2: player}` and the terrain classifier's are
> `(FLOOR, WALL, BUSH, WATER, FENCE)`. **Nothing detects a crate or a power-cube pickup**, so
> `box` and `pickup` cannot be filled — and per §9.8's rule a channel nobody will supply leaves
> the spec rather than being fed zeros. Eight channels are covered: the five terrain ones from
> `OccupancyMap`, `enemy_revealed` and `hero` from the tracker, `projectile` from
> `projectile_detection` (weights present, `require()` resolves).
>
> Dropping the two changes the grid's shape from `(10, 13, 21)` to `(8, 13, 21)`, so it must be
> decided BEFORE the retrain, not after. It is not a free drop, because `configs/train.yaml`
> still pays `cube_pickup: 0.5` for a pickup the agent would no longer be able to see.
>
> **RESOLVED — dropped, and `cube_pickup` zeroed with them.** §9.11. The grid group in
> `configs/agent_obs_deploy.yaml` is now the eight covered channels and the shape is
> `(8, 13, 21)`; the operator will add crate and pickup detection in a later pass. So §6.2 is
> **unblocked**: every remaining channel has a supplier on disk.

#### Built — `perception/grid.py`, checked cell-for-cell against `_build_grid`

Nothing in this module is new perception. It crops the accumulated terrain map, scatters the
tracks §6.1 already produces, and its entire contract is **placement** — putting each of those
where `brawl_sim/core/observation.py:_build_grid` puts it. So that function is what checks it:
`tests/test_deployment_grid.py` runs a live `BrawlVecEnv` on `island_invasion`, hands `GridBuilder`
a perfect-perception view of that env's own world, and asserts all eight planes equal the
corresponding channels of the sim's own grid, every cell, every decision, for forty decisions.
34 tests, all passing.

"Perfect perception" is the INPUT, not the thing under test — the occupancy map is seeded from the
sim's padded tile bank and the gas from the sim's zone rectangle, because the question is
placement. Three details the test had to get right, each of which would otherwise have made it
agree for the wrong reason:

- **The sim's tile (0, 0) is not the deployment frame's (0, 0).** `OccupancyMap.origin` centres a
  128×128 grid on wherever tracking started; there is no pre-built map to align to. The test puts
  everything through one offset, which also checks that `GridBuilder` is frame-agnostic.
- **Ten brawlers on a 60×60 map do not meet inside forty decisions**, so the run would have agreed
  on two planes of zeros. Four bots are moved into the crop and the hero's super is charged
  (Mortis's own attack is a dash and spawns no projectile). Measured coverage on this seed:
  terrain and `hero` non-empty on 40/40 decisions, `in_zone` 39, `projectile` 22,
  `enemy_revealed` 18 — asserted, so a regression to one lucky cell fails.
- **The gas seeding is cumulative**, which makes the run a live check of the monotonicity
  assumption below rather than a restatement of it.

Four decisions inside the module are worth stating here.

**The `hero` channel is a constant.** `_view_origin` is `hero_tile - (view_w // 2, view_h // 2)`,
so the hero's own cell is the crop centre by construction — channel 8 is `1` at `(6, 10)` and `0`
everywhere else in *every* training sample where the hero is alive, and all-zero where it is not.
It is written that way rather than projected.

**`enemy_revealed` takes `Track.seen_now`, not "the track exists".** Under `fair: true` an enemy
the hero cannot see appears in NO grid channel — `enemy_any` and `enemy_hidden` are the ones that
would carry it and `obs_select` refuses both. So a coasted track, carried forward on a
constant-velocity prediction because the detector missed it, must not be deposited. That is also
exactly the rule `entities.revealed_to_hero` uses in the `enemies` group, so the two halves of the
observation cannot disagree about who is visible.

**The gas is accumulated, not read per frame.** The poison closes inward and never recedes, so a
cell seen gassed is gassed for the rest of the match; `GasMap` latches it. This is not a
heuristic, it is the game's rule, and without it a cell that went under the joystick after being
gassed would read back as clear. The deposit uses `ZoneMask.cells` rather than `at_least(0.05)`,
because `zone.py` says those answer different questions — `at_least` exists for the occupancy
map's *abstain* path where over-flagging is free, `cells` answers "is this cell in the zone" for
the agent. The cost is that a false positive is permanent.

#### The one real decision: what an unobserved cell reads as, and why it is FLOOR

`occupancy.py` instructs its consumers to treat UNKNOWN as its own state. **This observation has
no state to treat it as.** The policy was trained on eight planes with fixed meanings; there is no
ninth to add without another from-scratch run, and no way to train one either — the sim is fully
observed, so the channel would be constant `1` in every training sample, which is precisely the
lie §9.8's rule exists to prevent. UNKNOWN must therefore map onto a real tile.

Measured at the 4 Hz decision rate over two clips with occupancy accumulating normally:

| clip | median UNKNOWN | worst | after 10 s | final |
|---|---|---|---|---|
| `day12_recording1` (BlueStacks) | **8.8%** | 17.2% | 1.5% | 5.9% |
| `showdown_alternate_map2` | 1.5% | ~6% | 0.0% | 0.0% |

and the per-cell rate, in tenths of the ticks that cell was UNKNOWN, `@` the hero:

```
.....................
.....................
.....................
.....................
.....................
.....................
..........@..........
44442................
66653...........222..
66653...........222..
76664...........222..
77764.........22222..
77764.........33.....
```

**The hole is the HUD**, and it does not sweep out: the joystick is bottom-left, the attack
buttons bottom-right, both are opaque, and the camera holds the hero nearly still on screen, so
the hole is roughly *hero-fixed*. It closes when the hero walks into already-seen ground and
reopens at the exploration frontier — the BlueStacks curve runs 17% → 0% by t=12 s → 11% by
t=20 s. Nothing above the hero row is ever unknown on either clip, because §6's 21×13 view was
sized to fit inside the visible trapezoid; the only hole is UI.

**WALL is the tempting fill and it is the wrong one.** It matches `maps/loader.py`'s out-of-map
padding and is conservative in the abstract, but the table says the error would not be spread
evenly: it would put a phantom wall in the bottom-left of the observation on 40–77% of ticks, and
that phantom would *follow the hero*. A policy whose entire kit is a dash would be told,
permanently, that it cannot retreat down-left. A systematic lie that tracks the agent is worse
than an unbiased one.

**So UNKNOWN is FLOOR** — all four static planes zero. It is also the modal class (61% / 73% of
observed crop cells on the two clips). The residual error splits by channel in opposite
directions, which is the honest way to state the trade: on `blocks_projectile` zero is the
CONSERVATIVE answer (believe you are exposed, and that cover you cannot see is not there); on
`blocks_unit` it is the optimistic one (the hero may dash at a wall it cannot see, which the game
clips at the cost of one dash). `unknown_tile` is a constructor argument so the choice can be
taken back without an edit.

Two caveats on the numbers. They are a mild UNDER-estimate — the hero sits +0.80 tiles below the
viewport centre (`configs/default.yaml`), so the real crop is about one row further into the HUD
than the approximation used. And out-of-map ground mostly does not arise: when the hero stands at
a map edge the ground beyond it is on screen and gets classified as whatever the game draws there,
because the hole is only ever BELOW.

### 6.3 The `self` group — proprioception, not perception — BUILT

`agent_obs_lowinfo.yaml`'s stated rule is sim-to-real parity: *"the agent may see only what
brawl_vision can actually recover from a screen capture."* That rule was applied to `enemies` and
`projectiles`; the `self` group is still "identical to configs/agent_obs.yaml."

**The rule as written is too strict for this group, and the reason is structural: the deployed
agent is not a passive observer. It is the thing issuing the actions.** Anything that is a
deterministic function of *(our own action history + the kit constants in `configs/brawlers.yaml`
+ elapsed time)* is available exactly, with no CV at all. Verified against the sim rather than
assumed:

- `core/movement.py:54` — `ent_facing = angle_of(move_dir)` whenever the hero moves.
- `core/hero.py:276` — `ent_facing = angle_of(dash_dir)` on a dash.
- `core/hero.py:124` — `long_dash_ready` is "has this entity gone `long_dash_seconds` without
  attacking", i.e. a stopwatch on our own fire events.
- `core/hero.py:114-115` — `ent_attack_cd` and `ent_invuln_t` are plain `-= dt` countdowns seeded
  from per-kind constants.

Every one of those is driven by the action stream we generate. So the deployment loop keeps a
**shadow hero state**: the hero's own timer fields, ticked at `cfg.dt`, advanced by the actions we
issue, using the same `configs/brawlers.yaml` params the sim loads. Re-auditing all 21 fields:

| Field | Source | How |
|---|---|---|
| `hero.pos_norm` | CV | Camera pose + detection. |
| `hero.vel` | CV | Frame-to-frame position. |
| `hero.hp` | CV | `hp_detection` — the **numeral**, which is what `read.py` produces. Replaced `hero.hp_frac` in §9.10: its denominator is max_hp, and a power cube adds a flat +400 to that with nothing on screen saying who holds how many. |
| `hero.ammo_frac`, `ammo_whole` | CV + shadow | The ammo bar in the hero's own box stack, below the HP bar — `hp_detection/locate.py` documents the layout and `find_bar` already reads bars there. Also dead-reckonable, which makes it the **desync canary** (below). |
| `hero.super_ready`, `super_charge_frac` | CV | Two sources: the super-charge bar in the box stack (same `find_bar`) or the button's fill. Prefer the bar — it is the same reader as ammo. |
| `hero.in_bush` | CV | Terrain lookup at hero tile. |
| `hero.in_zone` | CV | `terrain/zone.py` at hero tile. |
| `hero.cubes` | ~~CV~~ | **Dropped** — §9.8. Recoverable (the pip row is in the same box stack as the HP bar), but no reader will be built, so it leaves the spec rather than being fed a constant. |
| `meta.n_enemies_alive` | CV | "Brawlers left: N" — `brawl_vision/hud.py`, **built** (§6.4). Note it is the reading **minus one** while the hero is alive: the game counts you. |
| `hero.can_attack`, `attack_cd` | **shadow** | Countdown from `attack_cooldown`, seeded on our own fire. |
| `hero.facing_vec` | **shadow** | Set by our own `move_dir` / `dash_dir`. Never needed sprite orientation. |
| `hero.dashing`, `dash_t`, `dash_dir` | **shadow** | Dash is triggered by our own attack; `dash_duration` is a kit constant. |
| `hero.long_dash_ready`, `long_dash_frac` | **shadow** | Stopwatch since our last attack vs `long_dash_seconds`. |
| `hero.invuln` | **shadow** | `-= dt` countdown, seeded by a DASH — not, as this row said before it was built, at spawn. `core/spawn.py` writes pos, hp, ammo, facing and alive and nothing else; `ent_invuln_t` has exactly one writer, `start_dash`, and `obs_schema` says so itself: "invuln_t > 0 (dash i-frames)". |
| `meta.time_frac` | wall clock | Seconds since the gate went true ÷ 150, **clamped to [0, 1]**. §9.6. |

**Consequence: proprioception costs the `self` group nothing.** That removes what looked like the
largest blocker in this design. The earlier reading of it — seven unrecoverable fields forcing a
narrowed spec and a from-scratch run — was wrong; it treated the loop as a spectator.

One field does leave the group, and for an unrelated reason: `hero.cubes` is dropped in
`configs/agent_obs_deploy.yaml` by operator decision, not because proprioception failed to reach
it (§9.8). And a retrain *is* now happening — driven by the three `entities.*` fields, none of
which this section's argument covers. What stands is that no `self` field forced it.

**This matters most for Mortis specifically**, which is the MVP brawler. The dash and long-dash
fields are his entire kit, so a spectator-only reading would have gutted exactly the columns his
policy leans on hardest.

**The real risk moves from "missing" to "drift."** Dead reckoning is exact only while the shadow
state and the game agree. It desyncs when an action does not land as modelled — an input dropped
by the backend, or a dash cut short by a wall the sim's geometry and the game's disagree about.

The mitigation is that several fields are observable *and* dead-reckonable, which makes them free
consistency checks:

- **`ammo` is the canary.** The pip bar is directly readable and the shadow predicts it. A
  mismatch means an attack did not land as modelled — the earliest and cheapest desync signal
  available, and it costs one comparison per decision.
- **Position** is CV-tracked and dead-reckonable; divergence beyond a threshold means movement is
  not going in as issued.
- On desync: resync every observable field from CV, reseed the unobservable timers conservatively
  (assume *not* ready rather than ready), and log it. Sustained desync is a fail-closed condition
  per §8.

#### Built — `perception/shadow.py`, checked against the sim rather than against itself

`ShadowHero` runs `env._run_tick`'s phases in `env._run_tick`'s order for the fields it owns.
Every claim that makes is a claim about matching `brawl_sim`, so the sim is what checks it:
`tests/test_deployment_shadow.py` drives a live `BrawlVecEnv` and a `ShadowHero` off **one**
scripted action stream and asserts they agree field for field, decision for decision, at
`action_repeat` 1 and 5. The bots are pinned idle through `env.step`'s own `override` argument and
given 1e7 HP, so nothing but the hero's own actions can move a compared field. Sixty decisions,
covering an empty clip, six dashes, and one deliberately charged long dash — with a second test
asserting the script really does reach all three, so a stream that never fires cannot pass
vacuously. `ammo`, `attack_cd`, `dash_t`, `attack_idle_t` and `ammo_whole` are asserted **exactly**
equal, not approximately: they are the same float32 operations in the same order, and a tolerance
there would hide the one thing that genuinely needed measuring.

**Timers must be float32, and that is not cosmetic.** The sim's are float32 tensors with `cfg.dt`
promoted into them. Counting `x = max(0, x - 0.05)` down from Mortis's `attack_cooldown: 0.35`
takes **seven** ticks in float32 and **eight** in float64, where a residue of 4.2e-17 keeps it
above zero for one more tick. A float64 shadow would hold `can_attack` false and block ammo regen
for one sub-tick longer than the sim on every single attack — a 50 ms error, in the same direction,
every time. What one dash actually looks like, all three confirmed equal to the sim's:

    sub-tick     1  2  3  4  5  6  7  8
    dashing      *  *  *  *  *  *  .  .     6 ticks -- exactly dash_duration
    invuln       *  *  *  *  *  *  *  .     7 ticks -- one longer
    reload held  *  *  *  *  *  *  *  .     7 ticks -- exactly attack_cooldown

The i-frames outlast the dash by one sub-tick because `dash_t` is decremented in phase 8 of the
tick that set it while `invuln_t` waits for phase 2 of the next. Both are seeded from the same
0.30. None of this is an artifact to round away — it is what the policy trained against.

**A nominal decision is not five sub-ticks unless you are careful.** `0.25 // 0.05` is **4**, not
5: binary 0.05 sits a hair above a twentieth, so flooring a division silently drops one sub-tick
out of every nominal decision. The parity test caught it on decision zero. `advance` subtracts with
a nanosecond of slack instead of dividing.

**The wall clock decides how many sub-ticks, the sim's arithmetic decides what each one does.**
`advance(seconds)` banks real elapsed time and runs whole `dt` sub-ticks out of it. The
alternative — a fixed five per decision — keeps sim parity by construction and loses the game: a
loop running at 15 Hz would leave the shadow's cooldown running after the real one expired, which
presents as an agent that quietly under-fires and **never trips the canary**, because the ammo it
predicts is the ammo it would have had.

**The action mask is one sub-tick stale, and provably in the safe direction.** `hero.action_mask`
is evaluated after phase 2's timers; the deployed one is evaluated at decision time, before them.
Every term moves one way only — `attack_cd` and `dash_t` only fall, `ammo` only rises — so the
deployed mask is a strict subset of the sim's. It costs up to 50 ms on a shot issued at the exact
tick a cooldown expires, and it can never offer an action the game will refuse.

**One field this section promised is NOT dead-reckonable, and the correction matters.**
`core/hero.add_super_charge` counts hits landed on living players — boxes are excluded by
`env._bookkeeping` precisely so a hero cannot farm his super off crates. Whether our dash connected
is not in our action stream, so `super_charge` cannot be reckoned **upward** at all. It is CV,
from `read_super`, and the shadow does three things with it that CV alone cannot: gate it on the
cooldown and the dash (which only the shadow knows), spend it the instant we tap, and **latch that
spend** so a stale frame still showing magenta cannot re-arm the mask. That last one is not
hypothetical housekeeping: a super costs charge rather than ammo, so a double-tapped super is
exactly the desync the ammo canary cannot see.

**The canary's thresholds, from §6.4's measurements at both ends.** Reader noise is ±0.06 on one
pip with 98.3% of reads within 0.15 of an integer; the smallest real desync is 1.0, a dash that
did or did not happen. `AMMO_TOLERANCE = 0.5` is the midpoint on a linear or a log scale alike.
Two guards sit on top: a **grace window** after each attack, because the frame being compared was
captured before the tap went in, and a **debounce** of three consecutive strikes (0.75 s at 4 Hz),
because `read_ammo` misses 12% of frames and the resync it would trigger is destructive. A miss
neither strikes nor clears — it is not evidence either way. Where the canary is blind is worth
naming: the grace window mutes it during a burst, and Mortis can fire every 0.35 s. It is fully
live through the 2.25 s reload that follows, which is exactly when "did we spend two or three?"
is answerable.

**Detection and response stay separate.** `check_ammo` returns a verdict and never acts on it;
`loop.py` decides, because `loop.py` is also what has to count sustained desyncs into §8. A resync
firing silently inside a check would hide the very failure the check exists to report. When it does
fire, every unobservable timer is reseeded to the state that offers the policy *less*: a full
cooldown, no dash, no i-frames, an uncharged long dash, no super.

**Position is a ceiling, not a threshold.** This section named position as the second desync
signal. A symmetric "CV position diverged from the prediction" check is not implementable and would
be useless if it were — the shadow has no terrain, so every wall the hero walks into reads as
divergence. What is sound is one-sided: `travel_budget` accumulates the *furthest* the hero could
have gone (`move_speed × dt` walking, the **unclipped** `dash_speed × step` dashing), and terrain
only ever shortens the real distance. A CV displacement that *exceeds* the budget is the tracker or
the odometry being wrong, never the hero being fast.

**And a dash into a wall costs the shadow nothing.** §10 item 7 asks for footage of one on the
assumption that a blocked dash is the clearest desync case. Working through what `start_dash`
actually clips: the terrain march shortens `clipped_distance`, hence `dash_speed`, hence the
landing point. It does not touch `dash_duration` — and every field the shadow owns is seeded from
the duration or from the kit constants. `dash_t`, `invuln_t`, `attack_cd`, `ammo` and
`attack_idle_t` are all clip-independent. The footage is still worth having, to tighten the travel
budget; it is not the emergency it looked like.

#### The three readers this needs, and where they live — ALL THREE BUILT

The shadow state is bookkeeping and belongs here. The CV rows of that table are not: `hero.ammo_
frac`/`ammo_whole`, `hero.super_charge_frac` and `meta.n_enemies_alive` had no reader in
`brawl_vision`, and **they are `brawl_vision`'s job, not this package's.** Same rule the rest
of the split follows: `brawl_deployment` wires perception to control and owns nothing that reads a
pixel. (`hero.cubes` was a fourth; §9.8 drops it instead.)

**Two of the three go in `hp_detection`, not in a new module, and `locate.py` already says why.**
Its docstring records the measured layout of a brawler's readout: *"a power-level pip, the
player's NAME, the HP number, the HP bar, and — for your own brawler only — an ammo bar and a
super-charge bar."* So ammo and super charge are box-relative, in the same stack the HP bar sits
in, and `locate.find_bar(crop_hsv, hue_range, below_y)` is already the generic "find a coloured
bar at this offset" primitive. These are new hue ranges and offsets against an existing locator,
not a new perception stage. `ammo_whole` is then `round(frac × max_ammo)` from
`configs/brawlers.yaml`.

That leaves **`meta.n_enemies_alive`** as the only genuinely screen-space readout — "Brawlers
left: N" is drawn in the HUD, not attached to any brawler, so it is the one that wants a `hud`
module. It is also the cheapest: one digit in the same fixed font `hp_detection/glyphs.py`
already classifies by NCC against ten mean templates.

**`ammo` was the one to build first**, because this section's whole drift guard depends on it —
and it is the only one of the three where being wrong is silent rather than obvious. All three
now exist and are measured: `hero_bars.read_ammo`, `hero_bars.read_super`, and
`brawl_vision/hud.py:HudReader.brawlers_left`. §6.4 carries what each of them measured.

**One correction to the row above, from building it.** `meta.n_enemies_alive` is *not* what the
HUD says. The game counts brawlers **including you**, so the observation is the reading minus
one — and only while the hero is alive, because the game stops counting you the moment you die.
`BrawlersLeft.enemies_alive(hero_alive=...)` takes that operand with no default rather than
letting the call site guess, since the error it prevents is a plausible integer either way.

`meta.time_frac` is the one field in this group with neither a reader nor a shadow, and it needs neither: §9 item 6 resolved it to **wall-clock seconds since the gate went true ÷ 150, clamped to [0, 1]**. `loop.py` owns it, because `loop.py` owns the gate.

---

### 6.4 The three readers — all BUILT

§6.3 leaves three fields to CV: `hero.ammo_frac`/`ammo_whole`, `hero.super_charge_frac`/
`super_ready`, and `meta.n_enemies_alive`. All three are built. Measured against
`tests/fixtures/vision/bluestacks-example-new.mp4` — **at that clip's own 1920×1080, not at the
2002×1126 stated here previously.** `ClipReader` trims to 16:9 and never resizes, and the clip is
natively 16:9, so nothing brings it to 1126.

That correction matters because **deployment resizes to 2002×1126** (§9.4 item 4), so these are
1080 numbers being applied at 1126. The readout stack scales with viewport height — measured, the
super track runs 116 px at 1080 and 120 at 1126, and the ammo pip a median 33 and 34 — which
leaves the two bar constants right at deployment by luck rather than by design: `hp.ammo_pip_px`
34 sits inside the 33–35 spread at both, and `hp.super_track_px` is a ±22 gate that is never
divided by, so a 4 px offset costs nothing. `hud.py` was written after this was found and scales
its own constants by frame height instead. ~~**Open, small: whether to recentre
`hp.super_track_px` on the 120 deployment actually sees.**~~ **Closed — no.** §6.5 measures the
whole readout stack at both resolutions on 713 enemy boxes and the viewport costs 1–2 pp; a ±22
gate that is never divided by is not where that goes.

#### The finding that shapes all three: these denominators are recoverable, and HP's was not

`hp_detection/read.py` explains at length why it ships no fill ratio: *"the depleted remainder is
a pale lavender that does not separate cleanly from white text or from a pale sprite… a ratio
computed against the filled run alone would be the constant 1.0 dressed up as a measurement."*
That is the right call there and it does **not** transfer to these three, which is the reason this
work is worth doing at all rather than being blocked by the same wall:

| track | filled | remainder | denominator |
|---|---|---|---|
| HP bar | green, hue 56 | pale, washes out on the damage flash | **not** separable |
| ammo | orange, hue 10–11, sat 155–223 | dark slate, val 60–80 | separable, **and** the track is exactly 3 equal segments |
| super | yellow, hue 27–29, val ~250 | dark purple, hue 126, sat ~130, val ~97 | separable — strongly |
| brawlers left | white glyphs on a dark plate | n/a — it is a NUMBER, not a bar | n/a |

Two of these are bars whose empty portion is painted in a saturated colour rather than a wash, and
the third is text. None of them has HP's problem.

#### Where they live

`brawl_vision`, not `brawl_deployment` — this package wires perception to control and owns nothing
that reads a pixel. Within `brawl_vision` the split is by **what the pixels are attached to**,
which is the same rule `hp_detection`'s own docstring uses to justify sitting inside
`object_detection`:

- **Ammo and super are box-relative.** They are drawn in the hero's own readout stack and move
  with the hero, so they cannot run without a detection, exactly like `hp_detection`. They go in
  `object_detection/hp_detection/hero_bars.py`, next to the `locate.py` they anchor against.
  (The package name becomes slightly narrow. Renaming it would touch every importer for no
  behavioural gain; the docstring says so instead.)
- **`meta.n_enemies_alive` is screen-space.** "Brawlers left: N" is HUD text at a fixed position
  and has nothing to do with any detection. It goes in `brawl_vision/hud.py`, a new sibling.

#### The readout stack, measured

`locate.py` documents the order — power pip, name, HP number, HP bar, *"and — for your own
brawler only — an ammo bar and a super-charge bar."* Those last two are what this section reads.
Row offsets below the HP bar's own row, over 40 frames:

    dy  0–9    HP bar          hue 56, sat ~195, val ~250
    dy 10–14   gap             dark
    dy 15–25   AMMO track      hue 10–11
    dy 26–28   gap
    dy 29–43   SUPER track     hue 126 (empty) / 27–29 (filled) / 162 (ready)

**Anchor on the digit row, not on the HP bar's extent.** `find_bar` returns the *filled* run, so
its x-range shrinks with damage — measured 112 px at 8000 HP and 75 px at 5560. An early
measurement pass keyed its window to that length and clipped the third ammo pip, reporting a full
pip as 29% full. The bar's **row** is stable and fine to use; its **width** is not, and the tracks
below it stay full width regardless of health.

#### Ammo — three equal segments

Measured pip layout, at a track ~115 px wide: segments start at ~0, ~38, ~77 px, each ~35 px of
paint with a ~3 px gap. Four states appeared across five sampled frames — 3 full, 2 full + 1
empty, 1 full + 2 empty, and a partial — which is the whole range the field can take.

    ammo       = Σ over the 3 slots of (orange pixels in slot / slot width)
    ammo_frac  = ammo / 3          obs_schema: "ammo / max_ammo", and max_ammo: 3 for Mortis
    ammo_whole = floor(ammo)       obs_schema: "floor(ammo) -- whole shots available"

The partial slot is not an artifact to round away: `hero.ammo` is documented as *"fractional,
reloads continuously"*, and the game draws the reloading pip partially filled. Reading it is what
makes the shadow state's canary sharp instead of quantised to thirds.

**Built, and the design above is what changed under measurement.** `hero_bars.py` implements this
and `tests/test_vision_hero_bars.py` pins the failure modes. Three things in the plan were wrong:

*The slot width does not come from the detected track.* The plan said to divide the track's own
extent by three and never to hardcode a width. The extent is **not recoverable**: over 60 frames a
mask of "orange ∪ dark slate" returned 35–236 px for a ~115 px track, because the map behind the
widget is itself dark blue-purple, so the empty part of the track does not separate from the
background even though it separates cleanly from the filled part. What replaced it needs no extent
— **the three pips are identical, so painted pixels ÷ one pip's width is the ammo count directly.**
That width is a constant (`hp.ammo_pip_px: 34`, the mode of 353 runs), which is what the plan
warned against; the guard the plan wanted is kept instead as a *reported* cross-check —
`AmmoReading.pip_px` carries the frame's own widest run, and the footage test fails if its median
walks more than 2 px from the calibration.

*Colour alone was not enough, and the fix was structure.* Counting every orange run put **16% of
readings above the physical maximum of 3, peaking at 3.71** — the map's orange crate borders pass
the same colour test. Pips sit at a fixed pitch, so a run whose offset is not near a multiple of it
is not a pip. That dropped the maximum to 3.09, and capping each slot at 1.0 — *a pip cannot be
more than full* — removed the rest. This is `locate.py`'s own lesson, *"colour alone does not
distinguish a bar from a background"*, reached again for a different bar.

*A full pip reads 0.94–1.03, not 1.00, and that is the floor.* Pips render 33–35 px against the
calibrated 34. Two escapes were checked and neither works: 34 is not simply the wrong constant
(mean 34.33, sd 0.66 over 144 frames; an earlier pass suggesting 35 was reporting each frame's
*longest* run, a maximum compared against a mode), and the widget does not meaningfully scale
(pitch and width correlate at +0.81, but pitch sd is 0.88 px on 38 and pitch barely tracks digit
height at +0.33 — shared anti-aliasing, not camera zoom; a per-frame denominator moves the spread
from 0.88–1.03 to 0.96–1.07, no better, and it needs two visible pips to work at all). So the
residual is reported rather than snapped away, because a snap threshold would quantise out the
partial reload this field exists to see. On `ammo_frac` the error is ±0.02.

Measured over 175 reads of `bluestacks-example-new.mp4`: 98.3% within 0.15 of an integer, nothing
above 3, 98.3% of slot shapes physically possible, and 12% returning "could not read" — that last
is the pre-existing `find_digit_row` / `find_bar` miss rate, not new, and §6.3's shadow state
already costs one tick of confirmation for it.

**One methodological note, because it cost a debugging pass.** The first validation run reported
only 82.3% plausible slot shapes and I went looking for a detection bug — the anchor, the lattice.
There was none. The validator asked whether each slot was within **0.02** of full, which is tighter
than the reader's own ±0.06 precision, so honest full pips reading 0.97 were being counted as
malformed. *A tolerance tighter than the measurement it judges measures the tolerance.* The check
now runs at 0.1 and the reader is unchanged.

#### Super — and the third colour state that would otherwise be a disaster

    super_charge_frac = yellow run length / track length
    super_ready       = the track is MAGENTA

At full charge the bar does not read as "all yellow" — **the whole track turns magenta**, hue 162
at saturation 255, measured on a frame where Mortis's super was up. A reader that counts yellow
and divides would report **0.0 charge at exactly the moment the super is ready**, which is the
worst single error available in this field and is silent. So magenta is a state, not noise, and
the test for it comes first.

`super_ready` is then the magenta test rather than `frac >= 1.0` — more direct, and it is what
gates action masking at inference: an uncharged super means bin 2 is masked the way `MaskablePPO`
saw it in training.

The alternative source — the super BUTTON in the bottom-right HUD, at a fixed position — was
considered and is worse for this purpose. It is essentially binary (lit / not lit), so it can
support `super_ready` but not `super_charge_frac`, and the continuous field is the one the policy
was trained on. It stays available as a free cross-check.

**Built, and the design above survived measurement — including the magenta warning, which was
right.** `hero_bars.read_super`, `tests/test_vision_hero_bars.py` (18 more tests, synthetic;
one footage test carries `@pytest.mark.vision`). Three colour states, each measured from runs
identified by *structure* rather than by the thresholds being calibrated — anchor on the
unambiguous yellow, then read the 20 px immediately to its right, which is the empty remainder by
construction:

| state | hue | sat | val | n px |
|---|---|---|---|---|
| filled | 26–29 | 114–197 | 219–255 | 3890 |
| **ready (magenta)** | **162–166** | **255 at every percentile p5–p95** | 189–234 | 6494 |
| empty | 124–133 | 102–143 | 74–99 | 1720 |

**One thing changed, and it is the interesting one: the denominator comes from the frame, not from
a constant.** The plan said `yellow run / track length` without saying where the track length
comes from, and after the ammo work the natural assumption was that it would have to be another
calibrated constant — the ammo track's extent is *not* recoverable, because its empty remainder is
dark slate against a dark blue-purple map. The super track's empty remainder is a distinct colour,
so filled + empty recovers the extent and the reading normalizes itself. `hp.super_track_px` still
exists, but only as a credibility gate on the extent found; it is never divided by.

**That separation is narrow, and a first pass proved how narrow.** Using `sat >= 80, val 60–140`
for the empty colour — which reads as a reasonable band — the merged "track" came out **110–225 px
for a 116 px widget**, because that band is wide enough to include the map itself. Every charge
reading was deflated in proportion, and nothing about the output looked wrong. The fix was to stop
guessing the mask and measure the remainder directly; the shipped band is `val 62–118`, and the
tightness of the value bound is the only thing separating the widget from its background.

Measured over 204 frames of `bluestacks-example-new.mp4` carrying a player box: **83.3% read**,
16.7% "could not read" (the pre-existing `find_digit_row`/`find_bar` miss rate again, not new).
Track extent p10 114 / p50 116 / p90 118, sd 2.6. Charge covered the full range — 32 reads under
0.1, 61 at ready, the rest spread evenly between.

**The check that would catch a wrong denominator, and did not fire.** A super fills gradually and
empties instantly, so consecutive readings should climb in small steps and fall in rare large
ones; a drifting extent shows up as *upward* jumps. Over 161 consecutive pairs: **zero jumps above
0.20**, 22 between 0.05 and 0.20, 55 small positives, 60 flat, and exactly 2 large drops — the two
times the super was used. Median |Δ| 0.012. That is what a correct denominator looks like, and it
is asserted in the footage test rather than left as a number in prose.

#### Brawlers left — built, and the templates transferred

`brawl_vision/hud.py`, 30 tests. The one open question was the glyph scale, and it resolved in the
cheap direction.

**The templates transfer, so there is no second bank.** `hp_detection/glyphs.py` classifies by NCC
against ten mean templates harvested at **15–17 px**; these digits are the same typeface at
**40–42 px**, 2.6× larger. Measured after the structural gates, across ten clips: score p05
**0.893–0.930** per clip against the HP population's own p05 of 0.698. `glyphs.normalize` is
genuinely scale-free — downsampling a big clean glyph to the 18×14 canvas loses nothing that
distinguishes a digit — and `scripts/vision_hp_calibrate.py` needs no HUD mode.

**The HUD scales with viewport height, exactly, which the control buttons do not.** Every constant
is a reference pixel at a 1126 px viewport. The 1080 measurements scaled by 1126/1080 land inside
the 1126 measurements on every row — colon x 350→364.9 against 365–367, digit height 39–40→40.7–41.7
against 40–42. That matters because `gameplay.py` found **two** button layouts in this same
footage and had to calibrate the attack ring per recording: (1490, 816) r 65–78 in
`zone_grows_from_east` and `day10_gameplay`, (1752, 782) r 46 in `day12_recording1`. All three put
the colon at x = 366. **The game's UI-scale setting moves the controls and leaves the top-left
match readout alone.**

**The colon does three jobs and the third was free.** It separates the count from the label —
whose cap-height letters are 39–41 px, exactly digit-sized, so height cannot. It says whether the
widget is on screen at all, which nothing else does, since the band still holds white map pixels
during the fly-in. And **its x IS the label's rendered width**, so gating its position gates the
string in front of it: a longer label lands its colon outside a 26 px window. Checking the label
glyph by glyph would buy nothing over that.

**Read rate and the free ground truth.** 4684 sampled frames over ten clips, **4538 reads**. The
three clips under 99% start before the match does — every non-read precedes the frame the counter
first appears on, none follows it — and exactly one mid-match frame is refused as `too-long`,
which is the reader failing closed. Brawlers left is monotone non-increasing within a match, which
is ground truth for free: over 4528 consecutive pairs there are **zero increases**. Asserted in the
footage test, not left in prose.

**The one failure it cannot see, recorded rather than fixed.** If the trailing digit of a `10`
fails the white threshold, the row simply ends after the `1` — there is no gap to notice, because
the count is left-aligned to the colon and one- and two-digit counts start at the same x. It reads
1, which is legal. Unreachable in 4538 reads, and structurally it needs the `0`'s white to fail
while the `1`'s survives two pixels away. The error is always *low*, never high, which is what
makes it catchable by §6.3.

**Off-mode it stays quiet, but that is weaker evidence than it looks.** Three training-cave clips,
284 frames, zero reads — and their band is bare map, so this shows the reader not inventing a
counter rather than telling one labelled readout from another. `hud_mask.json` says the cave shows
"Damage per second" in this corner; these clips do not. A test covers it instead.

#### Failure policy, shared by all three

Each returns an explicit "could not read" rather than a plausible number, for the reason
`agent_obs_lowinfo.yaml` states and §6.3 repeats: a fabricated value in a column the policy was
trained to trust fails silently. The consumer is §6.3's shadow state, and what it does with a
missing read differs per reader, which the phrasing here previously flattened. **Ammo** is the one
it genuinely dead-reckons, so CV is a pure *check* and a missing read costs one tick of
confirmation. **Super charge** it can only reckon downward — charge accrues from hits landing,
which is not in our action stream — so a missing read is a real gap, and the shadow fails closed by
reporting not-ready rather than holding a stale one. **Brawlers left** it cannot reckon at all.
Sustained disagreement on ammo is the desync signal, and §8's fail-closed path is where that ends
up.

#### What I would revisit

- ~~**`hero.hp_frac` has the same denominator problem.**~~ **Resolved by §9.10, and not the way
  this section expected.** The note here proposed recovering the denominator from the BlueStacks
  skin, where the depleted HP remainder is a clean dark purple rather than the pale lavender
  `read.py` measured on iOS. That would have worked for the bar and still been wrong, because the
  denominator is not a rendering problem: **max_hp itself is unobservable**, since a cube adds a
  flat +400 to it for the hero and every enemy alike. The fix was to stop needing a denominator —
  the spec now takes absolute `hero.hp` / `entities.hp`, which is what the reader always produced.
  Worth keeping as a worked example of a measurement question that was really a modelling one.
- **Pip pitch as a scale check.** Three equal segments at a known pitch is a free ruler. If the
  measured pitch ever drifts from a third of the track, something upstream rescaled and the
  calibrated-viewport assumption (§9.4) has broken.

### 6.5 `entities.hp` — the enemy numeral, measured 2026-09-08

The last unmeasured column in the deployed spec. §6.4's readers are all hero-side; `entities.hp`
points the same `hp_detection` stack at boxes labelled `enemy`, and its accuracy had only ever
been hand-scored on iOS footage for the HERO. Measured now on both BlueStacks clips, through
`smooth.HealthTracker` (the layer `assemble.py` will actually call, not the memoryless reader),
at the deployment viewport. Driver: `scratchpad/enemy_hp2.py`, `scratchpad/enemy_hp3.py`.

| clip | enemy boxes | has hp | trusted (≥0.50) | seen this frame | staleness p90 |
|---|---|---|---|---|---|
| `bluestacks-example-new` | 325 | **84.0%** | 56.9% | 68.9% | 1 frame |
| `bluestacks-example-zone` | 388 | **84.5%** | 69.6% | 75.5% | 1 frame |
| hero, same runs | 1279 | 98.0–99.1% | 91.3–96.6% | 94.8–97.6% | 0 frames |

**The viewport resize costs nothing, which retires §6.4's open worry from the other side.** The
same measurement at the clips' native 1920×1080 gives 85.2% / 86.9% against 84.0% / 84.5% at
2002×1126 — 1–2 pp, in native's favour, across 713 enemy boxes. The readout stack's constants were
fit at 1080 and are being applied at 1126; that is now a measured 1–2 pp rather than an assumption.
Not worth acting on, and specifically not worth recentring `hp.super_track_px` for.

**The 15% gap does not concentrate where it would hurt.** Per-track (`enemy_hp3.py`): of 66 enemy
tracks across both clips, **every single track that never produced an HP lived 4 frames or fewer**
— they are glancing detections at the tracker's own `max_misses` floor, not enemies the policy
would ever have been told about. Tracks that persist read HP within **1 frame at p50, 3 at p90**,
i.e. inside one 250 ms decision window, and then hold coverage of 0.96 / 0.67 over their lives
with the tracker carrying the value through the gaps (staleness p90 = 1 frame).

**So the rule for `assemble.py`, and it needs no fabricated value:** a track is promoted to an
enemy slot only once it has a committed HP. Until then the slot is `entities.alive = 0` with the
rest of its columns zeroed — which is exactly the state the sim produces for an empty slot, so it
is in-distribution rather than invented, and it costs at most one decision of latency. Once
promoted, the slot holds the tracker's last committed value for the life of the track. At no point
does a number get made up, which is the §9.8 rule this column was the last candidate to break.

One residual, reported rather than fixed: **1 absurd value in 713 enemy boxes** (an `hp` of 20,
at the viewport, on the new clip; zero on the zone clip and zero at native). That is the
truncation failure `smooth.py`'s `_is_truncation` exists to catch, leaking at roughly 0.1%. It is
below the rate at which it would be worth adding a range gate that could reject a genuine low-HP
enemy — the case where the number matters most.

### 6.6 `assemble.py` — the last stage, and it delegates — BUILT 2026-09-08

Everything §6.1–6.5 built produces one of the observation's ingredients. This turns them into the
tensor the checkpoint reads. **It does not lay out the columns itself.**

The tempting shape is obvious and wrong: read the spec, concatenate the fields in order, divide by
the normalization scales. That is a second implementation of `obs_select.build_agent_obs` — code
that already ran 300M times during training — and **every one of its failure modes is silent.** A
column in the wrong order, a divisor off by the map size, the projectile group sorted differently:
the policy does not crash, it just gets quietly worse, in a way no test that owns only one of the
two implementations can see.

So `assemble` builds a **minimal `full_obs`** — the same nested dict of leading-`(N,)` tensors
`core/observation.py` produces, at `N = 1`, holding only the fields the loaded spec names — and
hands it to `build_agent_obs`. Column order, normalization, the projectile group's
top-K-by-`time_to_closest` selection and the fairness masking are then not merely *equivalent* to
training's; they **are** training's, by construction. The module is ~300 lines and none of them is
arithmetic on a column.

Three consequences that all look like bugs and are not:

* **The `view` is built with all 12 sim channels, 4 of them never filled.** `GridBuilder` produces
  the 8 the spec asks for; `assemble` scatters them to their canonical indices
  (`obs_select._CHANNEL_INDEX`) and leaves `enemy_any`, `enemy_hidden`, `box`, `pickup` zero.
  `_build_grid_group` does an `index_select` over exactly the 8 configured channels, so those
  zeros are **provably unread** — which is the distinction that makes this legal under §9.8.
  Splicing in a finished 8-channel array would also work, and would quietly stop working the day a
  spec reorders its channels.
* **The entities axis carries a hero slot.** `obs_select` slices index 0 off
  (`_HERO_AXIS_PREFIX`), so tracker slot *k* lands at `entities[:, k + 1]`. Writing enemies at
  index 0 shifts every enemy one slot and silently deletes the last one.
* **The zone group is driven by the loaded spec, not a fixed list.** Two spec versions are live at
  once (§9.16) and they want different zone fields. A hardcoded list would silently supply the
  wrong one after a retrain; a missing field raises instead.

**The test is parity, not shape.** `tests/test_deployment_assemble.py` runs a real `BrawlVecEnv`,
pulls out of its observation exactly the quantities the deployed suppliers produce — the shadow's
timers, the tracker's world positions and velocities, `HealthTracker`'s two HP columns, the
brawler count, the terrain lookups, the 8 spec channels of the view — feeds them through
`assemble`, and demands the result equal `build_agent_obs`'s on the same frame, column for column,
over six stepped frames and on **both** deploy specs. A separate test walks 40 steps firing every
tick so the projectile group is exercised with live occupants, since that group is the only one
that *selects* rather than copies and deployment's slot indices differ from the sim's.

That test found one real defect on its first run: `assemble` returned `t[0].numpy()` on the reused
output buffers, which **shares memory**, so every tick silently rewrote the previous tick's
observation. It now copies — one 3 KB copy per decision, four times a second, which is not the
kind of cost §7 is watching for.

Remaining gap: the **zone estimator** (a `GasMap` scan along ±x/±y clamped at
`cfg.zone_margin_horizon_tiles`) is still unwritten. `assemble` takes the zone fields as an
argument, so it is the caller's job and it does not block anything here.

### 6.7 `policy.py` — the checkpoint and its guards — BUILT 2026-09-08

A thin wrapper over `MaskablePPO.predict`, and thin is the point. What the file actually
contributes is **load-time refusal**, because every way of deploying a checkpoint against the
wrong observation is silent at inference.

**The run is the only argument.** `DeployedPolicy.from_run(run_dir)` reads that run's own
`train.yaml` and takes the spec, the env config and the env overrides *from it*. Nothing accepts a
spec path. `agent_obs_deploy.yaml` and `agent_obs_deploy2.yaml` differ by two columns in one
group; a checkpoint handed the wrong one gets a shape error if it is lucky and a silently shifted
`zone` group if it is not. `make_assembler()` then hands `ObservationAssembler` the *same loaded
spec object*, so the two cannot disagree — which is why §6.6's constructor now takes objects
rather than paths.

Four guards, in order:

1. **`algo != maskable_ppo` → refuse.** Not because `predict` would raise on `action_masks`, but
   because an unmasked policy can request a super it has not charged.
2. **`normalize.obs: true` → refuse.** *The silent one.* Such a run keeps its observation
   statistics in `best_vecnormalize.pkl` and is meaningless without them; the deployed path has no
   `VecNormalize`, so loading one feeds raw values to a network trained on whitened ones — no
   error, no crash, every column quietly wrong at once. **`mortis_deploy-20260907-041522` has
   `normalize: {obs: false, reward: true}`, so the deployed path needs no statistics at all.**
   Reward normalization is training-only. That is a fact about *this* run, which is why it is
   checked rather than assumed.
3. **Observation space ≠ `agent_space(spec, cfg)` → refuse.** The only cause worth naming is a
   spec or env config edited after the run finished — the failure the add-a-file rule exists to
   prevent.
4. **Action nvec ≠ `cfg.action_nvec` → refuse.** Catches a pre-super checkpoint, whose observation
   still matches.

Guards 1 and 2 fire **before** `MaskablePPO.load`, which is what makes the whole set testable from
a temp directory holding nothing but a `train.yaml` — `runs/` is gitignored, so a test that needed
the 44 MB checkpoint could not run on a clean checkout.

**The mask is the shadow's, not a second copy.** `act(obs, attack_legal)` takes
`ShadowHero.attack_mask()` verbatim and concatenates it after an all-True move half, giving the
`[move (17), attack (3)]` layout `wrappers/sb3_vecenv.py:action_masks` produced on every training
step. The formula (`alive & cd <= 0 & dash_t <= 0`, plus ammo for attack and charge for super)
lives in the shadow, which owns the timers it reads. One extra check: an all-False attack column
is **rejected**, because `hero.action_mask` makes no-fire unconditionally legal and MaskablePPO's
response to a degenerate categorical is NaN logits rather than an exception.

`Decision` carries the mask it was given alongside the action, so telemetry can tell *"the policy
chose not to fire"* from *"the policy could not"* — identical in the action alone, and completely
different problems when a live run goes quiet.

**MEASURED 2026-09-08: the policy runs on the CPU.** Batch of 1, 200 decisions after warmup:

| device | ms / decision |
|---|---|
| **cpu** | **0.96** |
| cuda | 7.56 |

The network is 1.04M parameters, so a single-observation forward pass is pure launch-and-transfer
overhead and the GPU loses by 8×. (Machine was in use; the absolute numbers are loose, the ratio
is not.) This is the opposite of the natural assumption and it is doubly convenient — the 250 ms
budget in §7.1 is untouched either way, and keeping the policy off the GPU leaves the whole 16 GB
of VRAM to the two detectors, which is where §7's real pressure is.

**What it deliberately does not own: the interlock.** `act()` returns a decision even when the
shadow is desynced or the match is over, because whether to *emit* it is `loop.py`'s job (§8) and
half an interlock here would make it possible to believe the wrong half is in force.

`deterministic=True` by default, matching `eval.deterministic: true` — the setting `best_model.zip`
was *selected* under. **One hazard for `loop.py`:** a deterministic policy whose input stops
changing repeats one action forever. In the sim the state moves on and it self-corrects; against a
frozen capture it does not, so a stuck-input detector belongs in the loop, not here.

---

## 7. Resource budget

Against 16 GB VRAM / 32 GB RAM / 1 TB free, targeting **one** agent.

| Component | VRAM | RAM | Notes |
|---|---|---|---|
| BlueStacks | — | 2–4 GB | Emulator's own allocation. |
| YOLO detector(s) | ~1–2 GB | — | Two models: entity at 4 Hz, projectile at 20 Hz (§9.5). **Now on CUDA** (§9.12); device VRAM rises ~0.8 GB with both sessions warm and is flat over 900 inferences. |
| Policy | <100 MB | — | Small MLP + CNN extractor. |
| Frame buffers | — | <200 MB | See the hard rule below. |
| Occupancy map | — | tens of MB | Tile-space, small. |

Comfortable. The risk is not steady-state, it is **unbounded growth**, so three hard rules:

1. **Never retain a frame history.** At 20 Hz, 1920×1080 BGR is ~6 MB/frame — 124 MB/s, ~22 GB
   for a three-minute match. Frame dumping is **opt-in, capped, and off by default.** A ring
   buffer of the last N frames (N ≤ 40) is the only standing retention.
2. **Bounded telemetry.** Per-decision logging goes to a fixed-size ring or an append-only CSV of
   scalars, never per-frame images or per-frame tensors.
3. **No parallelism anywhere.** One capture, one model set, one policy. No vectorised env, no
   worker pool, no async prefetch. The user has stated single-agent is the requirement; honouring
   it keeps the whole budget trivially satisfied.

### 7.1 Timing, measured

Profiled live at 1440p with the real pipeline, steady state, on this machine:

| 20 Hz perception stage | mean | notes |
|---|---|---|
| `mss` grab, 2560×1440 | 15.7 ms | the floor without DXGI |
| normalize + copy | 1.2 ms | was **10.9 ms** — see below |
| resize → 2002×1126 | 2.8 ms | `INTER_AREA` |
| match gate | 0.4 ms | was **5.5 ms** — see below |
| rectify | 0.8 ms | |
| odometry | 4.5 ms | 11 correlation windows |
| **total** | **25.4 ms** | against a **50 ms** budget at 20 Hz |

Plus, on the 4 Hz decision tick: HP read (~1.7 ms measured in `hp_detection`), tracking
(microseconds), the grid (microseconds), assembly and policy, on top of the detectors below.

> **The detectors are now on the GPU (§9.12 resolved), and these are the measured numbers**,
> 40 calls each after a warm-up, `predict()` end to end including letterbox and NMS:
>
> | detector | input | CPU (before) | **CUDA (now)** |
> |---|---|---|---|
> | entity | 640×640 | 26.1 ms | **7.2 ms** (p90 7.4, max 8.2) |
> | projectile | 960×960 | 47.2 ms | **10.8 ms** (p90 11.0, max 11.1) |
>
> 3.6× and 4.4×, which settles §9.5's unpaid bill. A **plain 20 Hz tick** is now 25.4 + 10.8 =
> **36.2 ms** against 50 — the projectile pass fits on every tick, which is what §9.5 asked for.
> A **decision tick** is 25.4 + 10.8 + 7.2 + 1.7 ≈ **45.1 ms** before assembly and policy, so
> that tick is still the one at the edge, with roughly 5 ms for the two stages not yet built.
>
> **If that tips over, the overrun is affordable, and it is worth saying why rather than
> treating 50 ms as a hard wall.** The budget exists to keep `odometry.max_shift_tiles: 2.0` satisfied, and that bound is
> per-frame: Mortis's dash is the fastest thing on screen at 0.89 tiles per 50 ms frame, so 2.0
> tiles is **2.25× a normal frame interval** — an occasional 60 ms tick has margin, a sustained
> 110 ms one does not. The thing to watch is therefore the decision tick's *tail*, not its mean,
> and the loop already has `FrameSource.read_seconds` and `DeployCapture.resize_seconds` to
> attribute it. If it does need shaving, the first move is batching the two detectors into one
> session call, not dropping the decision rate.
>
> The first inference after construction is slow — the 5070 Ti is Blackwell (sm_120) and the
> wheel's kernels JIT from PTX — so the loop should warm both detectors on a dummy frame before
> the match gate opens rather than paying it on the first real decision.

**Two fixes got it there, and both were removing redundant work rather than tuning.** The first
cost 12.2 ms and was in `brawl_vision`: `ScreenCapture._raw` dropped mss's alpha channel with
`[..., :3]`, which returns a *channel-strided* view whose every later `ascontiguousarray` gathers
three bytes out of every four by hand — 10.78 ms per 1440p frame, against 0.73 ms for a
bit-identical `cvtColor`. The second cost 5.1 ms: the match gate computed Sobel gradients over
all 2.25 M pixels to score one 35 px circle, where a 116×116 window gives the same answer 160×
cheaper. Neither was a micro-optimisation — together they were a third of the frame budget, and
overrunning that budget breaks `odometry.max_shift_tiles`, which is a correctness failure rather
than a slow one.

**The loop must still measure and report its own margin** — `FrameSource.read_seconds` and
`DeployCapture.resize_seconds` exist precisely so "the pipeline is slow" can be told apart from
"reading frames is slow."

---

## 8. Failure modes and the interlock

| Failure | Detection | Response |
|---|---|---|
| Match ended | §5 ring score | Release contacts, stop input. **Primary interlock.** |
| Window lost focus / moved | `window.py` geometry check | Release, stop, warn. |
| Capture stall | `read_seconds` / frame timestamp gap | Release, stop. |
| Movement contact dropped | Rest-anchor ring score (§4.2) | Re-acquire at anchor. |
| Odometry lost (cut) | Existing `min_agreement_ratio` gate | Hold last action, re-seed; stop if sustained. |
| Policy exception | try/except in `loop.py` | Release, stop, re-raise. |
| Shadow desync | Ammo canary — `ShadowHero.check_ammo`, three consecutive strikes (§6.3) | `resync()` from CV, reseeding unobservable timers to *not ready*; log it. Repeated trips inside one match: release, stop. |

Everything routes to the same place: **release all contacts and stop emitting.** Fail closed. A
stopped agent is recoverable by hand; an agent mashing inputs into a menu is not.

---

## 9. Open decisions and blockers

**Blockers.**

1. ~~**Does Nulls Brawl act on injected touches?**~~ **Resolved — yes.** Run in a training
   ground rather than a live match, which was the better call: no bots hunting the agent, so the
   test could run long and be read frame by frame. Mortis walked on command, the stick teleported
   to our anchor, four taps did 2000 DPS and killed a dummy, and the movement contact stayed down
   throughout. §4.1. **This was the last hard blocker in the control path.**

2. ~~**Input backend.**~~ **Resolved** — §4.1. Candidate A, 16 directions, no retrain.
3. ~~**The `self` group.**~~ **Resolved** — §6.3. Proprioception supplies it. No `self` field
   forced a retrain; one (`hero.cubes`) leaves the group anyway under §9.8.

4. ~~**Capture resolution.**~~ **Resolved — stay at 1440p and resize in Python**, per the
   operator's call (switching the display per session is too disruptive). The resize target is
   **not** 1080p, though, and that turned out to matter more than the original question:

   **Resize to 2002×1126, the viewport the vision stack was calibrated at.** Both targets are
   downscales from a 1440p grab, so neither invents pixels — but 1080p would need a rescaled
   `CameraModel` carried forever and would render every brawler 4% smaller than the frames the
   detector was trained on, while 2002×1126 lets the shipped homography, HUD mask and thresholds
   apply completely unchanged. `capture.DeployCapture` does this; it resizes the *normalized*
   frame to a viewport rather than multiplying by a constant, so a trimmed content box (measured:
   a fullscreen grab normalizes to 2558×1439, not 2560×1440) or a windowed emulator absorbs
   automatically.

   **The result that made this safe rather than merely tidy: the iOS calibration transfers to
   BlueStacks unmodified.** Rectifying a BlueStacks frame with the shipped homography puts wall
   blocks exactly on the 48 px tile grid across the entire frame — correct perspective *and*
   correct absolute scale, meaning the game's virtual camera does not depend on the device.
   Odometry on that footage: inlier ratio 1.000 at both median and 1st percentile, response 0.92,
   59/59 frames `ok`. Live at 1440p: refined gate score 0.908, odometry inlier 1.000.

   **One hazard this exposed: device pixels and viewport pixels are different spaces.** Touch
   injection addresses the Android screen's own 1920×1080; the gate reads a 2002×1126 frame. They
   coincided exactly while the fixture was 1080p — a coincidence that passes every test and then
   breaks. `Calibration.button()` and `Calibration.viewport_button()` are now separate.

**Decisions — all resolved as of the §6.1 review; kept for the reasoning, not as a worklist.**

4. ~~**`n_move_bins` 16 vs 8.**~~ **Moot** — it was only forced by Candidate B, and the device's
   16 native slots removed the need for a keymap entirely. 16 bins stay; checkpoints stay valid.
5. ~~**Projectile detection rate (§6.1). Resolved — 4 Hz.**~~ **REOPENED and re-resolved — 20 Hz**,
   because 4 Hz cannot produce `projectiles.vel` at all and the original decision never checked
   that. Measured on `showdown_alternate_map.mp4` (60 fps, 1500 frames, real combat — the
   BlueStacks fixture turned out to be mostly Mortis wandering alone, and measuring it first
   produced "98% of frames have no projectile", which was a fact about the clip):

   | sample rate | mean samples per projectile | **≥2 samples** | ≥3 |
   |---|---|---|---|
   | 60 Hz | 6.99 | 71% | 56% |
   | 30 Hz | 3.49 | 53% | 42% |
   | **20 Hz** | **2.33** | **44%** | 32% |
   | 10 Hz | 1.16 | 27% | 14% |
   | **4 Hz** | **0.47** | **7%** | 1% |

   Two observations is the minimum for a velocity; one is a position with no direction attached.
   **At the 4 Hz this item originally chose, 93% of projectiles never get a second look** — so
   `projectiles.vel` and `projectiles.time_to_closest` would be zero or noise for almost every
   projectile, and `time_to_closest` also decides which 12 projectiles reach the policy at all.
   The cause is short lifetimes: median 50 ms on screen, p75 151 ms, p90 299 ms.

   20 Hz rather than 30 or 60 because 20 Hz is the capture rate §1.2 already fixed for odometry
   reasons; going above it means raising the whole loop, which is a much larger change than this
   field is worth. 44% is not good, and the honest reading is that **fast projectiles are partly
   unobservable at any rate this loop will run at** — a limitation to record rather than engineer
   away.

   Note the original rationale — "20 Hz is 5× the YOLO cost" — was already wrong when it was
   written (item 12), and the bill it left behind is now **paid**: the projectile detector runs
   at 10.8 ms on CUDA rather than 47.2 ms on the CPU, so a plain 20 Hz tick fits its 50 ms
   budget with room (§7.1, §9.12). The 44% observability ceiling above is unaffected — it is a
   property of how briefly a projectile exists, not of how fast we can look.
6. ~~**`meta.time_frac` (§6.3).**~~ **Resolved — wall-clock ÷ 150 s**, operator's call.
   `core/observation.py:434` defines the field as `step_count / max_episode_steps`; with
   `configs/default.yaml`'s `max_episode_steps: 3000` at `dt: 0.05` that is elapsed match time
   over a 150-second budget, running 0 → 1. So the deployed value is seconds since the match gate
   went true, divided by 150. Its marginal content is small either way — the zone schedule, the
   other thing that separates late-match from early-match, is already exposed as
   `zone.next_shrink_in` and `zone.safe_area_frac`.

   **Clamp it to [0, 1].** Real matches will not all run 150 s (the operator flagged this, and it
   is right). Running short is harmless: the policy simply never sees the top of the range, which
   is the same thing that happens in the sim whenever a match ends early. Running LONG is the
   hazard, because `step_count / max_episode_steps` cannot exceed 1 in training — the episode
   truncates there — so a 200-second match would feed 1.33 into a column whose entire training
   distribution is bounded by 1. That is out-of-distribution input, and clamping costs one call.

7. ~~**Which brawler?**~~ **Resolved — Mortis**, which makes §6.3's shadow state load-bearing
   rather than incidental: dash and long-dash are his whole kit.

8. ~~**The four fields with no reader behind them.**~~ **Resolved — drop them**, operator's call.
   `entities.cubes`, `entities.can_attack` and `entities.dashing` come out because nothing on
   screen reports another player's ammo state or cube count, so the alternative was feeding a
   constant into three columns the policy was trained to trust — exactly what
   `agent_obs_lowinfo.yaml`'s own rule forbids. **`hero.cubes` comes out too**, for a different
   reason: it *is* recoverable (the pip row is right there in the same box stack as the HP bar),
   but the operator does not want a reader built for it, and a field nobody intends to supply is
   in the same position as one nobody can. Dropping it is what keeps that honest — the one thing
   not on the table is leaving it in the spec and feeding it zero. This narrows the observation
   and so invalidates the current checkpoint; it is a retrain, not a swap, and it belongs in a
   new file — `configs/agent_obs_deploy.yaml`, written and tested — rather than an edit to
   `agent_obs_lowinfo.yaml`, which the existing run's `train.yaml` still points at by path.
   `tests/test_configs_files.py` pins the new file's relationship to its parent in both
   directions: nothing added, and exactly these four removed.

9. ~~**`entities.rel_vel` — drop it too?**~~ **Resolved — it stays**, on the operator's own
   condition: *"if rel_vel is difficult to do in CV then we can drop it."* It is not difficult.
   §6.1 already produces it, it is tested, and §6.1's measurements show the error is both smaller
   than first reported (8% past the physical bound, not 38% — the odometry term cancels in the
   subtraction) and far inside what the policy tolerates (zeroing the column costs 1.3 pp against
   a 2.6 pp standard error).

   The ablation is not an argument for keeping it either — it could not detect the policy using
   the field at all. What decides it is cost: the column is already built and free to supply, and
   it is the only enemy field carrying motion, which is what any future need to lead a target
   would be built on. **One line in `configs/agent_obs_deploy.yaml` reverses this** if the
   operator would rather narrow further; the file marks the line.

10. ~~**`hero.hp_frac` / `entities.hp_frac` — the denominator nobody can see.**~~ **Resolved —
    read absolute HP instead**, operator's call, and the cleanest of these decisions because it
    costs nothing. `hp_frac` is `hp / max_hp`, and `cubes.hp_per_cube` adds a FLAT +400 to max_hp
    for the hero and for every enemy, with nothing on screen saying who holds how many — the same
    fact that dropped `entities.cubes` in item 8. Dividing by the kit constant in
    `configs/brawlers.yaml` is correct only until the first cube is collected, and wrong silently
    thereafter.

    Both fields become `hero.hp` / `entities.hp`, which is what `hp_detection/read.py` has always
    produced: it ships a NUMERAL and explicitly refuses to ship a fill ratio. So the swap needs no
    new reader, and the spec width does not move (`self` 24, `enemies` (9, 9)).

    It is also a small information **gain**. Base HP spans 6000–10000 across the roster, so "50%"
    is 2 Mortis hits or 5 depending on who it is; the absolute number separates them. And since
    cubes push HP above every base value, a large reading is itself evidence of a cube-fed enemy —
    who also hits ~10% harder per cube. `hp_frac` threw that away.

    Scale: `obs_select` normalizes `hp` by `_NORM_HP_SCALE` (20000), sized by the same rule
    `_NORM_SPEED_SCALE` follows — the sim's largest possible HP, `(10000 + 16×400) × 1.25 =
    20500`, lands at 1.03, and a fresh Mortis at 0.40. `meta.n_enemies_alive` picked up
    `_NORM_COUNT_SCALE` in the same change, having been the largest raw magnitude in the vector at
    up to 9.

    **Two consequences to know.** This rescales `projectiles.damage` and the `cubes` fields in
    `agent_obs.yaml` / `agent_obs_lowinfo.yaml`, so a checkpoint trained under either before this
    change cannot be resumed after it (values change, not widths). And enemy numeral read rates on
    BlueStacks footage are **not yet measured** — the reader handles both labels and the tests
    cover both, but the `entities.hp` column has no measured miss rate behind it the way ammo now
    does.

11. ~~**`grid.box` / `grid.pickup` — two channels with no detector (§6.2).**~~ **Resolved — drop
    both for this training run**, operator's call, *"although I will return and add them later."*
    The grid goes from `(10, 13, 21)` to `(8, 13, 21)`.

    **This was not a free drop, and the paired half is the part worth remembering.**
    `configs/train.yaml` set `reward.cube_pickup: 0.5`, paying the agent every time it walked over
    a power cube. Keeping that reward while removing the two channels that show where cubes are
    trains a policy to chase something it cannot see — reward with no observable cause, which
    lands as noise on every other gradient. So `cube_pickup` is now `0.0`. Cubes still exist in
    the sim, still add HP and damage, and the agent still benefits from collecting one; it is
    just no longer paid a bonus for an event it has no way to have aimed at.

    `train.yaml` is SHARED across all three obs specs, so this zero also applies to full-info
    runs. Those can restore it per-run with `--set reward.cube_pickup=0.5` and the comment in the
    file says so. `tests/test_configs_files.py` pins the two together in **both** directions —
    restoring the channels without the reward, or the reward without the channels, fails.

12. ~~**⚠ OPEN BLOCKER — both ONNX detectors are running on the CPU.**~~ **RESOLVED — both now
    report `CUDAExecutionProvider`, at 3.6× and 4.4× the speed** (entity 26.1 → **7.2 ms**,
    projectile 47.2 → **10.8 ms**; §7.1). It took **four** fixes, and the last three are the ones
    worth keeping, because each fails *silently*, or loudly in a place that does not look like
    ONNX. Device VRAM rises ~0.8 GB with both sessions warm and does not grow over 900 inferences
    (device-wide delta; NVML reports 0 for per-process memory on Windows/WDDM, so per-process
    attribution is not available here).

    **Fix 1 — the install was left broken.** `pip install "onnxruntime-gpu[cuda,cudnn]"` followed
    by `pip uninstall -y onnxruntime` deletes the shared files: both wheels write into
    `site-packages/onnxruntime/`, so the uninstall took the Python package with it and left only
    the two provider DLLs `onnxruntime-gpu` alone ships. `import onnxruntime` then yields an
    **empty namespace package** — no `__version__`, no `get_available_providers`, no
    `InferenceSession` — while `pip list` still shows `onnxruntime-gpu 1.29.0`. Repaired with
    `pip install --force-reinstall --no-deps onnxruntime-gpu==1.29.0`, which restores the package
    without re-resolving the ~3 GB of `nvidia/*` wheels. **Order matters: uninstall the CPU wheel
    BEFORE installing the GPU one, never after.**

    **Fix 2 — `preload_dlls()` is not called at import, and nothing tells you.**
    `onnxruntime-gpu` ships no CUDA libraries; the `[cuda,cudnn]` extras supply them as
    `nvidia/*` wheels and `onnxruntime.preload_dlls()` is the function that puts them on the DLL
    search path. Its own docstring says the call can be skipped when torch is imported first —
    but only when torch's CUDA MAJOR matches, and it does not here (torch `+cu128`, ORT built
    against CUDA 13), so `cublasLt64_12.dll` is present and `cublasLt64_13.dll` is not. Without
    the call, session creation logs `Error loading onnxruntime_providers_cuda.dll which depends
    on cublasLt64_13.dll which is missing`, falls back to CPU, and raises nothing. It now lives in
    `detector.preload_cuda_dlls()`, called before every session, and `ObjectDetector` **raises**
    when `device: cuda` was asked for and the session came back CPU — the silent fallback is the
    whole failure mode.

    **Fix 3 — the preload breaks torch unless torch is imported first.** Putting CUDA 13's cuDNN
    on the search path makes torch resolve its own `torch/lib/cudnn_cnn64_9.dll` against it
    instead of the CUDA 12 copy beside it, and torch dies at IMPORT with `OSError: [WinError 127]
    The specified procedure could not be found` — a thousand lines from anything to do with
    detection. Caught by 11 unrelated `test_vision_detector.py` failures that appeared only when
    a detector was constructed earlier in the same session. `preload_cuda_dlls()` imports torch
    first so it binds its own libraries while the path is still clean.

    **Fix 4 — importing torch first is only half of it: the preload must then SKIP cuDNN.** Fix 3
    was verified by checking `ObjectDetector.provider` and a `torch.nn.Conv2d`, and both passed —
    but a provider name is not an inference. Re-running the *latency* script afterwards showed
    what the provider check could not: the entity detector had gone from 7.9 to 13.3 ms and
    printed `EP Error ... Failed to initialize CUDNN Frontend ... HEURISTIC_QUERY_FAILED`, and the
    **projectile detector failed outright** with `CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE`
    on `/model.0/conv/Conv`. The cause is that there is exactly **one `cudnn64_9.dll` per
    process**, and its engine sublibraries must come from the same install. With torch loaded that
    slot is already torch's CUDA 12 copy and cannot be replaced; `preload_dlls()` then loads CUDA
    13's *sublibraries* next to it and the mixed set is worse than either whole one. The fix is
    one keyword: **`preload_dlls(cudnn=False)` when torch is present**, leaving cuDNN entirely to
    torch, and `cudnn=True` when it is not (a torch-free process has no cuDNN otherwise). Every
    other CUDA library is version-suffixed — `cublasLt64_13.dll` vs `_12` — so both majors coexist
    and the preload is pure gain there.

    Measured, all four combinations, 20 calls each after warm-up:

    | preload | entity | projectile | torch |
    |---|---|---|---|
    | none | CPU 26.1 ms | CPU 47.2 ms | fine |
    | `preload_dlls()`, no torch in process | 7.4 ms | 14.2 ms | **cannot import** |
    | torch, then `preload_dlls()` | 13.3 ms | **fails** | fine |
    | torch, then `preload_dlls(cudnn=False)` | **7.2 ms** | **10.8 ms** | fine |

    The live loop needs both halves in one process — the detectors are ONNX, the policy is torch —
    so the last row is the only one that is a configuration rather than a trap. **ORT ends up
    running on a cuDNN built for the other CUDA major, which sounds worse than it measures:**
    against its own CPU session on real BlueStacks frames, the entity model agrees to **1.2 px and
    3e-4 confidence** and the projectile model to 0.135 px and 5e-3, with one count difference in
    250 frames on a box sitting at the confidence threshold. If that ever stops being true, the
    clean fix is `torch==2.11.0+cu130` from PyTorch's cu130 index — same version, matching major —
    which is a ~2.5 GB reinstall of a shared environment and so is the operator's call, not mine.
    ONNX Runtime prints a CUDA-major-mismatch warning at each preload; it is accurate and
    harmless.

    Both `detector.device` and `projectile.device` are already `auto` in `configs/vision.yaml`, so
    nothing needed a config change. The original diagnosis is kept below because the way this
    fails is worth not rediscovering.

    ---

    **Original finding.** Found while measuring item 5, and it is the reason that item could not
    simply be implemented:

    ```
    ort.get_available_providers()  ->  ['AzureExecutionProvider', 'CPUExecutionProvider']
    torch.cuda.is_available()      ->  True   (RTX 5070 Ti, torch 2.11.0+cu128)
    ```

    There is no CUDA execution provider in the installed `onnxruntime`, so every detector
    inference runs on the CPU while the GPU sits idle. Measured per-frame latency on this
    (deliberately contended) machine: **entity detector 26.1 ms median at 640×640, projectile
    detector 47.2 ms median at 960×960** (p90 51.6, max 60.8). Against the 50 ms tick that 20 Hz
    means, the existing loop alone is already at 25.4 ms and adding the projectile pass takes it
    to **72.6 ms — 1.45× over budget, before any of §6.2's grid work**.

    **`pip install onnxruntime-gpu` alone does not fix it, and the way it fails is the part worth
    recording.** That was tried. Afterwards:

    ```
    ort.get_available_providers()  ->  ['TensorrtExecutionProvider', 'CUDAExecutionProvider', ...]
    ObjectDetector.from_config(cfg).provider  ->  'CPUExecutionProvider'
    ```

    The provider list is a statement about the *build*, not about what can actually run. Session
    creation logs `Error loading onnxruntime_providers_cuda.dll which depends on
    cublasLt64_13.dll which is missing`, falls back to CPU, and hands back a perfectly working
    detector with identical latency. **Nothing raises.** So the only honest check is
    `ObjectDetector.provider` on a constructed detector — `get_available_providers()` will lie.

    The cause: `onnxruntime-gpu` 1.29 is built against **CUDA 13**, the wheel declares no
    dependency on any CUDA library, and this machine has no CUDA toolkit installed at all
    (`nvidia-smi` works because that is the driver, not the toolkit). Torch does not help either —
    it works by bundling its own CUDA **12** in `torch/lib`, so `cublasLt64_12.dll` is present and
    `cublasLt64_13.dll` is not. The libraries come from extras that a bare install skips:

    ```bash
    pip uninstall -y onnxruntime
    pip install "onnxruntime-gpu[cuda,cudnn]"
    ```

    `[cuda,cudnn]` pulls NVIDIA's own wheels (~3 GB) into `site-packages/nvidia/`, which
    `onnxruntime/__init__.py:202` explicitly knows how to find — that is what makes this work with
    no system toolkit. The uninstall matters independently: both wheels write the same files under
    `site-packages/onnxruntime/`, both are installed right now, and whichever pip touched last is
    the one running. `pyproject.toml` now carries the whole recipe as a `detect-gpu` extra rather
    than a comment, so a fresh clone cannot repeat the bare-install mistake.

    The prediction recorded here at the time — "single-digit milliseconds on a 5070 Ti" — was
    close: 7.2 ms for the entity model, 10.8 ms for the larger projectile one. The Blackwell
    (sm_120) warm-up concern was also real, which is why §7.1 now says to warm both detectors on a
    dummy frame before the match gate opens.

13. ~~**Small, open, not blocking — `hp.super_track_px` is centred on the wrong
    resolution.**~~ **CLOSED 2026-09-08, operator's call: not worth doing.** *"I don't
    expect the capture resolution to change at all. This current configuration on
    BlueStacks is what I expect to test and deploy on."* The provenance comments stay
    corrected; the constant stays at 116 and `hero_bars.py` does not get `REF_H`. Reopen
    only if a second capture resolution ever appears. The original finding: Found
    while building §6.4's third reader. The ammo and super constants are documented as measured
    "at the calibrated 2002×1126 viewport" and were in fact measured at **1920×1080**:
    `ClipReader` trims to 16:9 and never resizes, and `bluestacks-example-new.mp4` is natively
    16:9. Deployment, meanwhile, resizes to 2002×1126 (item 4).

    The readout stack scales with viewport height, measured directly rather than assumed:

    ```
                       @1080   @1126
    super track px      116     120
    ammo pip px (p50)    33      34
    ```

    Neither constant is wrong in effect. `hp.ammo_pip_px` 34 sits inside the 33–35 spread at both
    heights, and `hp.super_track_px` is a **±22 sanity gate that is never divided by** — the super
    reader's denominator is the extent the frame itself found — so a 4 px offset costs nothing
    today. The decision is only whether to recentre it on 120, which makes the tolerance mean what
    it says without changing a single reading. **Not** free-standing: `hero_bars.py` does no
    scaling at all, so the deeper version of this is whether to give it the `REF_H` treatment
    `hud.py` uses. Worth doing only if a third capture resolution ever appears; the provenance
    comments in `hero_bars.py`, `configs/vision.yaml` and §6.4 are corrected either way.

14. **⚠ OPEN — the `zone` group's four scalars have no supplier, and it is the last group that
    doesn't.** Surfaced by finishing §6.2: with the grid built, `self`, `enemies`, `projectiles`
    and `grid` all have suppliers and `zone` is what is left.

    **Status, 2026-09-08.** What is still open here is the *estimator*, which lands with
    `assemble.py`. The measurement is done (§9.15), the decision is taken (deploy on the current
    checkpoint), and the training-side shape a future retrain needs is built (§9.16). Read this
    item for why the four fields are hard; read §9.16 for what was done about it.

    ```yaml
    fields: [zone.hero_margin, zone.active, zone.safe_area_frac, zone.next_shrink_in]
    ```

    Seven columns, not four: `core/observation.py:398` defines `hero_margin` as the signed
    distance to **each of the four edges** of the `zone_lo`/`zone_hi` rectangle. What exists is
    `terrain/zone.py`, which answers "is this cell gassed" per frame, and §6.2's `GasMap`, which
    accumulates that into the world frame. Neither is a rectangle, and the rectangle is what three
    of these four fields are made of.

    | field | width | status |
    |---|---|---|
    | `zone.active` | 1 | **free.** The sim's own value is the constant `cfg.zone_enabled` — `True` in every training sample. Showdown always has gas, so `True` is not a fabrication, it is the same constant. |
    | `zone.hero_margin` | 4 | **needs a fitted rect.** All four edges, so a distance transform on `GasMap.gassed` gives the nearest one and not the other three. |
    | `zone.safe_area_frac` | 1 | **needs a rect and a map area.** Deployment has no map extent — only a 128×128 canvas centred on wherever tracking started. |
    | `zone.next_shrink_in` | 1 | **unobservable.** `zone_next_t - time`; nothing on screen counts down to a shrink, and Nulls Brawl's schedule is its own. |

    **One estimator would cover six of the seven columns**: fit an axis-aligned rectangle to the
    clear region of `GasMap`, which is exactly the shape the sim's zone has, and remember the
    largest one ever seen as the map extent. That gives all four margins and `safe_area_frac`,
    and it degrades in the right direction — early in a match the gas has not started and the
    fitted rect is the explored region, which understates the map. Whether the real game's safe
    area is rectangular enough for this is a question for footage, not for this document.

    `next_shrink_in` is the one with no path. Its training range is `[0, step_seconds]` = `[0,
    1.5]` and it is a sawtooth, so a constant would be a lie in a column the policy can actually
    use — **the §9.8 failure exactly**. The honest options are the §9.8 ones: build the estimator
    and **drop `next_shrink_in`** (a fourth spec file and another from-scratch run), or keep the
    group whole and accept one fabricated column. §9.8's rule says the first.

    Flagged rather than decided: it is the operator's call, and the retrain is the expensive half.
    **It does not block `assemble.py`** — the other four groups have suppliers today and can be
    assembled and tested first, with `zone` the last thing wired in.

    **MEASURED 2026-09-07, and it moves both halves of this item.** Driver:
    `scratchpad/zone_shape.py` — rectify → odometry → `detect_zone` → §6.2's `GasMap`, at the
    20 Hz `observe_zone` runs at, with no classifier in the loop.

    *First, the deployment-fidelity clip cannot answer this question at all.*
    `bluestacks-example-new.mp4` contains **no poison**: 2 gassed cells at t=2.0 s of a 34.7 s
    gameplay span, dropped by the next segment reset, and nothing after. The match ends before the
    gas starts. The only clip we own that contains the phenomenon is `day10_gameplay.mp4` (iOS,
    77 s), where gas first appears at **30.8 s** and accumulates to **276 world cells**.

    *The safe region is not a clean rectangle.* Taking the bounding box of the clear cells and
    asking what fraction of the SEEN cells inside it are gassed — zero for a perfect rect — gives
    **median 0.046, p90 0.120, max 0.130**, and it grows with the gas: 0.047 from 34–58 s, 0.130
    by 77 s. So a fitted rect is an approximation with a several-tile error at the edges, not a
    reconstruction. This measurement does not separate the three candidate causes (a ragged gas
    front, the poison being drawn as discrete clouds rather than a border band, or sticky
    misdetections accumulating in `GasMap`), and separating them needs footage that reaches the
    late game.

    *`next_shrink_in` is worse than unobservable — it is unobservable for a structural reason.*
    The cadence idea (time the shrink events and dead-reckon the countdown between them, the way
    §6.3's shadow dead-reckons the hero's timers) does not survive contact: new gassed cells
    arrive on **137 of 1543 ticks with a median gap of 0.05 s**, i.e. on consecutive frames. A
    world map that only ever grows cannot tell "the gas advanced" from "the camera panned over
    gas that was already there", and the camera pans constantly because it follows the hero. There
    is no shrink *event* to time.

    So the three routes, priced:

    | route | cost | what it gives up |
    |---|---|---|
    | fit the rect, drop `next_shrink_in` | a fourth spec file + a from-scratch run (~9.4 h at the last run's rate) | the countdown, and 4–13% edge error on the margins |
    | drop the whole `zone` group | the same run | all gas awareness — and `configs/default.yaml` calls the zone the designed answer to disengagement |
    | keep the group whole | free | one fabricated column, which is the §9.8 failure this project keeps refusing |

    §9.15 measures what the rows of that table are actually worth, and it changes the answer:
    the group as a whole carries **10.4 pp of win rate**, so the second row is out — but
    `next_shrink_in` alone carries **1.5 pp, one standard error**, so the first row's retrain buys
    little that pinning the column at zero does not. The recommendation is there.

    **RE-MEASURED 2026-09-08 on `bluestacks-example-zone.mp4`** — 66 s at 1920×1080, recorded by
    the operator specifically to contain the zone spawning and closing in. It settles the half of
    this that footage can settle and refutes the other half.

    *What it settles: the detector transfers.* The gas starts 21 s into the gameplay segment and
    `terrain/zone.py`'s green band — measured in Phase G on iOS clips — **fires on it unchanged on
    BlueStacks**, on 988 of 1983 ticks (essentially every tick after onset in which the gas is on
    screen). On this map the bushes are RED and the only bright green on screen is the poison, so
    the confusion the band's own comment warns about does not arise here. Colour is not a problem.

    *What it refutes: the fitted rect.* Over the whole clip the fitted safe box **never moves
    inward once**, while 19–36% of the seen cells inside it are known-gassed (median 0.191, max
    0.356 — four times the iOS clip's figure). The cause is not a non-rectangular zone; it is
    `GasMap` itself. **The map only learns "gassed" for cells that are on screen, and the poison
    advances precisely into the places the hero has walked away from.** A cell seen early and never
    revisited stays `clear` for the rest of the match. So the rect is biased outward, and biased
    outward *specifically in the direction the gas is coming from* — the worst possible direction
    for a margin the agent is supposed to flee along.

    *What it does not settle: the schedule.* Timing the shrinks needs a front that stays on screen,
    and this camera follows the hero: the deepest visible gas moves 7 tiles in 3 s at t≈21–24 s,
    which is the camera revealing a band it had only partly shown, not the gas advancing that fast.
    Reveal and advance are not separable from a moving camera at any accumulation rule, which is
    the same wall §9.14 hit on the iOS clip, now confirmed on the real capture path.

    **So the supplier that survives is a LOCAL one, not a global rect.** `zone.hero_margin` is four
    signed distances; measure them as distance from the hero to the nearest gassed cell along ±x
    and ±y in `GasMap`, and the estimator inherits exactly the right failure mode — it is accurate
    when the gas is near (which is when the camera is showing it, and when the value is small and
    the policy cares) and optimistic when the gas is far (when the value is large and §9.15 says a
    few tiles of error costs nothing). A global rect has the opposite profile. `safe_area_frac`
    keeps no such escape: it needs a map extent deployment never learns, and §9.15 prices ±0.05 at
    −2.8 pp, which makes it the weakest column in the group by a clear margin.

15. ~~**How much does the trained policy actually use the `zone` group?**~~ **MEASURED — the
    group is worth ~10 points of win rate, and the countdown inside it is worth nothing.** Same
    method as the `entities.rel_vel` ablation (§9.9): `TierEvaluator` reseeds identical scenarios,
    only the observation the policy reads is perturbed. 250 episodes per tier, 1000 envs,
    `runs/mortis_deploy-20260907-041522/best_model.zip`. Driver: `scratchpad/zone_ablation.py`.

    | condition | easy | medium | hard | elite | overall | vs baseline |
    |---|---|---|---|---|---|---|
    | baseline | 0.872 | 0.748 | 0.728 | 0.588 | **0.734** | — |
    | `next_shrink_in` = 0 | 0.888 | 0.756 | 0.676 | 0.556 | 0.719 | −1.5 pp |
    | `next_shrink_in` ~ U(0, 1.5) | 0.888 | 0.668 | 0.704 | 0.500 | 0.690 | −4.4 pp |
    | `hero_margin` + N(0, 1 tile) | 0.880 | 0.768 | 0.736 | 0.600 | 0.746 | +1.2 pp |
    | `hero_margin` + N(0, 3 tiles) | 0.900 | 0.724 | 0.720 | 0.504 | 0.712 | −2.2 pp |
    | `safe_area_frac` + N(0, 0.05) | 0.892 | 0.676 | 0.732 | 0.524 | 0.706 | −2.8 pp |
    | **whole group = 0** | 0.820 | 0.636 | 0.604 | 0.460 | **0.630** | **−10.4 pp** |

    The standard error on an overall figure is **~1.5 pp** (250 episodes per tier at p≈0.7 gives
    2.9 pp per tier; the overall is a mean of four). Read the table against that:

    * **Zeroing the whole group costs 10.4 pp — seven standard errors.** The policy uses the zone.
      Whatever else happens, the group is not droppable on the grounds that it is ignored.
    * **Zeroing `next_shrink_in` alone costs 1.5 pp — one standard error, i.e. nothing this
      measurement can see.** The countdown is the one column the policy appears not to read.
    * **Corrupting the countdown is worse than blanking it** (−4.4 pp vs −1.5 pp, the two about
      2 SE apart). Suggestive rather than established, but it points the same way as §9.8's rule:
      if a value cannot be supplied, a wrong one is the worst of the three options.
    * **A 1-tile margin error is free; 3 tiles is marginal** (−2.2 pp, 1.5 SE), and the fitted
      rect's measured edge error (§9.14) sits inside that range. `safe_area_frac` is the fragile
      one (−2.8 pp for ±0.05), which matters because the map-extent estimate behind it is the
      weakest part of the estimator.

    **What this does NOT say.** An inference-time ablation measures what a policy trained WITH a
    column does without it — not what a policy trained without it would have learned. The −10.4 pp
    is an upper bound on the damage of dropping the group, not the price of the retrain.

    **A SECOND PASS, 2026-09-08, because the first one perturbed the wrong thing.** The table
    above adds symmetric gaussian noise to `hero_margin`. The deployed estimator does not make that
    error. It makes two others, both one-sided, and §9.14's re-measurement is what named them:
    **stale** (the map only learns "gassed" for cells on screen, and the poison advances into the
    places the hero left, so every margin reads LARGER than the truth — at 1 tile per 1.5 s, L
    seconds of not looking is +L/1.5 tiles) and **clamped** (a local estimator sees only as far as
    the camera has recently shown, and reports its horizon beyond that). Same evaluator, same 250
    episodes per tier. Drivers: `scratchpad/zone_bias.py`, `scratchpad/zone_axis.py`.

    | condition | overall | vs baseline |
    |---|---|---|
    | baseline | **0.734** | — |
    | stale +1 tile (1.5 s) | 0.728 | −0.6 pp |
    | stale +2 tiles (3 s) | 0.718 | −1.6 pp |
    | stale +4 tiles (6 s) | 0.712 | −2.2 pp |
    | clamped at 10 tiles | 0.715 | −1.9 pp |
    | clamped at 6 tiles | 0.682 | −5.2 pp |
    | clamp 10 + stale +2 | 0.717 | −1.7 pp |
    | clamp x10/y6 (one frame's reach) | 0.697 | −3.7 pp |
    | clamp x10/y6 + stale +2 | 0.724 | −1.0 pp |
    | clamp x14/y10 + stale +2 | 0.704 | −3.0 pp |

    **Staleness is close to free** — six seconds of it, four tiles of optimism, costs 2.2 pp at a
    1.5 pp standard error. That retires the worry the second pass was run to test, and with it the
    need for the pessimistic-decay accumulation rule that was drafted as insurance.

    **The sensing-horizon rows do not resolve.** Clamping everything at 6 tiles looked like a
    cliff (−5.2 pp, 3.5 SE) and motivated the asymmetric follow-up — the camera is 21 tiles wide
    and 13 tall, so one frame reaches ~10 tiles in x and only ~6 in y. But the follow-up's three
    conditions land at 0.697 / 0.724 / 0.704 and **their ordering is not consistent with any
    causal story**: adding staleness to the tight clamp *improved* it, and the loosest clamp
    scored below the tightest. Three conditions spanning 2.7 pp at a 1.5 pp SE is noise, not a
    curve. The honest statement is that every realistic estimator condition costs somewhere
    between 0 and 4 points, and this measurement cannot separate them.

    **What survives both passes is one number: no zone information at all costs 10.4 pp**, and
    every attempt to feed the policy a plausibly-degraded version of it costs a small fraction of
    that.

    **So the recommendation, for the operator to accept or overrule:** deploy on the current
    checkpoint. Supply `hero_margin` as local directional distances (§9.14), pin `next_shrink_in`
    at 0 — a lie whose cost is measured at ≤1 SE, recorded here, and reversible — and accept
    `safe_area_frac` as the group's weak column, since nothing in deployment can supply a map
    extent and it is the only field a retrain would genuinely rescue. **Spending 9.4 hours plus
    sim-side work to chase a 2–4 pp effect that this evaluator cannot even resolve, before the
    live loop has run once, is the wrong order of operations.** The retrain stays a known lever:
    if the deployed agent dies to gas in ways the telemetry traces to these columns, that is the
    evidence that justifies it, and it will also say which of the three routes to take.

16. **BUILT 2026-09-08 — the training side of the zone reconstruction, so the lever is a run and
    not a design task.** The operator accepted §9.15's recommendation (deploy on the current
    checkpoint) and asked for the sim to be updated anyway, so that whenever a retrain happens it
    trains on the shape the live loop can produce rather than on ground truth. That is this item.
    It changes nothing about the checkpoint being deployed now, which still reads
    `configs/agent_obs_deploy.yaml`.

    **The new field.** `zone.hero_margin_local`, registered in `core/obs_schema.py` and produced
    by `core/observation.py`'s zone block:

    ```python
    "hero_margin_local": torch.clamp(hero_margin, -horizon, +horizon)   # horizon in tiles
    ```

    Same four signed distances, same units, same sign convention — `(x-lo.x, hi.x-x, y-lo.y,
    hi.y-y)`, negative outside — put through a **bounded sensing horizon**. `zone.hero_margin`
    is untouched and still what the full-information specs read.

    **Why a clamp is the right model of the estimator, and not a safety rail.** The sim subtracts
    the hero from a rect it owns. Deployment has no rect: `perception/grid.py`'s `GasMap` holds
    cells it has *seen* gassed and answers "how far to gas along −x, +x, −y, +y". That scan
    terminates. Past the horizon it returns nothing and the answer saturates, so a policy trained
    on an unbounded margin meets a column that can never again exceed the horizon — §9.8's silent
    lie, in the one group §9.15 measured at 10.4 pp. The clamp is symmetric because the blindness
    is: standing deep in the gas, the nearest clear ground is as unsensed as distant gas is from
    safety.

    **The horizon is one number, not two copies.** `configs/default.yaml` gains
    `zone.margin_horizon_tiles: 10.0` → `SimConfig.zone_margin_horizon_tiles`, and `config.py`'s
    `validate()` rejects a non-positive one (a zero horizon collapses the field to four constant
    zeros, which is the failure this whole family of files exists to prevent). **The deployed
    estimator must READ that value out of the run's `env_config`, never carry its own copy** —
    two hardcoded 10.0s in two packages is exactly how this drifts. 10 tiles is where §9.15's
    clamp rows put the knee (−1.9 pp at 10, −5.2 pp at 6), and
    `test_the_margin_horizon_actually_clamps_something_on_the_real_map` fails if someone widens it
    until it stops clamping — a no-op horizon silently restores the unbounded field while the
    estimator keeps saturating, and the spec still names the right column.

    **The new spec: `configs/agent_obs_deploy2.yaml`.** A fourth file, per the standing rule —
    `runs/mortis_deploy-20260907-041522/train.yaml` names `agent_obs_deploy.yaml` by path, and
    that is the checkpoint being deployed this week. Identical to its parent in every group but
    `zone`, which goes **7 columns → 5**:

    | field | deploy | deploy2 | why |
    |---|---|---|---|
    | `zone.hero_margin` | ✓ | → `hero_margin_local` | recoverable, but only out to a horizon |
    | `zone.active` | ✓ | ✓ | suppliable and truthful |
    | `zone.safe_area_frac` | ✓ | dropped | needs a map extent an odometry-anchored canvas never learns; §9.15 prices ±0.05 at −2.8 pp; the grid's `in_zone` channel already carries the safe area's *local* shape, which is what a fleeing decision uses |
    | `zone.next_shrink_in` | ✓ | dropped | structurally unobservable (§9.14); zeroing cost 1.5 pp, feeding a plausible wrong value cost 4.4 pp |

    `tests/test_configs_files.py` pins the pair in both directions: every non-`zone` group must be
    byte-identical to the parent's, and the two dropped fields must stay dropped. The five
    deploy-family property tests (type masking, the supply list, the hp swap, the normalization
    divisor, the cube-pickup pairing) are now parametrized over both files, so a new spec in this
    lineage inherits the guards instead of quietly escaping them.

    **What was measured and deliberately NOT modelled**, because each would inject training noise
    to buy back a fraction of a point:

    * *Staleness.* A margin up to 6 s old cost 2.2 pp (§9.15). No delay line.
    * *Axis asymmetry.* The camera reaches ~10 tiles in x and ~6 in y, but the three-condition
      ablation came back non-monotone (0.697 / 0.724 / 0.704) — noise, not a curve. One scalar
      horizon, not two.
    * *`GasMap`'s blind spot.* Cells never on screen stay `clear` forever, so the estimator's
      error is one-sided **optimistic** in exactly the places the hero walked away from. Modelling
      it needs a per-env visibility memory in the sim, which is a large change for a low-single-
      digit effect. The clamp bounds that error at the horizon; it does not remove it. **This is
      the residual, and it is a deployment-side problem** — the fix is a better accumulation rule
      in `GasMap`, not a different observation.

    **Still to build:** the estimator itself (a `GasMap` scan along ±x/±y, clamped at the config's
    horizon), which lands with `assemble.py`. Nothing here has been trained yet; `deploy2` has no
    run behind it.

17. ~~**`hero.pos_norm` has no honest supplier either — and it is in the `self` group.**~~
    **MEASURED AND CLOSED 2026-09-08 — a wrong origin is free, so anchor at the map centre.**

    Found while writing `perception/assemble.py`, not while auditing §6.3. `hero.pos_norm` is
    `hero.pos / (map_w, map_h)` and the deployed loop has **neither half**: odometry's
    `position_tiles` is measured from wherever tracking started (an arbitrary per-match origin),
    and nothing on screen gives the map's extent. It is the same missing quantity that removed
    `zone.safe_area_frac` in §9.14, sitting in the one group §6.3 signed off as *fully*
    suppliable. Everything else in `self` is proprioception or a velocity; this one column is the
    exception, which is why it was missed.

    Priced at inference exactly as §9.15 priced the zone group — same checkpoint
    (`runs/mortis_deploy-20260907-041522/best_model.zip`), same `TierEvaluator` reseeding the same
    scenarios, 250 episodes/tier, only the observation perturbed. Overall SE ≈ 1.4 pp, per-tier
    ≈ 2.8 pp:

    | `hero.pos_norm` | easy | medium | hard | elite | **overall** |
    |---|---|---|---|---|---|
    | true (baseline) | 0.872 | 0.748 | 0.728 | 0.588 | **0.734** |
    | **+ 0.15 (constant offset)** | 0.868 | 0.748 | 0.720 | 0.600 | **0.734** |
    | **+ 0.35 (constant offset)** | 0.888 | 0.728 | 0.776 | 0.580 | **0.743** |
    | ~ U(0, 1), resampled per step | 0.892 | 0.692 | 0.732 | 0.564 | **0.720** |
    | 0 | 0.868 | 0.696 | 0.716 | **0.468** | **0.687** |

    The baseline reproduces §9.15's 0.734 exactly, which is the check that the harness is the same
    one.

    **Read it as an asymmetry, not a ranking.** A constant offset — right scale, wrong origin,
    which is *precisely* odometry anchored at spawn — is free at **two different magnitudes**,
    including +0.35, which pushes the column past 1.0 and outside its entire trained range.
    Zeroing it costs 4.7 pp overall (3.3 SE) and **12 pp on elite** (4.3 tier-SE). So the policy
    is not reading this column for "where am I on the map" — it would then care about the offset.
    It is reading it for something a constant shift preserves and zero destroys, and zero
    additionally *asserts*, every tick, the specific claim "pinned in the top-left corner." That
    is the never-feed-a-constant rule turning up as a number, in the one case where dropping the
    field was the tempting alternative.

    **Decision: do not add a fifth spec.** `deploy2` keeps `hero.pos_norm`, and
    `assemble.MapFrame` supplies it as:

    * `origin = (map_w / 2, map_h / 2)` — odometry's arbitrary start lands at the map centre, the
      point with the most room before the column saturates in any direction.
    * `extent = (map_w, map_h)` — **the sim's map size, not a guess at the real one.** Odometry is
      in tiles and `hero.pos` is in tiles, so dividing by the sim's extent makes one tile of real
      movement produce exactly the column movement it produced in training. Guessing Nulls Brawl's
      actual map size would introduce a **scale** error, which is a different perturbation from
      the one measured above and was **not** tested. This choice quarantines the whole uncertainty
      into the origin — the half that measured free.

    `MapFrame` is a four-line frozen dataclass with an `explain()` that logs "ASSUMED, not
    measured", and `tests/test_deployment_assemble.py` asserts that moving it changes
    `self[0:2]` and **nothing else** — if absolute position ever leaks into a relative field, that
    test is where it surfaces.

    **Residual, unpriced:** the offset is constant *within* a match (odometry does not re-anchor)
    but differs *between* matches. Both tested magnitudes were free, which is two points of
    evidence for insensitivity, not a curve. If a live run shows position-dependent weirdness,
    this is the first column to suspect and a third magnitude is a 20-minute run.

---

## 10. What you can do to make this easier

Concrete asks, roughly in order of payoff.

**Setup**

1. ~~Set the display to 1920×1080 for agent sessions.~~ **Settled 2026-09-08 — the monitor
   stays at 2560×1440**, operator's call ("a major inconvenience otherwise"). **Nothing to
   change, and specifically do not add a 1920×1080 intermediate resize.** This ask was wrong to
   make: `DeployCapture` normalizes whatever `mss` grabs and resizes to the calibrated
   2002×1126 viewport, so a 1440p grab lands at **0.78×, a downscale** — the good direction, and
   the one `MAX_UPSCALE` exists to protect. A 1080p desktop would actually be the *tighter* case
   (1.04× upscale, inside the 1.10 slack but only just). The "anchors shift by 1.333×" worry was
   already answered by the two-space split: tap targets are DEVICE pixels over ADB and never
   touch the monitor, and the match gate reads `Calibration.viewport_button()`, which is fixed to
   the viewport. Inserting 2558 → 1920 → 2002 would be a second resample for no gain.
   `capture.monitor: 1` in `configs/vision.yaml` is already the right index.
2. ~~Enable ADB.~~ ~~Still need the **port** it reports.~~ **Fully closed 2026-09-08 — nothing to
   report, the emulator publishes it.** BlueStacks writes
   `bst.instance.<name>.status.adb_port="5555"` into `C:\ProgramData\BlueStacks_nxt\bluestacks.conf`,
   so `control/adb.find_adb_serial()` reads it instead of asking. **Verified live on this machine:
   instance `Pie64` at `127.0.0.1:5555`**, and `find_adb()` resolves the bundled
   `C:\Program Files\BlueStacks_nxt\HD-Adb.exe` since no adb is on PATH. Note there are two keys
   per instance and only `status.adb_port` is authoritative — `adb_port` is the configured value,
   not necessarily the one being served.
3. ~~Turn off every BlueStacks overlay.~~ **Done — confirmed off by the operator 2026-09-08**
   (sidebar, FPS counter, notification toasts, the CPU/GPU stats readout that was visible in the
   first recording). The reason it mattered stands: anything drawn inside the captured rectangle
   becomes perception input. Re-check only if a new overlay is switched on.
4. ~~UI scale.~~ **Settled** — the HUD is fixed for this project and §5's anchors are measured
   against it.

**Data**

5. ~~Re-record at deployment fidelity.~~ **Done** — `bluestacks-example-new.mp4`, 1920×1080 @
   30 fps, fullscreen, Mortis, containing a full pre-match → gameplay → post-match arc. It
   validated §5 outright. Two follow-ups when convenient, neither blocking:
6. **A few seconds of deliberate joystick work** — hold a direction, swing a full circle, stop —
   gives §4.3's saturation radius something to be measured against. The current clip has ordinary
   play, which does not isolate it.
7. **One Mortis dash into a wall**, if it happens naturally. Downgraded from "the clearest
   desync case" now that §6.3 is built: the terrain clip moves `dash_speed` and the landing point,
   and **every field the shadow owns is seeded from `dash_duration` or the kit constants**, so a
   blocked dash leaves all of them exactly right. Still useful, for the travel budget's tolerance
   rather than for the timers.
8. ~~A match that actually reaches the poison, at deployment fidelity.~~ **Done —
   `bluestacks-example-zone.mp4`**, 66 s at 1920×1080 with the gas spawning at t≈21 s and closing
   in until the hero dies. It proved the detector transfers to BlueStacks unchanged and it
   disproved the fitted-rect supplier, both in §9.14. One narrower follow-up if the countdown ever
   needs to be measured rather than pinned:
9. **Thirty seconds parked at the gas edge, not walking away from it.** The schedule cannot be
   timed from a camera that follows a moving hero — reveal and advance look identical (§9.14). If
   the hero stands still with the front on screen and lets it come, the front's steps are
   unambiguous and `next_shrink_in`'s period falls out of one clip. Only worth recording if the
   pinned-zero decision in §9.15 is one you would rather not take.

**Things you can do that I cannot**

7. ~~Finish the `onnxruntime-gpu` install.~~ **Done, by me, on 2026-09-07.** It was not an
   install you could do in one command — the wheel was one of four faults and the other three
   were code. Both detectors now report `CUDAExecutionProvider` and run 3.6–4.4× faster (§9.12
   for the full diagnosis, §7.1 for the numbers). Two things worth keeping: `pip uninstall -y
   onnxruntime` *after* installing `onnxruntime-gpu` **deletes the shared package directory**,
   leaving an importable namespace with no runtime in it (repair with `pip install
   --force-reinstall --no-deps onnxruntime-gpu==1.29.0`, which does not re-download the CUDA
   wheels); and torch and ONNX Runtime have to **split cuDNN between them**, which is the one
   place this setup is still fragile.

8. ~~**The §4.1 spike itself, if you would rather drive it.**~~ **Moot — the spike was run and
   closed** (§4.1): the emulator's touch device takes 16 concurrent contacts and persistent
   contacts hold across `sendevent` writes, which is the finding the whole control design rests
   on.
9. ~~**Confirming the joystick's saturation radius by hand.**~~ **Moot — measured in §4.3.**
   Commanding 140 px deflected the knob to 91.7 / 92.6 / 92.0 across three directions, so the game
   clamps at ~92 px and `radius_px` is set to 110. The same run confirmed there is no y flip
   (bin 2 → (+64.4, **+65.6**), i.e. 45° *down*-right).

**Not needed**

10. Do **not** set up a second BlueStacks instance, GPU passthrough tuning, or anything for
   parallelism. Single agent is the stated requirement and the budget in §7 is comfortable
   because of it.

---

## 11. What I would revisit as this grows

- **Match queueing** is deliberately out of scope, but §5's machinery is most of what it needs —
  the same ring score that says "match over" also says "we are in a menu", which is where
  navigation would start.
- **The two-rate loop** is currently 20/4 because that mirrors training. If perception ever costs
  more than the 250 ms budget, the honest fix is lowering the *decision* rate and retraining at a
  matching `action_repeat`, not skipping perception ticks — odometry's per-frame shift bound is
  what breaks first.
- **`project.to_tiles`'s anchor offset** is documented as a known calibration displacement
  (~+2 tiles in y). It is fine for training-data annotation. Whether it is fine for a live control
  loop, where a systematic 2-tile position bias feeds straight into the policy, is worth measuring
  before trusting it.
