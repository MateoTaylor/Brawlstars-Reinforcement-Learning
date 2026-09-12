"""The hero's own state, dead-reckoned from the actions we issue. BRAWL_DEPLOYMENT_DESIGN.md 6.3.

Every field here is one the deployed policy was trained to read and no camera supplies well:
seconds left on a cooldown, whether a dash is still running, how long it has been since we last
attacked. Section 6.3's argument for why they are recoverable at all fits in one line -- **the
deployed agent is not a spectator, it is the thing issuing the actions** -- so anything that is a
function of (our own action stream + `configs/brawlers.yaml` + elapsed time) is available exactly,
with no CV at all.

#### What this owns, and what it deliberately does not

Owned outright, because they are pure functions of our own actions and the clock:

    ammo  ammo_frac  ammo_whole  attack_cd  can_attack  dashing  dash_t  dash_dir
    invuln  long_dash_ready  long_dash_frac  attack_idle_t  facing  facing_vec

Held here but SOURCED FROM CV, because the shadow can only ever see them fall:

    super_ready  super_charge_frac      -- see "The super is the exception" below

Not here at all -- `assemble.py`'s job, from the tracker and the readers:

    pos_norm  vel  hp  in_bush  in_zone  meta.time_frac  meta.n_enemies_alive

`hud.BrawlersLeft.enemies_alive(hero_alive=...)` takes the operand this class's `alive` supplies,
which is the only reason `alive` lives here rather than in the assembler.

#### The tick is `env._run_tick`'s, in its order

`_tick` below runs the same five phases the sim runs, in the same sequence, for the subset of
state this class holds:

    phase 2   tick_timers      ammo regen (gated on attack_cd, gate read BEFORE the decrement),
                               attack_cd / invuln_t decrement, attack_idle_t counts UP
    phase 3   decode_action    the fire bit, ANDed with action_mask
    phase 6   _attack_phase    start_dash (or the super), then attack_idle_t := 0
    phase 7   apply_movement   facing tracks move_dir whenever it is nonzero
    phase 8   advance_dash     dash_t decrement and the completion cleanup

Two orderings inside that are load-bearing and easy to get backwards. `tick_timers` reads the
`attack_cd <= 0` gate BEFORE decrementing it, so a 0.35 s cooldown blocks reload for exactly seven
ticks. And `attack_idle_t` is zeroed AFTER `start_dash` has read it, because `start_dash` is what
decides whether this dash is the long one.

`dash_t` is decremented in phase 8 and nowhere else -- `core/hero.py`'s module docstring opens
with why doing it in both places halves every dash.

**Sub-ticks 2..K of a decision hold the move bin and drop the fire bit**, exactly as `env._held`
does, so one decision is at most one attack attempt. Here that falls out of `act()` queueing a
single pending attack that the next `_tick` consumes.

#### Two clocks, and which one wins

The sim advances `action_repeat` ticks of exactly `cfg.dt` per decision and calls that 0.25 s. A
deployment loop does not get that: capture, detection and inference jitter, so the wall clock and
the tick count come apart. **`advance(seconds)` banks real elapsed time and runs whole `dt`
sub-ticks out of it, carrying the remainder.** The game's cooldowns run on the wall clock, so the
wall clock decides HOW MANY sub-ticks; the sim's arithmetic decides what each one does. A loop
running at 15 Hz simply spends more sub-ticks per decision, and `can_attack` still goes true at
the wall-clock moment the game would accept a tap.

The alternative -- a fixed five sub-ticks per decision -- keeps sim parity by construction and
loses the game: a slow loop would leave the shadow's cooldown running after the real one expired,
which presents as an agent that quietly under-fires and never trips a canary, because the ammo it
predicts is the ammo it would have had.

#### float32, because float64 is off by a tick

The sim's timers are float32 tensors and `cfg.dt` is a Python float promoted into them, so every
countdown is float32 arithmetic. That is not cosmetic. Counting `x = max(0, x - 0.05)` down from
Mortis's `attack_cooldown: 0.35`:

    float32   7 ticks   (0.35 s -- what the sim does)
    float64   8 ticks   (a residue of 4.2e-17 keeps it > 0 for one more tick)

A shadow in float64 would hold `can_attack` false, and block ammo regen, for one sub-tick longer
than the sim on every single attack. So every timer field here is a `numpy.float32` scalar and
every constant is cast before use. `dash_duration: 0.30` is the same story from the other side: it
takes seven ticks to expire in BOTH precisions, which is the 1.19e-8 residue
`core/hero.advance_dash` documents at length and pays for with `step_seconds = min(dash_t, dt)`.

What one dash actually looks like, sub-tick by sub-tick, all three flags confirmed equal to the
sim's by `tests/test_deployment_shadow.py`:

    sub-tick     1  2  3  4  5  6  7  8
    dashing      *  *  *  *  *  *  .  .     6 ticks -- exactly dash_duration
    invuln       *  *  *  *  *  *  *  .     7 ticks -- one longer
    reload held  *  *  *  *  *  *  *  .     7 ticks -- exactly attack_cooldown

The i-frames outlast the dash because `dash_t` is decremented in phase 8 of the very tick that set
it while `invuln_t` waits for phase 2 of the next one. Both are seeded from the same 0.30. Nothing
here is a rounding artifact to tidy up: it is the sim's behaviour, so it is the behaviour the
policy trained against.

#### The action mask this exposes is one sub-tick stale, and always in the safe direction

`hero.action_mask` is evaluated in phase 3, AFTER phase 2's timers. `attack_mask()` here is
evaluated at decision time, which is before them. Every term moves one way only -- `attack_cd`
and `dash_t` only fall, `ammo` only rises, `alive` only ever goes false -- so the pre-timer mask is
a strict subset of the post-timer one. The cost is up to one sub-tick (50 ms) of delay on a shot
issued at the exact tick a cooldown expires. The benefit is that the mask can never offer an
action the game will refuse, which is the direction that matters: a refused tap is a silent lost
shot and a desync the canary then has to find.

#### The super is the exception: dead-reckonable DOWN, not up

`core/hero.add_super_charge` counts hits landed on living PLAYERS -- boxes are excluded by
`env._bookkeeping` specifically so a hero cannot farm his super off crates. Whether our dash
connected is not in our action stream, so charge cannot be dead-reckoned upward at all. It is CV,
from `hero_bars.read_super`, and this class does three things with it that CV alone cannot:

  * **Gates it.** The observation's `super_ready` is the sim's charge test alone; the ACTION MASK
    additionally requires `alive & attack_cd <= 0 & dash_t <= 0`, which only the shadow knows.
  * **Spends it.** Firing zeroes the charge here the instant we tap, rather than waiting for a
    frame to show it.
  * **Latches the spend.** After we fire, a CV read that still says READY is refused until one
    read says otherwise. Without that latch a single stale frame re-arms the mask and the policy
    taps a super it already spent -- and because a super costs charge rather than ammo, the ammo
    canary cannot see that happen.

Stale reads fail closed: past `SUPER_STALE_SECONDS` with no successful read, `super_ready` reports
False and the fraction holds its last value. Holding a fraction is stale, not fabricated; claiming
readiness is neither.

#### A dash into a wall costs the owned fields nothing

Section 10 item 7 asks for footage of one, on the assumption that a blocked dash is the clearest
desync case. Working through what `start_dash` actually clips: the terrain march shortens
`clipped_distance`, and therefore `dash_speed`, and therefore where the hero lands. It does not
touch `dash_duration`, and every field this class owns is seeded from the duration or from the kit
constants -- `dash_t`, `invuln_t`, `attack_cd`, `ammo`, `attack_idle_t`. So a dash into a wall
leaves the shadow exactly right about all of them and wrong only about position, which is CV's
anyway. The footage is still worth having, for the travel budget below; it is not the emergency it
looked like.

#### The canary, and the numbers under its thresholds

`check_ammo` compares `hero_bars.read_ammo`'s fractional reading against the predicted one. The
gap between noise and signal is wide and both ends are measured (section 6.4):

    reader noise      +/-0.06 on one pip, 98.3% of reads within 0.15 of an integer
    smallest desync    1.0    -- a dash that did or did not happen
    AMMO_TOLERANCE     0.5    -- the read is ABOVE the shadow: the game holds ammo we
                              already spent, which no amount of missing paint can fake
    ..._UNPAINTED      1.0    -- the read is BELOW the shadow: a recharging pip the
                              detector missed, worth a whole pip. See the constants.

Two guards keep a true reading from being called a desync. **A grace window** after each attack,
because the spend takes about a second to reach the screen (see `DESYNC_GRACE_SECONDS`), so a real
one-pip gap is expected for several decisions. **A debounce**, because `read_ammo` returns None on
12% of frames and a single bad read must not trigger a resync -- the resync is destructive,
reseeding timers conservatively. `DESYNC_STRIKES` consecutive failures outside the grace window is
0.75 s at 4 Hz.

A miss (`None`) neither strikes nor clears; it is not evidence either way. The debounce is also
what absorbs the OTHER failure the first live run showed: `read_ammo` returns a spurious 0.00 or
1.00 off a full clip roughly once every 60 reads, always for one or two samples, never three.

Worth knowing where the canary is blind: the grace window mutes it during a burst, and Mortis can
fire every 0.35 s. It is fully live through the tail of the 2.50 s reload that follows and through
every idle stretch -- about 80% of a real match, measured on run1 -- which is where "did we spend
two or three?" is answerable anyway.

**Detection and response are separate on purpose.** `check_ammo` never resyncs by itself;
`loop.py` decides, because it is also the thing that has to count sustained desyncs into section
8's fail-closed path. A resync that fired silently inside a check would hide the very failure the
check exists to report.

#### The travel budget: a ceiling, not a threshold

Section 6.3 also names position as a desync signal. A symmetric "CV position diverged from the
predicted one" check is not implementable here and would be useless if it were: the shadow has no
terrain, so every wall the hero walks into reads as divergence. What IS sound is one-sided.
`travel_budget` accumulates the FURTHEST the hero could have gone since `reset_travel()` --
`move_speed * dt` per walking tick, the UNCLIPPED `dash_speed * step` per dashing one. Terrain only
ever shortens the real distance, so a CV displacement that exceeds the budget by more than
`TRAVEL_TOLERANCE_TILES` is the tracker or the odometry being wrong, never the hero being fast.
The comparison itself belongs to whoever holds the CV position, which is `assemble.py`.

#### One correction to section 6.3's table

It lists `hero.invuln` as "a `-= dt` countdown, seeded at spawn". It is not seeded at spawn --
`core/spawn.py` sets pos, hp, ammo, facing and alive, and nothing else. `ent_invuln_t` is written
in exactly one place, `start_dash`, and `obs_schema` says so in its own description of the field:
"invuln_t > 0 (dash i-frames)". The countdown is right; the seed is the dash.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

# One definition of the action encoding for the whole package, and it lives with the thing that
# presses the buttons. `hero.decode_action`'s column-1 values: 0 = nothing, 1 = attack, 2 = super.
from ..control.buttons import ATTACK_FIRE, ATTACK_NONE, ATTACK_SUPER

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

# `sim.dt` and `action.n_move_bins` from configs/default.yaml. Not read from there at runtime: the
# policy was trained under one value of each and a mismatch is a retrain, not a config change --
# `joystick.py` says the same thing about `n_bins`, for the same reason.
DT = 0.05
N_MOVE_BINS = 16

# See "The canary" in the module docstring for where each of these comes from.
# **The comparison is ASYMMETRIC, because the sensor's error is.** `read_ammo` sums PAINTED pip
# fill, and unpainted pixels can only lose ammo -- the read can under-report a partially recharged
# pip, it cannot invent one. So the two directions carry different evidence:
#
#   CV ABOVE the shadow  -- the game holds ammo we did not model. Unexplainable by the read, so
#                           this side keeps the tight tolerance. It is also the direction that
#                           means OUR TAPS ARE NOT LANDING, which is the whole point of the canary.
#   CV BELOW the shadow  -- fully explained by a recharging pip the read missed, for anything up
#                           to one whole pip.
#
# MEASURED on the second live deployment (run1.csv, 2026-09-09): the recharging pip is detected
# only INTERMITTENTLY -- CV read 2.00 -> 2.56 -> 2.00 inside 0.5 s at t=16.5, and 2.00 -> 2.65 ->
# 2.00 at t=17.7, while the shadow sat between them the whole time. Outside the grace window the
# error distribution is median 0.12, p90 0.64, max 1.56, so a symmetric 0.5 sits BELOW the
# sensor's own noise floor and manufactures desyncs. All three of that run's resyncs were false
# and all three were on this side.
#
# Replaying that run's 100 reads through the strike machine: symmetric 0.5 trips 3 times (exactly
# the three seen live, at t=17.7/23.5/28.2), asymmetric trips 0 -- while still flagging 6 of 7
# samples of an injected "no input lands" failure, the same as the tight rule. Flooring both sides
# to whole pips was tried first and is WORSE (6 trips): it turns a 2.99-vs-3.00 boundary into a
# full pip of disagreement.
AMMO_TOLERANCE = 0.5
AMMO_TOLERANCE_UNPAINTED = 1.0
DESYNC_STRIKES = 3
# Seconds after a modelled attack during which an ammo disagreement is NOT counted, because the
# game has not shown the spend yet. MEASURED, not reasoned: the first live deployment (run1.csv,
# 2026-09-09) put the delay from a modelled tap to the pip bar reading one lower at 0.50-1.25 s,
# median 1.00 -- and `verify_tap` independently needs a 0.9 s `TROUGH_WINDOW_S` to catch the same
# drop, which is the same number arrived at from the other side.
#
# **0.30 was sized for one decision period and it was wrong by 4x.** All four resyncs of that run
# followed an attack cluster by 1-2 s: the shadow decrements on the tap, the bar had not moved
# yet, and the canary called the difference a desync. The second resync was worse than the first,
# because resyncing ADOPTS the CV value and the value it adopted was stale -- so a false trip does
# not merely waste budget, it injects the error it claimed to find.
#
# It does not matter which half of the delay is actuation (ADB -> emulator -> game) and which is
# the swing animation plus the bar drain; the canary compares against what is ON SCREEN, so the
# window has to cover the whole observable path either way. 1.5 s is the worst clean observation
# plus one decision period.
DESYNC_GRACE_SECONDS = 1.5
SUPER_STALE_SECONDS = 1.0
# Sized against `tracker.GATE_NOISE_TILES = 1.0`, the anchor noise on a hero box, because that is
# what the CV displacement this gets compared against carries.
TRAVEL_TOLERANCE_TILES = 1.0

OK, NO_READ, GRACE, SUSPECT, DESYNC = "ok", "no-read", "grace", "suspect", "desync"

_F32 = np.float32
_ZERO = _F32(0.0)
_TWO_PI = 2.0 * math.pi
# Slack on the sub-tick boundary in `advance`. A nanosecond is far below anything a perception
# loop can measure and far above binary 0.05's error against a twentieth of a second.
_TICK_EPS = 1e-9


@dataclass(frozen=True)
class ShadowParams:
    """The `configs/brawlers.yaml` fields the dead reckoning needs, and no others.

    Deliberately not `brawl_sim`'s `SimParams`: that is a per-kind tensor table built for a batch
    of environments on a device, and there is one hero on one screen here. The values are the same
    file, loaded once.
    """
    max_ammo: float
    reload_seconds: float
    attack_cooldown: float
    dash_distance: float
    dash_duration: float
    long_dash_seconds: float
    long_dash_multiplier: float
    super_charge_hits: int
    move_speed: float

    @classmethod
    def load(cls, kind: str = "hero_mortis", path: Path | str | None = None) -> "ShadowParams":
        """Read one brawler's block. `kind` is a top-level key in `configs/brawlers.yaml`.

        A field given as a RANGE (a two-element list, which `config._resolve_value` would sample
        per environment) raises rather than picking an end. Dead reckoning against a randomized
        constant is not dead reckoning, and the deployment has to commit to one number.
        """
        spec = yaml.safe_load(Path(path or _CONFIGS_DIR / "brawlers.yaml").read_text())
        if kind not in spec:
            raise KeyError(f"{kind!r} is not a brawler in brawlers.yaml")
        block = spec[kind]
        values = {}
        for f in fields(cls):
            if f.name not in block:
                raise KeyError(f"{kind!r} has no {f.name!r} -- it cannot be dead-reckoned")
            raw = block[f.name]
            if isinstance(raw, (list, tuple)):
                raise ValueError(
                    f"{kind}.{f.name} is a randomized range {raw}; deployment needs one value")
            values[f.name] = raw
        return cls(**values)


@dataclass(frozen=True)
class Desync:
    """One ammo comparison. `error` is CV minus shadow, so a POSITIVE error means the game has
    more ammo than we predicted -- an attack we thought landed did not."""
    status: str
    error: float | None
    strikes: int

    @property
    def tripped(self) -> bool:
        return self.status == DESYNC


class ShadowHero:
    """One instance per match. `reset()` at the in-match gate, `act()` per decision, `advance()`
    per perception tick, `observe()` when the assembler needs the `self` group."""

    def __init__(self, params: ShadowParams, *, dt: float = DT, n_move_bins: int = N_MOVE_BINS,
                 ammo_tolerance: float = AMMO_TOLERANCE,
                 ammo_tolerance_unpainted: float = AMMO_TOLERANCE_UNPAINTED,
                 desync_strikes: int = DESYNC_STRIKES,
                 desync_grace_seconds: float = DESYNC_GRACE_SECONDS,
                 super_stale_seconds: float = SUPER_STALE_SECONDS):
        self.p = params
        self.dt = _F32(dt)
        self._dt_py = float(dt)
        self.n_move_bins = n_move_bins
        self.ammo_tolerance = float(ammo_tolerance)
        self.ammo_tolerance_unpainted = float(ammo_tolerance_unpainted)
        self.desync_strikes = int(desync_strikes)
        self.desync_grace_seconds = float(desync_grace_seconds)
        self.super_stale_seconds = float(super_stale_seconds)

        self._max_ammo = _F32(params.max_ammo)
        # `cfg.dt / reload_seconds` from tick_timers, at the precision AND IN THE ORDER the sim
        # computes it. A Python scalar over a tensor is `Tensor.__rtruediv__`, which torch
        # implements as `reciprocal(tensor) * scalar` -- two float32 roundings, not one. The direct
        # quotient agrees at 2.25 and is one ULP low at 2.50 (and at Brock's 1.75), which the
        # parity test caught the moment reload_seconds was refit. Measured, CPU and CUDA alike.
        self._ammo_gain = (_F32(1.0) / _F32(params.reload_seconds)) * _F32(dt)
        self._attack_cooldown = _F32(params.attack_cooldown)
        self._dash_duration = _F32(params.dash_duration)
        self._bin_step = _TWO_PI / n_move_bins

        self.reset()

    # -- lifecycle -----------------------------------------------------------

    def reset(self, *, facing: float = 0.0) -> None:
        """Spawn state, matching `core/spawn.py`: a full clip, no cooldown, no charge, and an
        `attack_idle_t` of zero -- so the long dash is genuinely unavailable for the first
        `long_dash_seconds` of a match, exactly as in training.

        `facing` has no CV source at spawn. The sim seeds it toward the map centre; here it
        defaults to 0 rad (screen +x) and is corrected by the first nonzero move bin, which in
        practice is the first decision. What it costs until then is one idle dash going the wrong
        way, since `action.dash_on_idle: facing`. Pass the real bearing if the assembler has a
        pose by then.
        """
        self.alive = True
        self.ammo = _F32(self._max_ammo)
        self.attack_cd = _ZERO
        self.invuln_t = _ZERO
        self.attack_idle_t = _ZERO
        self.dash_t = _ZERO
        self.dash_dir = (_ZERO, _ZERO)
        self.dash_speed = _ZERO
        self.facing = _F32(facing)

        self.super_charge_frac = 0.0
        self._super_ready_cv = False
        self._super_read_at: float | None = None
        self._super_spend_pending = False

        self.elapsed = 0.0            # tick-quantised seconds since reset
        self._bank = 0.0              # wall-clock remainder not yet worth a sub-tick
        self._move = 0
        self._pending_attack = ATTACK_NONE
        self._last_attack_at = -math.inf

        self.travel = 0.0
        self.strikes = 0
        self.desyncs = 0
        self.ticks = 0

    def set_alive(self, alive: bool) -> None:
        """From CV -- the HP reader or the match gate. A dead hero keeps ticking its timers,
        exactly as `tick_timers` does for a dead entity, but every attack gate reads `alive`."""
        self.alive = bool(alive)

    # -- the decision --------------------------------------------------------

    def attack_mask(self) -> tuple[bool, bool, bool]:
        """`hero.action_mask`'s attack column: (no-fire, attack, super), all three legal-now.

        No move mask: `action_mask` builds it as all-ones and nothing has ever narrowed it, so
        there is nothing here to supply. See the class docstring on why this is one sub-tick stale
        and why that is the safe side.
        """
        ready = self.alive and self.attack_cd <= 0 and self.dash_t <= 0
        return (True, bool(ready and self.ammo >= 1.0), bool(ready and self.super_ready))

    def act(self, move: int, attack: int = ATTACK_NONE) -> int:
        """Record the decision we are about to send to the device. Returns the attack that will
        actually be modelled -- `ATTACK_NONE` if the mask refuses it -- which is what the caller
        should tap, so a caller that forgot to mask cannot desync the shadow.

        Nothing is applied here. The move bin is held from now until the next `act`, and the
        attack is consumed by the next sub-tick, which is `env._held`'s rule reached from the
        other direction: there is only ever one attack queued, so sub-ticks 2..K of a decision
        cannot repeat it.

        A second `act` before any sub-tick has run refuses the attack. The device would swallow
        that tap too -- `Buttons.tap` releases a still-held contact before re-pressing -- and
        modelling a shot the game did not take is the one error this class must never make.
        """
        self._move = int(move)
        if attack == ATTACK_NONE or self._pending_attack != ATTACK_NONE:
            return ATTACK_NONE
        legal = self.attack_mask()
        if attack == ATTACK_FIRE and legal[1]:
            self._pending_attack = ATTACK_FIRE
        elif attack == ATTACK_SUPER and legal[2]:
            self._pending_attack = ATTACK_SUPER
        else:
            return ATTACK_NONE
        return self._pending_attack

    def advance(self, seconds: float) -> int:
        """Run whole `dt` sub-ticks out of `seconds` of real time. Returns how many ran.

        The remainder is banked, not dropped, so a loop that never lands exactly on `dt` does not
        lose a tick every iteration. Long gaps are NOT capped: real time really did pass, so the
        cooldowns really did expire and the clip really did reload. The only thing a long gap gets
        wrong is that it assumes we kept holding the last move bin, which touches `facing` and the
        travel budget and nothing else.
        """
        if seconds <= 0:
            return 0
        self._bank += float(seconds)
        # Subtract rather than divide, with a nanosecond of slack. `0.25 // 0.05` is **4**, not 5:
        # binary 0.05 is a hair above a twentieth, so an exact floor silently drops one sub-tick
        # out of every nominal decision -- which the sim-parity test caught on its first decision.
        n = 0
        while self._bank + _TICK_EPS >= self._dt_py:
            self._bank -= self._dt_py
            n += 1
        if n <= 0:
            return 0
        for _ in range(n):
            self._tick()
        self.ticks += n
        return n

    # -- one sim tick --------------------------------------------------------

    def _tick(self) -> None:
        move_dir = self._dir_from_bin(self._move)

        # ---- phase 2: tick_timers. The reload gate is read BEFORE the decrement.
        if self.attack_cd <= 0:
            self.ammo = min(self.ammo + self._ammo_gain, self._max_ammo)
        self.attack_cd = max(_ZERO, self.attack_cd - self.dt)
        self.invuln_t = max(_ZERO, self.invuln_t - self.dt)
        self.attack_idle_t = self.attack_idle_t + self.dt

        # ---- phases 3 and 6: decode, then attack. The mask is re-applied here for the same
        # reason `env._attack_phase` re-applies it after `decode_action` already did: it makes an
        # illegal attack impossible from any source rather than merely unlikely.
        attack, self._pending_attack = self._pending_attack, ATTACK_NONE
        legal = self.attack_mask()
        attacked = False
        if attack == ATTACK_FIRE and legal[1]:
            self._start_dash(move_dir)
            attacked = True
        elif attack == ATTACK_SUPER and legal[2]:
            self._fire_super()
            attacked = True
        if attacked:
            # AFTER start_dash, which reads this to decide whether the dash was the long one.
            self.attack_idle_t = _ZERO
            self._last_attack_at = self.elapsed

        # ---- phase 7: apply_movement. Facing tracks a nonzero move bin whether or not we are
        # dashing -- the sim's only exception is a kind mid-melee-sweep, which Mortis is not. The
        # walk itself is skipped while dashing, which is what makes a dash REPLACE the step.
        if move_dir is not None:
            self.facing = _F32(math.atan2(float(move_dir[1]), float(move_dir[0])))
            if self.alive and self.dash_t <= 0:
                self.travel += self.p.move_speed * self._dt_py

        # ---- phase 8: advance_dash. `step_seconds` is the last-tick clamp; without it a dash
        # travels one whole tick further than its duration bought.
        if self.dash_t > 0:
            self.travel += float(self.dash_speed) * min(float(self.dash_t), self._dt_py)
            self.dash_t = max(_ZERO, self.dash_t - self.dt)
            if self.dash_t <= 0:
                self.dash_dir = (_ZERO, _ZERO)
                self.dash_speed = _ZERO

        # Advanced HERE rather than once per `advance` call, so `self.elapsed` is the start time of
        # whichever tick is running. `_last_attack_at` is read against it, and a batch-level update
        # would date every attack in the batch to the batch's own start.
        self.elapsed += self._dt_py

    def _start_dash(self, move_dir) -> None:
        """`core/hero.start_dash`, minus the terrain clip. See the module docstring: the clip moves
        `dash_speed` and the landing point, neither of which is a field this class owns, so the
        omission costs nothing but the travel budget's tightness."""
        distance = self.p.dash_distance
        if self.long_dash_ready:
            distance *= max(1.0, self.p.long_dash_multiplier)
        direction = move_dir if move_dir is not None else self._facing_vec()

        self.ammo = self.ammo - _F32(1.0)
        self.attack_cd = self._attack_cooldown
        self.dash_t = self._dash_duration
        self.dash_dir = direction
        self.dash_speed = _F32(distance / max(self.p.dash_duration, 1e-6))
        self.invuln_t = self._dash_duration
        self.facing = _F32(math.atan2(float(direction[1]), float(direction[0])))

    def _fire_super(self) -> None:
        """`env._attack_phase`'s super branch: spend the whole meter, take the cooldown, no ammo.

        The latch is this class's own -- see "The super is the exception". CV cannot be trusted to
        show the spend before the next decision, and re-offering a spent super is invisible to the
        ammo canary."""
        self.attack_cd = self._attack_cooldown
        self.super_charge_frac = 0.0
        self._super_ready_cv = False
        self._super_spend_pending = True

    def _dir_from_bin(self, move: int):
        """`geo.dir_from_bin(move - 1)`, or None for the idle bin. `joystick.contact_point` maps
        the same value to a screen point with the same formula, deliberately."""
        if move <= 0:
            return None
        theta = _F32(move - 1) * _F32(self._bin_step)
        return (_F32(math.cos(float(theta))), _F32(math.sin(float(theta))))

    def _facing_vec(self):
        return (_F32(math.cos(float(self.facing))), _F32(math.sin(float(self.facing))))

    # -- CV in ---------------------------------------------------------------

    def set_super(self, reading) -> None:
        """Take one `hero_bars.SuperReading`, or None for "could not read".

        A read arriving while a spend is pending is used only if it AGREES that the super is gone;
        that is what clears the latch. Anything else would let one stale frame re-arm the mask.
        """
        if reading is None:
            return
        if self._super_spend_pending:
            if reading.ready:
                return
            self._super_spend_pending = False
        self.super_charge_frac = float(reading.charge)
        self._super_ready_cv = bool(reading.ready)
        self._super_read_at = self.elapsed

    def check_ammo(self, reading) -> Desync:
        """Compare `hero_bars.read_ammo`'s reading against the predicted clip.

        Returns a verdict and never acts on it -- `loop.py` owns the response, because it is also
        what counts sustained desyncs into section 8's fail-closed path.
        """
        if reading is None:
            return Desync(NO_READ, None, self.strikes)
        error = float(reading.ammo) - float(self.ammo)
        if self.elapsed - self._last_attack_at < self.desync_grace_seconds:
            return Desync(GRACE, error, self.strikes)
        # See `AMMO_TOLERANCE`: negative error (the shadow above the read) is what a missed
        # recharging pip looks like, so it is allowed a whole pip; positive error is not
        # explainable by the sensor and keeps the tight bound.
        allowed = self.ammo_tolerance if error >= 0 else self.ammo_tolerance_unpainted
        if abs(error) < allowed:
            self.strikes = 0
            return Desync(OK, error, 0)
        self.strikes += 1
        status = DESYNC if self.strikes >= self.desync_strikes else SUSPECT
        return Desync(status, error, self.strikes)

    def resync(self, reading=None) -> None:
        """Recover from a tripped canary: take what CV can see, and reseed what it cannot to the
        state that offers the policy LESS rather than more.

        Conservative here means "assume not ready", every time. A cooldown that is really over
        costs one decision of waiting; a cooldown reported as over when it is not costs a tap the
        game refuses and another desync. Same for the long dash (report uncharged, so the policy
        does not plan a 5.34-tile dash it will not get), for invulnerability, and for the super.

        `facing` is left alone: it has no CV source and no safe default, and the next nonzero move
        bin overwrites it anyway.
        """
        if reading is not None:
            self.ammo = _F32(min(float(reading.ammo), float(self._max_ammo)))
        self.attack_cd = self._attack_cooldown
        self.dash_t = _ZERO
        self.dash_dir = (_ZERO, _ZERO)
        self.dash_speed = _ZERO
        self.invuln_t = _ZERO
        self.attack_idle_t = _ZERO
        self.super_charge_frac = 0.0
        self._super_ready_cv = False
        self._super_spend_pending = False
        self._pending_attack = ATTACK_NONE
        self.strikes = 0
        self.desyncs += 1

    def reset_travel(self) -> None:
        self.travel = 0.0

    @property
    def travel_budget(self) -> float:
        """Tiles, the FURTHEST the hero could have gone since `reset_travel()`. A one-sided
        ceiling, not a prediction -- terrain only ever shortens the real distance."""
        return self.travel

    # -- the observation -----------------------------------------------------

    @property
    def super_ready(self) -> bool:
        """The sim's `hero.super_ready` -- the charge test ALONE, without the cooldown and
        no-dashing terms `action_mask` adds. Reports False once the last CV read goes stale."""
        if self._super_spend_pending or not self._super_ready_cv:
            return False
        if self._super_read_at is None:
            return False
        return (self.elapsed - self._super_read_at) <= self.super_stale_seconds

    @property
    def long_dash_ready(self) -> bool:
        return (self.p.long_dash_seconds > 0
                and float(self.attack_idle_t) >= self.p.long_dash_seconds)

    @property
    def long_dash_frac(self) -> float:
        if self.p.long_dash_seconds <= 0:
            return 0.0
        return min(1.0, float(self.attack_idle_t) / self.p.long_dash_seconds)

    @property
    def can_attack(self) -> bool:
        return bool(self.alive and self.ammo >= 1.0 and self.attack_cd <= 0 and self.dash_t <= 0)

    def observe(self) -> dict:
        """The `hero.*` fields this class owns, under `core/observation.py`'s own key names and in
        its units -- tiles, seconds, radians, raw rather than normalized. `obs_select` does the
        normalizing, the same as it does for the sim."""
        return {
            "alive": self.alive,
            "facing": float(self.facing),
            "facing_vec": tuple(float(v) for v in self._facing_vec()),
            "ammo": float(self.ammo),
            "ammo_frac": float(self.ammo) / self.p.max_ammo,
            "ammo_whole": int(math.floor(float(self.ammo))),
            "attack_cd": float(self.attack_cd),
            "can_attack": self.can_attack,
            "dashing": bool(self.dash_t > 0),
            "dash_t": float(self.dash_t),
            "dash_dir": tuple(float(v) for v in self.dash_dir),
            "invuln": bool(self.invuln_t > 0),
            "attack_idle_t": float(self.attack_idle_t),
            "long_dash_ready": self.long_dash_ready,
            "long_dash_frac": self.long_dash_frac,
            "super_ready": self.super_ready,
            "super_charge_frac": self.super_charge_frac,
        }
