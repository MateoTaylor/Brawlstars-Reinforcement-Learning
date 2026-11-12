"""Status effects (BULL_HERO_PLAN.md D4): slow, stun and airborne, as per-entity countdowns.

Four (N,E) SimState fields, zeroed on reset by `state.zero_`, where 0 means "unaffected":

  - `ent_slow_t` and `ent_slow_frac`: walking speed x (1 - frac) while `slow_t > 0`
    (`movement.apply_movement`). A dash is not slowed.
  - `ent_stun_t`: no walking or turning, and no attack, super, gadget or melee sub-swing
    (`movement.apply_movement`, `env._attack_phase`, `hero.action_mask`).
  - `ent_airborne_t`: stunned, AND untargetable. Projectile hits and blasts
    (`projectiles.step_projectiles`), melee cones (`combat.melee_hitscan`) and dash capsules
    (`hero.advance_dash`) skip it, so a shot flies on through it to whatever stands behind. The
    hero's auto-aim, his tap-to-fire super and the reward's in-reach flag skip it too. The gas
    still burns it.

`hero.tick_timers` counts the three timers down in phase 2 (`tick`). Bull's charge super applies
airborne to each player it hits and a slow around its landing (`hero.advance_dash`, through
`apply_airborne` and `apply_slow`), and starts the charger's own copy of that slow's clock,
`ent_slow_window_t`, which `tick` counts down beside it (the obs' `hero.slow_window`). His
Stomper slows the players around him, or stuns the ones already slowed by anything
(`hero.advance_stomp`, through `apply_slow` and `apply_stun`). Only the hero applies them, so
nothing stuns the hero and the live shadow's copy of `hero.action_mask` needs no stun term. Every
gate is an AND with one of the predicates below, all-False while the timers are zero, or a
`torch.where` that then keeps the old value, so a run in which nothing applies an effect (every
Mortis run) is bit-identical to one without them (scripts/probes/hero_seam_digest.py).

Bots do not know they are stunned: their intent is computed as usual and gated like anyone's. Nor
do they skip an airborne target, so a bot may spend a shot that flies through one. A dash already
running when its dasher is stunned finishes; no dasher can be stunned yet (no bot kind dashes).
"""
import torch


def count_down(t: torch.Tensor, dt: float) -> torch.Tensor:
    """`t` one tick later: `t - dt`, or exactly 0 once less than half a tick is left.

    Without the snap, float32 subtraction leaves ~1e-8 s after the last full tick of 0.25, 0.3,
    1.5, 2.0, 2.5 and 3.0 s (measured 2026-10-23), which would hold the effect a tick longer. With
    it an effect lasts its duration rounded to whole ticks: set during tick t, a k-tick effect is
    in force through tick t + k - 1 and gone on tick t + k. The victim acts again `k * dt` seconds
    after it was hit, as `attack_cd`, snapped the same way, lets a brawler fire again
    `attack_cooldown` after a shot."""
    left = t - dt
    return torch.where(left > 0.5 * dt, left, torch.zeros_like(left))


def tick(state, cfg) -> None:
    """MUTATES: ent_slow_t, ent_slow_frac, ent_stun_t, ent_airborne_t, and the chargers'
    ent_slow_window_t. Phase 2, from `hero.tick_timers`. The slow's fraction is cleared on the
    tick its timer runs out, so `ent_slow_frac` is nonzero only while the slow lasts
    (`state.check_invariants`). The window counts down right beside the slow, by the same rule, so
    a charger's window and the landing slow on every player it caught reach 0 on the same tick."""
    state.ent_slow_t.copy_(count_down(state.ent_slow_t, cfg.dt))
    state.ent_slow_window_t.copy_(count_down(state.ent_slow_window_t, cfg.dt))
    state.ent_stun_t.copy_(count_down(state.ent_stun_t, cfg.dt))
    state.ent_airborne_t.copy_(count_down(state.ent_airborne_t, cfg.dt))
    state.ent_slow_frac.copy_(
        torch.where(state.ent_slow_t > 0, state.ent_slow_frac, torch.zeros_like(state.ent_slow_frac))
    )


def disabled(state) -> torch.Tensor:
    """(N,E) bool -- stunned or airborne: no walking, turning, attack, super or gadget."""
    return (state.ent_stun_t > 0) | (state.ent_airborne_t > 0)


def airborne(state) -> torch.Tensor:
    """(N,E) bool -- in the air, so every hit test and target pick skips it."""
    return state.ent_airborne_t > 0


def walk_speed(state, speed: torch.Tensor) -> torch.Tensor:
    """(N,E) -- `speed` x (1 - ent_slow_frac) where slowed, else `speed` itself."""
    return torch.where(state.ent_slow_t > 0, speed * (1.0 - state.ent_slow_frac), speed)


def apply_slow(state, hit: torch.Tensor, frac: torch.Tensor, seconds: torch.Tensor) -> None:
    """MUTATES: ent_slow_t, ent_slow_frac on the `hit` (N,E) rows, to `frac` for `seconds` (both
    (N,E)). A new slow replaces the running one only if it lasts longer, and then whole, fraction
    and all: two slows neither stack nor blend. A `seconds` of 0 changes nothing, so a kind with no
    slow can pass its zeros straight through."""
    longer = hit & (seconds > state.ent_slow_t)
    state.ent_slow_t.copy_(torch.where(longer, seconds, state.ent_slow_t))
    state.ent_slow_frac.copy_(torch.where(longer, frac, state.ent_slow_frac))


def apply_stun(state, hit: torch.Tensor, seconds: torch.Tensor) -> None:
    """MUTATES: ent_stun_t on the `hit` (N,E) rows: the longer of what is left and `seconds`
    (N,E), as `apply_airborne`. A running slow is left alone and keeps counting down under the
    stun."""
    state.ent_stun_t.copy_(torch.where(hit, torch.maximum(state.ent_stun_t, seconds), state.ent_stun_t))


def apply_airborne(state, hit: torch.Tensor, seconds: torch.Tensor) -> None:
    """MUTATES: ent_airborne_t on the `hit` (N,E) rows: the longer of what is left and `seconds`
    (N,E), so a second launch never shortens the first."""
    state.ent_airborne_t.copy_(
        torch.where(hit, torch.maximum(state.ent_airborne_t, seconds), state.ent_airborne_t)
    )
