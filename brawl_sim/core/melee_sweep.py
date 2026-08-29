"""Swept multi-hitscan melee: one attack that lands `hitscan_count` cones at different angles
across `attack_cooldown` seconds. See bot_overhaul.md Step C2 / D6 / D9.

**This generalizes melee rather than special-casing Buzz.** `hitscan_count <= 1` reproduces the
single instantaneous cone every melee brawler had before this module existed, so every non-swept
kind resolves to the old behavior by construction and pays nothing but a masked `torch.where`.
A future brawler with a different sweep is a config block, not a code change.

**No new SimState fields.** The sweep needs a clock and an anchor, and both already exist:

  - The CLOCK is `ent_attack_cd`. Step C1 made it the "no attack, no reload" window, and for Buzz
    that window IS the sweep (`attack_cooldown: 1.00`, five hitscans at 0.2 s spacing). Elapsed
    time into the attack is `attack_cooldown - ent_attack_cd`.
  - The ANCHOR is `ent_facing`, which `core/movement.apply_movement` stops updating while a sweep
    is running (D6: Buzz keeps moving but keeps facing his initial attack direction). So the
    facing at any point during the sweep is still the facing the sweep began with.

**The schedule is stateless**, derived from elapsed time rather than a counter:

    interval = attack_cooldown / hitscan_count
    k_now    = floor(elapsed / interval)
    k_prev   = floor((elapsed - dt) / interval)
    fire sub-swing k_now  <=>  k_now > k_prev  and  k_now < hitscan_count

There is deliberately no `ent_attack_seq` field to keep in sync. The same trick core/projectiles
uses for hazard tick scheduling: a floor() crossing cannot double-fire or drop a swing even when
`dt` does not divide `interval` evenly, and it needs nothing reset on death or respawn.

**Angles sweep CLOCKWISE**, which is increasing angle here: y increases downward
(CONVENTIONS.md "Coordinates"), so a positive rotation from +x toward +y is clockwise on screen.
Sub-swing `k` is centred at `facing - sweep/2 + k * sweep/(count-1)`, i.e. the fan spans exactly
`hitscan_sweep_rad` end to end and is symmetric about the anchored facing.
"""
import torch

from . import stats

_EPS = 1e-6


def is_swept(state, params) -> torch.Tensor:
    """(N,E) bool -- does this entity's kind fire a multi-cone sweep rather than one cone?

    Used to gate the facing freeze in core/movement.apply_movement. **Gated on the KIND, not on
    `ent_attack_cd > 0` alone**: every brawler now has a cooldown running after every attack, so
    freezing facing for "anyone on cooldown" would pin the facing of all five brawlers for
    0.25 s after every shot -- a far larger behavior change than this step is scoped for, and one
    that would silently alter dash direction and every melee cone in the game.
    """
    return stats.gather_kind(params.hitscan_count, state.ent_kind) > 1


def sweep_cone_dir(state, params, cfg):
    """(fire (N,E) bool, cone_dir (N,E) radians) for THIS tick's sub-swing.

    `fire` is True only on the ticks a sub-swing actually lands -- for Buzz that is 5 ticks out of
    the 20 his 1.0 s attack spans. `cone_dir` is meaningless where `fire` is False.

    Entities whose kind is not swept never fire from here (their `hitscan_count <= 1` makes
    `is_swept` False); their single cone is resolved by the ordinary attack path in
    `env._attack_phase` on the tick they pull the trigger.
    """
    count = stats.gather_kind(params.hitscan_count, state.ent_kind).to(torch.float32)
    sweep = stats.gather_kind(params.hitscan_sweep_rad, state.ent_kind)
    cooldown = torch.clamp(stats.gather_kind(params.attack_cooldown, state.ent_kind), min=_EPS)

    swept = is_swept(state, params) & state.ent_alive
    # A sweep is running exactly while its cooldown is: C1 set ent_attack_cd = attack_cooldown at
    # the moment of the attack, and nothing else writes it.
    running = swept & (state.ent_attack_cd > 0)

    elapsed = cooldown - state.ent_attack_cd
    interval = cooldown / torch.clamp(count, min=1.0)
    k_now = torch.floor(elapsed / interval)
    k_prev = torch.floor((elapsed - cfg.dt) / interval)

    fire = running & (k_now > k_prev) & (k_now < count)

    # Sub-swing k is centred at facing - sweep/2 + k*sweep/(count-1). `count - 1` is clamped so a
    # hypothetical count of 1 cannot divide by zero; such a kind is not `swept` and its `fire` is
    # False regardless, so the value computed for it is discarded garbage (the same
    # compute-for-everyone-select-later discipline the archetype combat functions use).
    denom = torch.clamp(count - 1.0, min=1.0)
    offset = -sweep / 2.0 + k_now * (sweep / denom)
    return fire, state.ent_facing + offset


def start_sweep_hits_now(state, params) -> torch.Tensor:
    """(N,E) bool -- entities whose kind is swept, for masking OUT of the ordinary one-cone attack
    path on the tick they pull the trigger.

    A swept attack must not also land an instantaneous cone at trigger time: `sweep_cone_dir`
    already fires sub-swing 0 on that same tick (elapsed = 0 crosses the k=0 boundary), so letting
    the ordinary path through as well would double the first swing's damage.
    """
    return is_swept(state, params)
