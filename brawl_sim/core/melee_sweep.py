"""Swept multi-hitscan melee: one attack that lands `hitscan_count` cones at different angles
across `attack_cooldown` seconds (Buzz's fan; Edgar's two-hit combo, with a zero sweep).
`hitscan_count <= 1` is the single instantaneous cone, so a non-swept kind pays only a masked
`torch.where`, and a new sweep is a config block, not a code change.

No SimState fields of its own -- the clock and the anchor already exist:

  - The CLOCK is `ent_attack_cd`, the "no attack, no reload" window the attack opens; the sweep
    spans it. Elapsed time into the attack is `attack_cooldown - ent_attack_cd`.
  - The ANCHOR is `ent_facing`, which `core/movement.apply_movement` stops updating while a sweep
    runs (the swinger keeps moving but keeps facing his initial attack direction).

The schedule is stateless, derived from elapsed time rather than a counter:

    interval = attack_cooldown / hitscan_count
    k_now    = floor(elapsed / interval)
    k_prev   = floor((elapsed - dt) / interval)
    fire sub-swing k_now  <=>  k_now > k_prev  and  k_now < hitscan_count

A floor() crossing cannot double-fire or drop a swing even when `dt` does not divide `interval`,
and needs nothing reset on death or respawn (core/projectiles' hazard ticks use the same trick).

Angles sweep CLOCKWISE, which is increasing angle here: y increases downward (CONVENTIONS.md
"Coordinates"). Sub-swing `k` is centred at `facing - sweep/2 + k * sweep/(count-1)`, so the fan
spans exactly `hitscan_sweep_rad` end to end, symmetric about the anchored facing.
"""
import torch

from . import stats

_EPS = 1e-6


def is_swept(state, params) -> torch.Tensor:
    """(N,E) bool -- does this entity's kind fire a multi-cone sweep rather than one cone?

    Gates the facing freeze in core/movement.apply_movement. Gated on the KIND, not on
    `ent_attack_cd > 0` alone: every brawler has a cooldown running after every attack, so
    freezing facing on cooldown would pin every kind's facing after every shot, silently
    altering dash direction and every melee cone.
    """
    return stats.gather_kind(params.hitscan_count, state.ent_kind) > 1


def sweep_cone_dir(state, params, cfg):
    """(fire (N,E) bool, cone_dir (N,E) radians) for THIS tick's sub-swing.

    `fire` is True only on the ticks a sub-swing actually lands (`hitscan_count` per attack).
    `cone_dir` is meaningless where `fire` is False.

    Entities whose kind is not swept never fire from here (their `hitscan_count <= 1` makes
    `is_swept` False); their single cone is resolved by the ordinary attack path in
    `env._attack_phase` on the tick they pull the trigger.
    """
    count = stats.gather_kind(params.hitscan_count, state.ent_kind).to(torch.float32)
    sweep = stats.gather_kind(params.hitscan_sweep_rad, state.ent_kind)
    cooldown = torch.clamp(stats.gather_kind(params.attack_cooldown, state.ent_kind), min=_EPS)

    swept = is_swept(state, params) & state.ent_alive
    # A sweep is running exactly while its cooldown is: for a swept kind only the attack sets
    # ent_attack_cd = attack_cooldown (env._attack_phase), and tick_timers counts it down.
    running = swept & (state.ent_attack_cd > 0)

    elapsed = cooldown - state.ent_attack_cd
    interval = cooldown / torch.clamp(count, min=1.0)
    k_now = torch.floor(elapsed / interval)
    k_prev = torch.floor((elapsed - cfg.dt) / interval)

    fire = running & (k_now > k_prev) & (k_now < count)

    # Sub-swing k is centred at facing - sweep/2 + k*sweep/(count-1). `count - 1` is clamped so a
    # count of 1 cannot divide by zero; such a kind is not `swept`, its `fire` is False, and the
    # value computed for it is discarded (compute for everyone, select later).
    denom = torch.clamp(count - 1.0, min=1.0)
    offset = -sweep / 2.0 + k_now * (sweep / denom)
    return fire, state.ent_facing + offset
