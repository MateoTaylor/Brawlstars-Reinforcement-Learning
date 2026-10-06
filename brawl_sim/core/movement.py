"""Non-dash movement: intent -> terrain slide.

**Entities do not collide with each other, by design.** Terrain is solid; bodies are not. Units
may overlap, walk through one another and stack -- `unit_radius` is a hitbox for projectiles and
melee (core/projectiles, core/combat), a spacing constraint for SPAWNING (core/spawn) and the
body's size against terrain, never an obstacle to another unit. A soft-separation spring is not a
stand-in collision model: it shoves a stationary unit across the map and sinks one pinned against
a wall into it. Body interaction, if ever wanted, has to be designed as "solid" or "springy" on
purpose.
"""
import torch

from . import geometry as geo
from . import melee_sweep
from . import stats
from . import terrain


def apply_movement(state, move_dir: torch.Tensor, bank, params, cfg) -> None:
    """move_dir: (N,E,2), length <= 1 or zero. MUTATES: ent_pos, ent_vel, ent_facing.

    The length is a throttle. The hero's decoded action is a unit vector or zero, so it always
    walks at full speed; a bot's intent is bots/policy.all_bot_intents's EMA-smoothed steering,
    which decays toward zero after the bot decides to stop. Renormalising that decaying vector
    would make a stopping bot coast at full speed along its old heading for tiles; scaling speed
    by the length makes the stop a short deceleration."""
    active = state.ent_alive & (state.ent_dash_t <= 0)  # skipped: dead or dashing

    speed = stats.effective_speed(state.ent_kind, params)  # (N,E)
    norm_dir = geo.normalize(move_dir)
    throttle = torch.clamp(geo.safe_norm(move_dir, dim=-1), max=1.0)
    delta = norm_dir * (speed * throttle).unsqueeze(-1) * cfg.dt
    delta = delta * active.unsqueeze(-1).to(delta.dtype)  # multiplicative mask, not indexing

    radius = params.unit_radius.unsqueeze(-1)  # (N,1), broadcasts against (N,E)
    old_pos = state.ent_pos
    # Terrain is the ONLY thing that can stop a walk -- see the module docstring on why bodies
    # are not consulted here.
    pos_final = terrain.resolve_move(bank.blocks_unit, state.map_id, old_pos, delta, radius, cfg)

    state.ent_vel.copy_((pos_final - old_pos) / cfg.dt)
    state.ent_pos.copy_(pos_final)

    # Facing tracks movement -- except mid-SWEEP: the swinger keeps moving while his hitscans land
    # but keeps facing the direction he started the attack in, so the fan stays a readable,
    # dodgeable arc instead of smearing wherever he walks. Gated on the KIND being swept, not on
    # `ent_attack_cd > 0` alone (see melee_sweep.is_swept).
    sweeping = melee_sweep.is_swept(state, params) & (state.ent_attack_cd > 0)
    move_nonzero = ((move_dir[..., 0] != 0) | (move_dir[..., 1] != 0)) & ~sweeping
    state.ent_facing.copy_(torch.where(move_nonzero, geo.angle_of(move_dir), state.ent_facing))
