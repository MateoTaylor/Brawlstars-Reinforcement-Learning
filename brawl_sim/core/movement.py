"""Non-dash movement: intent -> terrain slide. See BRAWL_SIM_BUILD_PLAN.md Step 12.

**Entities do not collide with each other, by design.** Terrain is solid; bodies are not. Two
units may occupy the same point, walk straight through one another, and stack arbitrarily deep
-- `unit_radius` is a hitbox for projectiles and melee (core/projectiles, core/combat) and a
spacing constraint for SPAWNING (core/spawn), never a movement obstacle.

This module used to run a "soft separation" pass here: a dense (N,E,E) spring that pushed
overlapping pairs apart every tick. It was removed deliberately, not lost. It did not implement
walk-through -- it implemented bumper cars. A unit walking into a stationary one shoved that one
7+ tiles across the map, and a unit walking into one pinned against a wall sank ~0.14 tiles into
it and stayed there, since the spring had nowhere to send the pinned body. Neither is a behaviour
anything wanted, and both were load-bearing enough to need this note: if a future step wants
bodies to interact, it should say which of "solid" or "springy" it means and build that, rather
than reintroduce the spring on the assumption it was the collision model.
"""
import torch

from . import geometry as geo
from . import melee_sweep
from . import stats
from . import terrain


def apply_movement(state, move_dir: torch.Tensor, bank, params, cfg) -> None:
    """move_dir: (N,E,2), normalized or zero. MUTATES: ent_pos, ent_vel, ent_facing."""
    active = state.ent_alive & (state.ent_dash_t <= 0)  # skipped: dead or dashing

    speed = stats.effective_speed(state.ent_kind, params)  # (N,E)
    norm_dir = geo.normalize(move_dir)
    delta = norm_dir * speed.unsqueeze(-1) * cfg.dt
    delta = delta * active.unsqueeze(-1).to(delta.dtype)  # multiplicative mask, not indexing

    radius = params.unit_radius.unsqueeze(-1)  # (N,1), broadcasts against (N,E)
    old_pos = state.ent_pos
    # Terrain is the ONLY thing that can stop a walk -- see the module docstring on why bodies
    # are not consulted here.
    pos_final = terrain.resolve_move(bank.blocks_unit, state.map_id, old_pos, delta, radius, cfg)

    state.ent_vel.copy_((pos_final - old_pos) / cfg.dt)
    state.ent_pos.copy_(pos_final)

    # Facing normally tracks movement -- except for an entity mid-SWEEP (Step C2 / D6): Buzz keeps
    # moving while his five hitscans land, but keeps facing the direction he started the attack in,
    # so the fan stays anchored where he aimed it. Without this the sweep would smear around
    # wherever he happened to walk and stop being a readable, dodgeable arc.
    #
    # Gated on the KIND being swept, not merely on `ent_attack_cd > 0`: since Step C1 every
    # brawler has a cooldown running after every attack, so the looser test would freeze all five
    # brawlers' facing for 0.25 s after every shot -- silently changing dash directions and every
    # melee cone in the game, far beyond what this step intends.
    sweeping = melee_sweep.is_swept(state, params) & (state.ent_attack_cd > 0)
    move_nonzero = ((move_dir[..., 0] != 0) | (move_dir[..., 1] != 0)) & ~sweeping
    state.ent_facing.copy_(torch.where(move_nonzero, geo.angle_of(move_dir), state.ent_facing))
