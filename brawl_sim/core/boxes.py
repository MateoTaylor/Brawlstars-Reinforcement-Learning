"""Box spawning, damage, and breaking. See BRAWL_SIM_BUILD_PLAN.md Step 22.

Boxes are static (box_pos never changes after spawn_boxes) and block nothing -- MapBank's
blocks_unit/blocks_proj have no notion of boxes at all, so units and projectiles pass over/
through their tile freely; only an explicit distance/cone/capsule check (projectiles.py,
hero.advance_dash, combat.melee_hitscan) ever "hits" one.

spawn_boxes generalizes the spec's "topk smallest n_boxes" to a full ascending sort of every
candidate spot (topk with k = min(cfg.max_boxes, MAX_BOX_SPOTS), largest=False) followed by a
`rank < n_boxes_env` mask, rather than calling topk with k = n_boxes directly. n_boxes is a
SimParams field (N08: scalar-or-range, resampled per env) and can legitimately differ across
envs in the same batch, but torch.topk takes one Python int k for the whole call -- there is no
"per-row k" -- so a single sorted ranking plus a per-row mask is the sync-free way to let each
env keep a different number of its own smallest-key spots. Assumes cfg.max_boxes <=
MAX_BOX_SPOTS (64, MapBank's fixed padded capacity, Step 6); true for every config in this
repo, and Step 5's map generation minimum (>=16 X markers) plus the default max_boxes=16 make
that comfortable headroom, not a coincidence to rely on carefully.

Because a full respawn (via reset_mask) always repopulates every one of the k selected slots
(and explicitly clears the rest), there's no need for projectiles.alloc_slots' incremental
collision-free allocator here -- that scheme exists for slots claimed piecemeal over many ticks
by many independent shooters/corpses; box respawn is a single all-at-once replace, so writing
ranks 0..k-1 directly into box slots 0..k-1 is both correct and simpler. resolve_broken_boxes,
by contrast, DOES reuse alloc_slots/_set_scalar/_set_vec2 from .projectiles for its pickup
spawn, for exactly the same reason combat.drop_cubes_on_death does: box-breaking happens
piecemeal, one box at a time, against the same shared pku_* pool other systems are also
claiming from on the same tick.
"""
import torch

from ..maps.loader import MAX_BOX_SPOTS
from . import projectiles as proj


def spawn_boxes(state, reset_mask: torch.Tensor, bank, params, cfg, gen) -> None:
    """MUTATES: box_pos, box_hp, box_max_hp, box_alive (masked rows only). Picks
    min(params.n_boxes, that map's n_box_spots, cfg.max_boxes) distinct spots per masked env,
    without replacement, via random keys + topk-smallest (see module docstring)."""
    B = state.box_pos.shape[1]
    device = state.box_pos.device
    N = reset_mask.shape[0]
    k = min(B, MAX_BOX_SPOTS)

    map_box_spots = bank.box_spots[state.map_id]  # (N, MAX_BOX_SPOTS, 2)
    n_spots_env = bank.n_box_spots[state.map_id]  # (N,)

    slot_idx = torch.arange(MAX_BOX_SPOTS, device=device).view(1, -1)
    valid_spot = slot_idx < n_spots_env.unsqueeze(-1)  # (N, MAX_BOX_SPOTS)

    keys = torch.rand((N, MAX_BOX_SPOTS), generator=gen, device=device)
    keys = torch.where(valid_spot, keys, torch.full_like(keys, float("inf")))

    _, sorted_idx = torch.topk(keys, k, largest=False, dim=1)  # (N, k), smallest keys first
    chosen_pos = torch.gather(map_box_spots, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, 2))

    n_boxes_env = torch.clamp(params.n_boxes, max=torch.clamp(n_spots_env, max=B))  # (N,)
    rank = torch.arange(k, device=device).view(1, -1)
    is_box = rank < n_boxes_env.unsqueeze(-1)  # (N, k)

    new_pos = state.box_pos.clone()
    new_pos[:, :k] = chosen_pos
    new_alive = state.box_alive.clone()
    new_alive[:, :k] = is_box
    new_alive[:, k:] = False

    box_hp_flat = params.box_hp.unsqueeze(-1).expand(-1, B)

    mask_e = reset_mask.unsqueeze(-1)  # (N,1), broadcasts against (N,B)
    state.box_pos.copy_(torch.where(mask_e.unsqueeze(-1), new_pos, state.box_pos))
    state.box_alive.copy_(torch.where(mask_e, new_alive, state.box_alive))
    hp_reset = torch.where(new_alive, box_hp_flat, torch.zeros_like(box_hp_flat))
    state.box_hp.copy_(torch.where(mask_e, hp_reset, state.box_hp))
    state.box_max_hp.copy_(torch.where(mask_e, hp_reset, state.box_max_hp))


def damage_boxes(state, dmg_box: torch.Tensor) -> None:
    """MUTATES: box_hp. dmg_box: (N,B), summed from every source (projectiles, AoE, melee
    cones, dash) by the caller before this is invoked -- clamped at 0."""
    state.box_hp.copy_(torch.clamp(state.box_hp - dmg_box, min=0))


def resolve_broken_boxes(state, params, cfg):
    """MUTATES: box_alive, pku_*, boxes_broken. Returns newly_broken (N,B) bool. One pickup
    per broken box, holding cubes_per_box cubes, spawned at the box's position -- reuses
    projectiles.alloc_slots/_set_scalar/_set_vec2 (see module docstring)."""
    newly_broken = state.box_alive & (state.box_hp <= 0)
    state.box_alive.copy_(state.box_alive & ~newly_broken)

    B = state.box_pos.shape[1]
    demand = newly_broken.to(torch.int64)  # (N,B), 0 or 1 per box
    idx3, ok3 = proj.alloc_slots(state.pku_alive, demand, max_per_entity=1)
    idx, ok = idx3.squeeze(-1), ok3.squeeze(-1)  # (N,B)

    cubes_new = params.cubes_per_box.unsqueeze(-1).expand(-1, B)
    age_new = torch.zeros_like(state.box_pos[..., 0])
    alive_new = torch.ones_like(ok)

    proj._set_vec2(state.pku_pos, idx, ok, state.box_pos)
    proj._set_scalar(state.pku_cubes, idx, ok, cubes_new)
    proj._set_scalar(state.pku_age, idx, ok, age_new)
    proj._set_scalar(state.pku_alive, idx, ok, alive_new)

    state.boxes_broken.copy_(state.boxes_broken + newly_broken.sum(dim=1).to(state.boxes_broken.dtype))

    return newly_broken
