"""The shrinking safe-zone rect: init, per-tick shrink schedule, damage, and the reusable grid
rasterizer for the observation's `in_zone` channel. See BRAWL_SIM_BUILD_PLAN.md Step 23.

`_outside_rect` duplicates bots/perception.in_zone's predicate (a two-line comparison) rather
than importing it: core/ modules never depend on bots/ (the dependency runs one way, bots/
already imports core/ freely) -- perception.py existed before this module and independently
needed the same check for D14's "bots flee the zone" (Step 15), but keeping core/ free of any
bots/ import is worth the two-line duplication rather than entangling the module graph or
relocating perception.in_zone's existing, already-tested public call sites over one predicate
this small.
"""
import torch

from . import geometry as geo


def _outside_rect(pos: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return (
        (pos[..., 0] < lo[..., 0]) | (pos[..., 0] > hi[..., 0])
        | (pos[..., 1] < lo[..., 1]) | (pos[..., 1] > hi[..., 1])
    )


def init_zone(state, reset_mask: torch.Tensor, params, cfg) -> None:
    """MUTATES: zone_lo, zone_hi, zone_next_t, zone_step (masked rows only). Resets the safe
    rect to the full map -- (0,0) to (map_w, map_h), "everything is safe" -- and schedules the
    first shrink at params.zone_start_time (config.py: start_fraction * max_episode_steps *
    dt). Unconditional on cfg.zone_enabled: even a disabled zone should carry a sane,
    non-degenerate rect (bots/policy.zone_contribution's degenerate-rect guard exists for the
    allocate()-zero-init case, not as a substitute for this)."""
    device = state.zone_lo.device
    N = reset_mask.shape[0]

    full_lo = torch.zeros((N, 2), device=device, dtype=state.zone_lo.dtype)
    full_hi = geo.vec2(cfg.map_w, cfg.map_h, device, state.zone_hi.dtype).expand(N, -1)

    mask = reset_mask.unsqueeze(-1)
    state.zone_lo.copy_(torch.where(mask, full_lo, state.zone_lo))
    state.zone_hi.copy_(torch.where(mask, full_hi, state.zone_hi))
    state.zone_next_t.copy_(torch.where(reset_mask, params.zone_start_time, state.zone_next_t))
    state.zone_step.copy_(torch.where(reset_mask, torch.zeros_like(state.zone_step), state.zone_step))


def step_zone(state, params, cfg) -> None:
    """MUTATES: zone_lo, zone_hi, zone_next_t, zone_step. No-op entirely when
    cfg.zone_enabled is False. Where state.time >= zone_next_t: lo += tiles_per_step,
    hi -= tiles_per_step, clamped (around the rect's own center, so the shrink stays
    symmetric) so hi - lo never drops below 2 on either axis; zone_next_t += zone_step_seconds;
    zone_step += 1. At most one shrink per call: config.validate() guarantees
    dt < zone_step_seconds, so state.time can never cross more than one zone_next_t boundary
    within a single tick."""
    if not cfg.zone_enabled:
        return

    due = state.time >= state.zone_next_t  # (N,)

    step = params.zone_tiles_per_step.unsqueeze(-1)  # (N,1), broadcasts against (N,2)
    center = (state.zone_lo + state.zone_hi) / 2.0
    new_lo = torch.min(state.zone_lo + step, center - 1.0)
    new_hi = torch.max(state.zone_hi - step, center + 1.0)

    due_b = due.unsqueeze(-1)
    state.zone_lo.copy_(torch.where(due_b, new_lo, state.zone_lo))
    state.zone_hi.copy_(torch.where(due_b, new_hi, state.zone_hi))
    state.zone_next_t.copy_(torch.where(due, state.zone_next_t + params.zone_step_seconds, state.zone_next_t))
    state.zone_step.copy_(torch.where(due, state.zone_step + 1, state.zone_step))


def current_fraction(state, params) -> torch.Tensor:
    """(N,) f32 -- the FRACTION OF MAX HP the zone deals per second right now, escalating with
    each shrink (`zone_hp_fraction + zone_fraction_growth * zone_step`).

    **This replaced a flat HP/s rate in Step B3, and the change is a change of SHAPE, not of
    magnitude.** The requirement (bot_overhaul.md D15) is that nothing survives more than ~5
    seconds in the zone -- which has to hold for a 6000 HP Brock and for a fully cube-stacked
    34000 HP Buzz alike. A flat rate cannot express that: at the old `dps: 1000` a base-HP Buzz
    lasted 10s and a cubed one lasted 34s, so power cubes bought zone tankiness and the player who
    most needed pushing out of the zone could ignore it longest. A fraction of max HP gives every
    body the same countdown, and it is also what the real game does -- `configs/default.yaml`
    already flagged the flat rate as a known divergence of shape.
    """
    growth = params.zone_fraction_growth * state.zone_step.to(params.zone_hp_fraction.dtype)
    return params.zone_hp_fraction + growth


def current_dps(state, params) -> torch.Tensor:
    """(N,) f32 -- the zone's current damage rate **for the hero specifically**, in HP/s.

    Since Step B3 the zone's rate is proportional, so there is no single HP/s figure that
    describes it any more -- every entity takes a different amount. This exists for
    `obs["zone"]["dps"]`, whose contract is one `(N,)` float, and resolving it against the hero's
    own max HP is both the shape that field needs and the number actually useful to the agent
    ("how fast does the zone kill ME"), rather than a rate abstracted away from any body.
    """
    return current_fraction(state, params) * state.ent_max_hp[:, 0]


def zone_damage(state, params, cfg) -> torch.Tensor:
    """(N,E) f32. All zero when cfg.zone_enabled is False. Otherwise
    `current_fraction * ent_max_hp * dt` for every alive entity outside [zone_lo, zone_hi].

    Per-entity, not one broadcast scalar: `ent_max_hp` is already cube-scaled
    (core/stats.effective_max_hp), so collecting power cubes raises the absolute damage taken in
    exact proportion and never extends how long an entity can stand in the zone. See
    `current_fraction` for why that is the point.
    """
    if not cfg.zone_enabled:
        return torch.zeros_like(state.ent_hp)

    zone_lo = state.zone_lo.unsqueeze(1)  # (N,1,2)
    zone_hi = state.zone_hi.unsqueeze(1)
    outside = _outside_rect(state.ent_pos, zone_lo, zone_hi)  # (N,E)

    frac = current_fraction(state, params).unsqueeze(-1)  # (N,1), broadcasts against (N,E)
    dmg = frac * state.ent_max_hp * cfg.dt
    zero = torch.zeros_like(dmg)
    return torch.where(outside & state.ent_alive, dmg, zero)


def zone_grid(state, cfg, out_h: int, out_w: int, origin) -> torch.Tensor:
    """(N, out_h, out_w) bool -- True where that cell's tile center is outside
    [zone_lo, zone_hi] ("in the zone", the damaging area). `origin`: (N,2) f32 (or anything
    broadcastable to it) tile-space (x, y) of the grid's column-0/row-0 corner. Reused for both
    the observation's world grid (origin=(0,0), out_h/out_w = map_h/map_w) and its egocentric
    view grid (origin = that env's view-crop top-left tile, Step 25) -- the same rasterization
    either way, just a different window."""
    N = state.zone_lo.shape[0]
    device = state.zone_lo.device

    origin = torch.as_tensor(origin, device=device, dtype=state.zone_lo.dtype)
    if origin.dim() == 1:
        origin = origin.unsqueeze(0).expand(N, -1)

    xs = torch.arange(out_w, device=device, dtype=state.zone_lo.dtype) + 0.5
    ys = torch.arange(out_h, device=device, dtype=state.zone_lo.dtype) + 0.5
    grid_x = (origin[:, 0].view(N, 1, 1) + xs.view(1, 1, out_w)).expand(N, out_h, out_w)
    grid_y = (origin[:, 1].view(N, 1, 1) + ys.view(1, out_h, 1)).expand(N, out_h, out_w)
    pos = torch.stack([grid_x, grid_y], dim=-1)  # (N, out_h, out_w, 2)

    zone_lo = state.zone_lo.view(N, 1, 1, 2)
    zone_hi = state.zone_hi.view(N, 1, 1, 2)
    return _outside_rect(pos, zone_lo, zone_hi)
