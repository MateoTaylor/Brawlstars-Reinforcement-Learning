"""The full, unredacted observation package: everything the sim knows, before any fairness
gating (Phase J / `obs_select.py`) trims it down for the agent. See BRAWL_SIM_BUILD_PLAN.md
Step 25.

Layering note (same discipline as `core/zone.py`, Step 23, and `core/spawn.py`, Step 24): this
module never imports `bots/`. `vis` (`bots/perception.visibility`) and `raw_los`
(`bots/perception.raw_los`) are precomputed by the caller -- the future `env.py` step loop,
Section 4 phase 16, which already needs both for bot targeting earlier in the same tick -- and
handed in as plain tensors. The only two pieces of `bots/perception.py` this module would
otherwise need are `in_bush` and `team_id`, and both are cheap enough to inline locally rather
than cross the layering boundary: `team_id` is a one-line `kind.clone()` (Solo Showdown is FFA,
so "team" is inert forward-compat infrastructure, per `perception.team_id`'s own docstring);
`_in_bush` below is the same 5-line tile lookup as `perception.in_bush`, duplicated for the same
reason `core/zone.py`'s `_outside_rect` duplicates `perception.in_zone`'s predicate.

`rank` is NOT a stored state field. Section 4's phase-13 pseudocode ("death resolution ... record
rank + cause") predates `combat.py` (Step 14), whose actual `resolve_deaths` never grew a rank
field -- there's no `ent_rank` anywhere in `SimState`. Rather than retrofit Step 14 to store one
(a schema change to already-tested, already-shipped code, for a value that's fully recoverable
from fields it already writes), `compute_rank` below derives it live every call from
`ent_alive`/`ent_death_step`. See its own docstring for the exact tie-breaking rule (and, as of
Step 27, for the 1-indexed-vs-0-indexed footgun between here and `core/events.py`).

View/world grid unification: `_build_grid` is the one 12-channel rasterizer for BOTH
`obs["view"]` (egocentric, hero-centered, `view_h x view_w`) and `obs["world"]`
(`origin=(0,0)`, the full map) -- "the same rasterization either way, just a different window",
exactly the pattern `core/zone.py`'s `zone_grid` (Step 23) already established and that this
module reuses directly for channel 4. Terrain channels 0-3 always read the PADDED bank tensors
(`bank.pad_*`, Step 6) with a `+ pad_h`/`+ pad_w` index shift regardless of which grid is being
built: for `obs["world"]` this shift lands exactly back on the unpadded region (Step 6's own
`test_build_map_bank_padded_shape_and_blocking` already established
`pad_tiles[:, pad_h:-pad_h, pad_w:-pad_w] == tiles`), so one indexing path serves both without
a branch. Occupancy channels 5-11 accumulate into a separate int32 buffer via
`index_put_(accumulate=True)` (uint8 has no atomic-add guarantee) and are clamped into
`[0, 255]` only at the very end when cast down to the `uint8` the plan specifies.

`build_obs` is a plain function, not a class with buffers preallocated in `__init__`, despite
the step text's "preallocate every output buffer in `__init__`; `build_obs` allocates nothing"
line. Every other hot-path module in this codebase (`movement.apply_movement`,
`combat.melee_hitscan`, `projectiles.step_projectiles`, ...) is a plain function that allocates
fresh intermediates every call, and "allocates nothing" has consistently meant "steady-state
resident memory doesn't grow" (verified via memory-delta, not literal zero `torch.zeros` calls)
ever since Step 21's stricter check established that reading of the phrase. A class that owns
persistent, in-place-written buffers would matter for `torch.compile`'s cudagraph mode (Step
31), but nothing before that step needs it, and no other module through Step 24 introduced one
either -- so this keeps the same shape as its neighbors and defers the buffer-owning class to
whichever step actually wires up `torch.compile`.
"""
import torch

from ..constants import N_KINDS, N_PROJ_CLASSES, N_PROJ_KINDS, ProjClass
from . import geometry as geo
from . import hero as hero_mod
from . import stats
from . import terrain
from . import zone as zone_mod

_HERO = 0  # entity slot 0 is always the hero (Section 2's "Shapes" rule)
_EPS = 1e-6
_ARTILLERY = int(ProjClass.ARTILLERY)


def _in_bush(state, bank) -> torch.Tensor:
    """(N,E) bool -- see module docstring on why this duplicates perception.in_bush."""
    ix, iy = terrain.to_tile(state.ent_pos)
    map_id = terrain._broadcast_map_id(state.map_id, ix.shape)
    h, w = bank.is_bush.shape[1:]
    ix = torch.clamp(ix, 0, w - 1)
    iy = torch.clamp(iy, 0, h - 1)
    return bank.is_bush[map_id, iy, ix]


def compute_rank(alive: torch.Tensor, death_step: torch.Tensor) -> torch.Tensor:
    """(N,E) i64, 1-INDEXED placement (1 = best). Every currently-alive entity ties at
    `rank == n_alive` (their eventual finishing order isn't decided until they die, so they're
    all still "in the running" for any of the top spots). A dead entity's rank is fixed at
    1 + however many entities ended up outliving it: everyone still alive now, plus anyone who
    died later (higher death_step). See module docstring for why this is computed live rather
    than stored.

    Public (promoted on its second use, same pattern as zone.current_dps): core/events.py
    (Step 27) reuses this for info["hero_rank"] -- 0-INDEXED there (0 = winner) per that step's
    own acceptance text, so that call site subtracts 1. Two different indexing conventions on
    the same underlying number is a real footgun; both call sites say so."""
    E = alive.shape[1]
    alive_i = alive.unsqueeze(2)   # (N,E,1)
    alive_j = alive.unsqueeze(1)   # (N,1,E)
    death_i = death_step.unsqueeze(2)
    death_j = death_step.unsqueeze(1)

    outlives = torch.where(alive_i, alive_j, alive_j | (death_j > death_i))
    eye = torch.eye(E, dtype=torch.bool, device=alive.device).unsqueeze(0)
    outlives = outlives & ~eye
    return 1 + outlives.sum(dim=2).to(torch.int64)


def _view_origin(state, cfg) -> torch.Tensor:
    """(N,2) i64 -- real (unpadded) tile-space (x, y) of the egocentric view crop's top-left
    corner, hero-centered."""
    hero_ix, hero_iy = terrain.to_tile(state.ent_pos[:, _HERO])
    return torch.stack([hero_ix - cfg.view_w // 2, hero_iy - cfg.view_h // 2], dim=-1)


def _in_window(pos: torch.Tensor, origin: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """pos: (N,K,2). origin: (N,2) i64 real tile-space top-left. (N,K) bool."""
    ix = torch.floor(pos[..., 0]).to(torch.int64) - origin[:, 0].unsqueeze(-1)
    iy = torch.floor(pos[..., 1]).to(torch.int64) - origin[:, 1].unsqueeze(-1)
    return (ix >= 0) & (ix < out_w) & (iy >= 0) & (iy < out_h)


def _scatter_count(
    occ: torch.Tensor, channel: int, pos: torch.Tensor, valid: torch.Tensor,
    origin: torch.Tensor, out_h: int, out_w: int,
) -> None:
    """occ: (N,C,out_h,out_w) i32. MUTATES occ[:, channel] += 1 at each valid occupant's cell
    (pos: (N,K,2), valid: (N,K)). Out-of-window occupants are clamped to a harmless cell and
    contribute a zero delta -- same "invalid entries collide but add nothing" trick as
    projectiles._set_scalar."""
    N, K = valid.shape
    ix = torch.floor(pos[..., 0]).to(torch.int64) - origin[:, 0].unsqueeze(-1)
    iy = torch.floor(pos[..., 1]).to(torch.int64) - origin[:, 1].unsqueeze(-1)
    in_window = valid & (ix >= 0) & (ix < out_w) & (iy >= 0) & (iy < out_h)

    ix_safe = torch.clamp(ix, 0, out_w - 1).reshape(-1)
    iy_safe = torch.clamp(iy, 0, out_h - 1).reshape(-1)
    n_idx = torch.arange(N, device=pos.device).view(N, 1).expand(N, K).reshape(-1)
    ch_idx = torch.full_like(n_idx, channel)
    delta = in_window.reshape(-1).to(torch.int32)

    occ.index_put_((n_idx, ch_idx, iy_safe, ix_safe), delta, accumulate=True)


def _build_grid(state, bank, vis: torch.Tensor, cfg, origin: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """(N,12,out_h,out_w) uint8. See module docstring for the channel layout and the
    view/world unification. `origin`: (N,2) i64 real tile-space top-left corner."""
    N, E = state.ent_pos.shape[:2]
    device = state.ent_pos.device
    pad_h, pad_w = cfg.view_h // 2, cfg.view_w // 2
    pad_hgt, pad_wid = bank.pad_tiles.shape[1], bank.pad_tiles.shape[2]

    dx = torch.arange(out_w, device=device, dtype=torch.int64)
    dy = torch.arange(out_h, device=device, dtype=torch.int64)
    ix_pad = origin[:, 0].view(N, 1, 1) + dx.view(1, 1, out_w) + pad_w
    iy_pad = origin[:, 1].view(N, 1, 1) + dy.view(1, out_h, 1) + pad_h
    ix_pad = torch.clamp(ix_pad.expand(N, out_h, out_w), 0, pad_wid - 1)
    iy_pad = torch.clamp(iy_pad.expand(N, out_h, out_w), 0, pad_hgt - 1)
    map_id_b = state.map_id.view(N, 1, 1).expand(N, out_h, out_w)

    grid = torch.zeros((N, 12, out_h, out_w), dtype=torch.uint8, device=device)
    grid[:, 0] = bank.pad_blocks_unit[map_id_b, iy_pad, ix_pad].to(torch.uint8)
    grid[:, 1] = bank.pad_blocks_proj[map_id_b, iy_pad, ix_pad].to(torch.uint8)
    grid[:, 2] = bank.pad_is_bush[map_id_b, iy_pad, ix_pad].to(torch.uint8)
    grid[:, 3] = bank.pad_is_water[map_id_b, iy_pad, ix_pad].to(torch.uint8)
    grid[:, 4] = zone_mod.zone_grid(state, cfg, out_h, out_w, origin.to(state.ent_pos.dtype)).to(torch.uint8)

    # local 0..6 -> channels 5..11
    occ = torch.zeros((N, 7, out_h, out_w), dtype=torch.int32, device=device)
    not_hero = (torch.arange(E, device=device).view(1, E) != _HERO)
    enemy_alive = state.ent_alive & not_hero
    revealed_to_hero = vis[:, _HERO, :]

    _scatter_count(occ, 0, state.ent_pos, enemy_alive, origin, out_h, out_w)
    _scatter_count(occ, 1, state.ent_pos, enemy_alive & revealed_to_hero, origin, out_h, out_w)
    _scatter_count(occ, 2, state.ent_pos, enemy_alive & ~revealed_to_hero, origin, out_h, out_w)
    _scatter_count(occ, 3, state.ent_pos[:, _HERO:_HERO + 1], state.ent_alive[:, _HERO:_HERO + 1], origin, out_h, out_w)
    _scatter_count(occ, 4, state.box_pos, state.box_alive, origin, out_h, out_w)
    _scatter_count(occ, 5, state.pku_pos, state.pku_alive, origin, out_h, out_w)
    _scatter_count(occ, 6, state.prj_pos, state.prj_alive, origin, out_h, out_w)

    grid[:, 5:12] = torch.clamp(occ, max=255).to(torch.uint8)
    return grid


def clone_obs(node):
    """Recursively `.clone()`s every tensor leaf, preserving dict structure. `build_obs`'s
    fields are views/direct references into `state`'s own tensors (zero-copy, matching every
    other hot-path function's "mutate in place" style, Step 25) -- fine for the routinely
    returned per-step observation, which is documented as valid only until the next `step()`/
    `reset()` call (Step 29), but NOT fine for anything that must survive a LATER in-place
    mutation of `state` within that same call, such as `env.py`'s `info["final_observation"]`/
    `info["final_info"]`: autoreset's `spawn.reset_envs` mutates `state` in place moments after
    the "final" obs/info is captured, which would otherwise retroactively corrupt it with the
    fresh post-reset values instead of the episode's actual last tick (a real bug this exact
    function was added to fix, not a hypothetical one). Also handles `compute_info`'s output
    (a flat dict, not nested) identically -- the recursion just bottoms out one level sooner."""
    if isinstance(node, dict):
        return {key: clone_obs(value) for key, value in node.items()}
    return node.clone()


def build_obs(state, bank, vis: torch.Tensor, raw_los: torch.Tensor, params, cfg) -> dict:
    """Returns the full nested-dict observation (D08/N12) -- every entity/projectile/box/
    pickup, always present, index-stable, never masked. `vis`/`raw_los`: (N,E,E) bool,
    precomputed by the caller (see module docstring)."""
    N, E = state.ent_pos.shape[:2]
    P = state.prj_pos.shape[1]
    device = state.ent_pos.device

    map_wh = geo.vec2(cfg.map_w, cfg.map_h, device, state.ent_pos.dtype)

    hero_pos = state.ent_pos[:, _HERO]
    hero_vel = state.ent_vel[:, _HERO]
    hero_facing = state.ent_facing[:, _HERO]

    # ---- shared cross-subtree intermediates ----
    dist_matrix = geo.safe_norm(state.ent_pos.unsqueeze(2) - state.ent_pos.unsqueeze(1), dim=-1)  # (N,E,E)
    hero_dist = dist_matrix[:, _HERO, :]  # (N,E)

    in_bush_all = _in_bush(state, bank)
    zone_lo_e, zone_hi_e = state.zone_lo.unsqueeze(1), state.zone_hi.unsqueeze(1)
    in_zone_all = zone_mod._outside_rect(state.ent_pos, zone_lo_e, zone_hi_e)  # (N,E)
    can_attack_all = (
        state.ent_alive & (state.ent_ammo >= 1.0) & (state.ent_attack_cd <= 0) & (state.ent_dash_t <= 0)
    )  # generalizes hero.action_mask's fire_ok formula to all E entities
    max_ammo_all = stats.gather_kind(params.max_ammo, state.ent_kind)
    super_ready_all = hero_mod.super_ready(state, params)
    super_frac_all = hero_mod.super_charge_frac(state, params)
    long_dash_ready_all = hero_mod.long_dash_ready(state, params)
    long_dash_frac_all = hero_mod.long_dash_charge_frac(state, params)
    rank_all = compute_rank(state.ent_alive, state.ent_death_step)

    revealed_to_hero = vis[:, _HERO, :]  # hero sees j
    hero_revealed_to = vis[:, :, _HERO]  # j sees hero
    los_from_hero = raw_los[:, _HERO, :]
    hidden_by_bush = in_bush_all & ~revealed_to_hero

    view_origin = _view_origin(state, cfg)

    obs: dict = {}

    # ---- hero ----
    obs["hero"] = {
        "pos": hero_pos,
        "pos_norm": hero_pos / map_wh,
        "vel": hero_vel,
        "facing": hero_facing,
        "facing_vec": geo.from_angle(hero_facing),
        "hp": state.ent_hp[:, _HERO],
        "max_hp": state.ent_max_hp[:, _HERO],
        "hp_frac": state.ent_hp[:, _HERO] / torch.clamp(state.ent_max_hp[:, _HERO], min=_EPS),
        "alive": state.ent_alive[:, _HERO],
        "cubes": state.ent_cubes[:, _HERO],
        "ammo": state.ent_ammo[:, _HERO],
        "ammo_frac": state.ent_ammo[:, _HERO] / torch.clamp(max_ammo_all[:, _HERO], min=_EPS),
        "ammo_whole": torch.floor(state.ent_ammo[:, _HERO]).to(torch.int64),
        "attack_cd": state.ent_attack_cd[:, _HERO],
        "can_attack": can_attack_all[:, _HERO],
        "dashing": state.ent_dash_t[:, _HERO] > 0,
        "dash_t": state.ent_dash_t[:, _HERO],
        "dash_dir": state.ent_dash_dir[:, _HERO],
        "super_ready": super_ready_all[:, _HERO],
        "super_charge": state.ent_super_charge[:, _HERO],
        "super_charge_frac": super_frac_all[:, _HERO],
        "long_dash_ready": long_dash_ready_all[:, _HERO],
        "long_dash_frac": long_dash_frac_all[:, _HERO],
        "attack_idle_t": state.ent_attack_idle_t[:, _HERO],
        "invuln": state.ent_invuln_t[:, _HERO] > 0,
        "in_bush": in_bush_all[:, _HERO],
        "in_zone": in_zone_all[:, _HERO],
        "tile": torch.stack(terrain.to_tile(hero_pos), dim=-1),
        "damage_dealt": state.ent_damage_dealt[:, _HERO],
        "damage_taken": state.ent_damage_taken[:, _HERO],
        "kills": state.ent_kills[:, _HERO],
        "shots_fired": state.ent_shots_fired[:, _HERO],
        "rank": rank_all[:, _HERO],
    }

    # ---- entities ----
    rel_pos = state.ent_pos - hero_pos.unsqueeze(1)
    rel_vel = state.ent_vel - hero_vel.unsqueeze(1)
    dir_outward = geo.normalize(rel_pos)
    closing_speed = -(rel_vel * dir_outward).sum(dim=-1)
    bearing = geo.angle_diff(geo.angle_of(rel_pos), hero_facing.unsqueeze(1))
    order = torch.argsort(hero_dist, dim=-1, stable=True)
    dist_rank = torch.argsort(order, dim=-1, stable=True)

    entities_obs = {
        "alive": state.ent_alive,
        "kind": state.ent_kind,
        "kind_onehot": torch.nn.functional.one_hot(state.ent_kind, N_KINDS).to(torch.uint8),
        "team": state.ent_kind.clone(),  # FFA: inert placeholder, see module docstring
        "pos": state.ent_pos,
        "pos_norm": state.ent_pos / map_wh,
        "vel": state.ent_vel,
        "speed": geo.safe_norm(state.ent_vel, dim=-1),
        "facing": state.ent_facing,
        "hp": state.ent_hp,
        "max_hp": state.ent_max_hp,
        "hp_frac": state.ent_hp / torch.clamp(state.ent_max_hp, min=_EPS),
        "cubes": state.ent_cubes,
        "ammo": state.ent_ammo,
        "ammo_frac": state.ent_ammo / torch.clamp(max_ammo_all, min=_EPS),
        "attack_cd": state.ent_attack_cd,
        "can_attack": can_attack_all,
        "dashing": state.ent_dash_t > 0,
        "dash_t": state.ent_dash_t,
        "dash_dir": state.ent_dash_dir,
        "invuln": state.ent_invuln_t > 0,
        "in_bush": in_bush_all,
        "in_zone": in_zone_all,
        "tile": torch.stack(terrain.to_tile(state.ent_pos), dim=-1),
        "rel_pos": rel_pos,
        "dist": hero_dist,
        "bearing": bearing,
        "rel_vel": rel_vel,
        "closing_speed": closing_speed,
        "in_view": _in_window(state.ent_pos, view_origin, cfg.view_h, cfg.view_w),
        "dist_rank": dist_rank,
        "revealed_to_hero": revealed_to_hero,
        "hero_revealed_to": hero_revealed_to,
        "los_from_hero": los_from_hero,
        "hidden_by_bush": hidden_by_bush,
        "death_step": state.ent_death_step,
        "death_cause": state.ent_death_cause,
        "damage_dealt": state.ent_damage_dealt,
        "damage_taken": state.ent_damage_taken,
        "kills": state.ent_kills,
        "last_hit_by": state.ent_last_hit_by,
    }
    if cfg.obs_include_privileged:
        decision_period = torch.clamp(stats.gather_kind(params.decision_period, state.ent_kind), min=1)
        entity_idx = torch.arange(E, device=device, dtype=torch.int64).view(1, E)
        decision_phase = (state.step_count.unsqueeze(-1).to(torch.int64) + entity_idx) % decision_period
        entities_obs["privileged"] = {
            "target_id": state.ent_target,
            "react_t": state.ent_react_t,
            "reveal_t": state.ent_reveal_t,
            "decision_phase": decision_phase,
            "move_intent": state.ent_move_smooth,
        }
    obs["entities"] = entities_obs

    # ---- projectiles ----
    prj_rel = state.prj_pos - hero_pos.unsqueeze(1)
    owner_kind = torch.gather(state.ent_kind, 1, torch.clamp(state.prj_owner, min=0, max=E - 1))
    hero_pos_p = hero_pos.unsqueeze(1).expand(-1, P, -1)
    t_closest, d_closest = geo.closest_approach(state.prj_pos, state.prj_vel, hero_pos_p)
    hit_radius = params.unit_radius.view(-1, 1) + state.prj_radius

    obs["projectiles"] = {
        "alive": state.prj_alive,
        "kind": state.prj_kind,
        "kind_onehot": torch.nn.functional.one_hot(state.prj_kind, N_PROJ_KINDS).to(torch.uint8),
        "pos": state.prj_pos,
        "pos_norm": state.prj_pos / map_wh,
        "vel": state.prj_vel,
        "speed": geo.safe_norm(state.prj_vel, dim=-1),
        "heading": geo.angle_of(state.prj_vel),
        "owner": state.prj_owner,
        "owner_kind": owner_kind,
        "damage": state.prj_damage,
        "radius": state.prj_radius,
        "aoe": state.prj_aoe,
        # D13: the agent must be able to tell a stationary 696-per-tick puddle from an incoming
        # bullet. `lobbed` (a bool) could not express a third class, so it became `class`.
        "class": state.prj_class,
        "class_onehot": torch.nn.functional.one_hot(state.prj_class, N_PROJ_CLASSES).to(torch.uint8),
        "lobbed": state.prj_class == _ARTILLERY,
        "dist_left": state.prj_dist_left,
        "age": state.prj_age,
        "rel_pos": prj_rel,
        "dist": geo.safe_norm(prj_rel, dim=-1),
        "time_to_closest": t_closest,
        "closest_dist": d_closest,
        "threatens_hero": state.prj_alive & (state.prj_owner != _HERO) & (d_closest <= hit_radius),
        "in_view": _in_window(state.prj_pos, view_origin, cfg.view_h, cfg.view_w),
    }

    # ---- boxes ----
    box_rel = state.box_pos - hero_pos.unsqueeze(1)
    obs["boxes"] = {
        "alive": state.box_alive,
        "pos": state.box_pos,
        "pos_norm": state.box_pos / map_wh,
        "hp": state.box_hp,
        "max_hp": state.box_max_hp,
        "hp_frac": state.box_hp / torch.clamp(state.box_max_hp, min=_EPS),
        "rel_pos": box_rel,
        "dist": geo.safe_norm(box_rel, dim=-1),
        "in_view": _in_window(state.box_pos, view_origin, cfg.view_h, cfg.view_w),
    }

    # ---- pickups ----
    pku_rel = state.pku_pos - hero_pos.unsqueeze(1)
    obs["pickups"] = {
        "alive": state.pku_alive,
        "pos": state.pku_pos,
        "pos_norm": state.pku_pos / map_wh,
        "cubes": state.pku_cubes,
        "age": state.pku_age,
        "rel_pos": pku_rel,
        "dist": geo.safe_norm(pku_rel, dim=-1),
        "in_view": _in_window(state.pku_pos, view_origin, cfg.view_h, cfg.view_w),
    }

    # ---- zone ----
    map_area = float(cfg.map_w * cfg.map_h)
    zone_size = state.zone_hi - state.zone_lo
    obs["zone"] = {
        "lo": state.zone_lo,
        "hi": state.zone_hi,
        "lo_norm": state.zone_lo / map_wh,
        "hi_norm": state.zone_hi / map_wh,
        "active": torch.full((N,), cfg.zone_enabled, dtype=torch.bool, device=device),
        "step": state.zone_step,
        "dps": zone_mod.current_dps(state, params),
        "next_shrink_in": torch.clamp(state.zone_next_t - state.time, min=0.0),
        "hero_margin": torch.stack([
            hero_pos[:, 0] - state.zone_lo[:, 0],
            state.zone_hi[:, 0] - hero_pos[:, 0],
            hero_pos[:, 1] - state.zone_lo[:, 1],
            state.zone_hi[:, 1] - hero_pos[:, 1],
        ], dim=-1),
        "safe_area_frac": (zone_size[:, 0] * zone_size[:, 1]) / map_area,
    }

    # ---- visibility ----
    obs["visibility"] = {"vis": vis, "los": raw_los, "dist_matrix": dist_matrix}

    # ---- grids ----
    obs["view"] = _build_grid(state, bank, vis, cfg, view_origin, cfg.view_h, cfg.view_w)
    if cfg.obs_include_world_grid:
        world_origin = torch.zeros((N, 2), dtype=torch.int64, device=device)
        obs["world"] = _build_grid(state, bank, vis, cfg, world_origin, cfg.map_h, cfg.map_w)

    # ---- action mask / meta ----
    obs["action_mask"] = hero_mod.action_mask(state, params, cfg)

    const_i64 = lambda v: torch.full((N,), v, dtype=torch.int64, device=device)  # noqa: E731
    obs["meta"] = {
        "map_id": state.map_id,
        "time": state.time,
        "step_count": state.step_count,
        "time_frac": state.step_count.to(torch.float32) / float(cfg.max_episode_steps),
        "n_alive": state.n_alive,
        "n_enemies_alive": state.n_alive - state.ent_alive[:, _HERO].to(state.n_alive.dtype),
        "episode_step_limit": const_i64(cfg.max_episode_steps),
        "map_h": const_i64(cfg.map_h),
        "map_w": const_i64(cfg.map_w),
        "view_h": const_i64(cfg.view_h),
        "view_w": const_i64(cfg.view_w),
    }

    return obs
