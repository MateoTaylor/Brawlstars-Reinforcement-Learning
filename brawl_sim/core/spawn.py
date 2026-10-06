"""Batched partial reset: map/kind/personality/position sampling and reset_envs, the function
that ties every subsystem's own reset logic together.

This module imports config.py directly (resample_params, ARCHETYPE_SHORT_NAMES). That makes no
cycle: config.py imports only constants and core/projectiles, and nothing on that path imports
config.

zero_(state, reset_mask) runs first and clears every resettable field for masked rows
(projectiles and pickups, the act_buf/act_head latency ring, time/step_count and the episode
counters), so reset_envs touches only what zero_'s blanket 0/False cannot supply: map_id,
ent_kind, ent_person, ent_pos, ent_hp/max_hp/ammo/facing/alive, the -1 sentinels ent_target and
ent_death_step, and ent_hunt_t; then it spawns boxes, initializes the zone and sets n_alive.

The samplers compute every row: reset_mask only sources N, and the caller commits the masked
rows with torch.where (resample_params' convention).
"""
import torch

from ..config import ARCHETYPE_SHORT_NAMES, resample_params
from ..constants import AGGRESSIVE_PERSONS, N_PERSONS
from . import boxes, geometry as geo, stats, zone
from .state import zero_


def _const_tensor(values, device, dtype: torch.dtype) -> torch.Tensor:
    """1-D tensor from a plain Python sequence of scalars, NOT `torch.tensor(values,
    device=device)`, which is a host sync under CUDA: geometry.vec2's fix, for any length."""
    return torch.stack([torch.full((), v, device=device, dtype=dtype) for v in values])


def sample_map_ids(reset_mask: torch.Tensor, bank, cfg, gen) -> torch.Tensor:
    """(N,) i64, one map index per env, into cfg.map_names' order (matching MapBank's leading
    dim). Computed fresh for every row (see module docstring). cfg.map_selection: "uniform"
    draws independently per env; "fixed" pins every env to cfg.fixed_map's index."""
    N = reset_mask.shape[0]
    device = reset_mask.device
    M = len(cfg.map_names)

    if cfg.map_selection == "uniform":
        u = torch.rand((N,), generator=gen, device=device)
        return torch.floor(u * M).clamp(max=M - 1).to(torch.int64)
    if cfg.map_selection == "fixed":
        idx = cfg.map_names.index(cfg.fixed_map)  # raises ValueError if fixed_map isn't listed
        return torch.full((N,), idx, dtype=torch.int64, device=device)
    raise ValueError(f"unknown map_selection {cfg.map_selection!r}; expected 'uniform' or 'fixed'")


def sample_enemy_kinds(reset_mask: torch.Tensor, cfg, gen) -> torch.Tensor:
    """(N,E) i64. Slot 0 is always HERO_MORTIS (0). Slots 1..E-1: a multinomial draw over the
    bot archetypes (ARCHETYPE_SHORT_NAMES) weighted by cfg.enemy_type_weights (inverse-CDF +
    searchsorted, the same "sort/rank plus a threshold" shape as projectiles.alloc_slots) when
    cfg.randomize_enemy_types, else cfg.fixed_enemy_types verbatim, tiled across every env.
    Computed fresh for the whole batch (see module docstring)."""
    N = reset_mask.shape[0]
    device = reset_mask.device
    E = cfg.n_entities
    n_enemies = cfg.n_enemies

    kinds = torch.zeros((N, E), dtype=torch.int64, device=device)  # slot 0 stays HERO_MORTIS

    if cfg.randomize_enemy_types:
        weights = _const_tensor(cfg.enemy_type_weights, device, torch.float32)
        cdf = torch.cumsum(weights, dim=0) / weights.sum()
        u = torch.rand((N, n_enemies), generator=gen, device=device)
        archetype = torch.searchsorted(cdf, u, right=True)  # index into ARCHETYPE_SHORT_NAMES
        # +1 because that tuple is BOT kinds only; Kind 0 is the hero and is never drawn here.
        enemy_kinds = torch.clamp(archetype, max=len(ARCHETYPE_SHORT_NAMES) - 1) + 1
    else:
        if len(cfg.fixed_enemy_types) != n_enemies:
            raise ValueError(
                f"fixed_enemy_types has {len(cfg.fixed_enemy_types)} entries, expected "
                f"{n_enemies} (cfg.n_enemies) since randomize_enemy_types is False"
            )
        fixed = [1 + ARCHETYPE_SHORT_NAMES.index(name) for name in cfg.fixed_enemy_types]
        enemy_kinds = _const_tensor(fixed, device, torch.int64).unsqueeze(0).expand(N, -1)

    kinds[:, 1:] = enemy_kinds
    return kinds


def sample_personalities(reset_mask: torch.Tensor, cfg, gen) -> torch.Tensor:
    """(N,E) i64 constants.Person, one per entity slot. Slot 0 is the hero's and is set to RUSH
    (0) purely so the field holds a valid enum -- nothing ever reads it, the same way slot 0's
    ent_target and ent_move_smooth are never read. Slots 1..E-1 draw independently from
    cfg.bots_personality_weights by inverse-CDF + searchsorted, the same shape
    sample_enemy_kinds uses for archetypes. Personality and archetype are drawn INDEPENDENTLY:
    any archetype can come up with any personality. Computed fresh for the whole batch.

    **The aggression floor.** After the weighted draw, an env with fewer than
    `cfg.bots_min_aggressive` bots holding an AGGRESSIVE_PERSONS role has the shortfall forced:
    a lobby of campers and trappers is one the agent wins by standing still. The forced slots are
    chosen by random rank (already-aggressive slots sort last, so no conversion is wasted), NOT
    from slot 1 up: entity slot order is stable for an episode and can appear in the
    observation, so "slot 1 is the dangerous one" would be a pattern the agent could read instead
    of learning to fight.
    """
    N = reset_mask.shape[0]
    device = reset_mask.device
    n_enemies = cfg.n_enemies

    persons = torch.zeros((N, cfg.n_entities), dtype=torch.int64, device=device)
    if not cfg.bots_personalities:
        return persons  # personalities off: every bot is a RUSH

    weights = _const_tensor(cfg.bots_personality_weights, device, torch.float32)
    cdf = torch.cumsum(weights, dim=0) / weights.sum()
    u = torch.rand((N, n_enemies), generator=gen, device=device)
    drawn = torch.clamp(torch.searchsorted(cdf, u, right=True), max=N_PERSONS - 1)

    aggressive = _const_tensor(AGGRESSIVE_PERSONS, device, torch.int64)  # (A,)
    is_aggressive = torch.zeros_like(drawn, dtype=torch.bool)
    for person in AGGRESSIVE_PERSONS:  # 2 compile-time constants, CONVENTIONS.md-sanctioned
        is_aggressive = is_aggressive | (drawn == person)

    n_aggressive = is_aggressive.sum(dim=1, keepdim=True)  # (N,1)
    deficit = torch.clamp(cfg.bots_min_aggressive - n_aggressive, min=0)
    # +2.0 pushes already-aggressive slots past every non-aggressive key (both are in [0,1)), so
    # the `deficit` lowest ranks are exactly the non-aggressive slots. config.validate guarantees
    # min_aggressive <= n_enemies, hence deficit <= the number of non-aggressive slots, so this
    # never has to convert a slot that was already aggressive.
    key = torch.rand((N, n_enemies), generator=gen, device=device) + is_aggressive.to(torch.float32) * 2.0
    rank = torch.argsort(torch.argsort(key, dim=1), dim=1)
    force = rank < deficit

    pick = torch.rand((N, n_enemies), generator=gen, device=device) * len(AGGRESSIVE_PERSONS)
    pick_idx = torch.clamp(pick.to(torch.int64), max=len(AGGRESSIVE_PERSONS) - 1)
    drawn = torch.where(force, aggressive[pick_idx], drawn)

    persons[:, 1:] = drawn
    return persons


def sample_spawn_positions(reset_mask: torch.Tensor, map_ids: torch.Tensor, bank, cfg, gen) -> torch.Tensor:
    """(N,E,2) f32 tile-center positions, one per entity slot (slot 0 = hero, same rule as
    everyone else). A single random rotation r in [0, n_spawns) per env, then entity k claims
    spawn slot (r + floor(k*n_spawns/E)) % n_spawns -- floor-partitioning k across n_spawns
    keeps consecutive entities' slot gaps within one of each other, and since bank.spawns is
    angle-sorted (loader.spawn_points), that's an even angular spread. While E <= n_spawns the
    slots are distinct, so positions are collision-free and passable (SPAWN tiles are never in
    TILE_BLOCKS_UNIT) with no checking here. Nothing enforces that bound (the loader requires
    only MIN_SPAWNS, 8): a map with fewer spawns than entities stacks some of them. Computed
    fresh for the whole batch (see module docstring)."""
    N = reset_mask.shape[0]
    device = reset_mask.device
    E = cfg.n_entities

    n_spawns_env = bank.n_spawns[map_ids]  # (N,)
    map_spawns = bank.spawns[map_ids]  # (N, MAX_SPAWNS, 2)

    u = torch.rand((N,), generator=gen, device=device)
    r = torch.floor(u * n_spawns_env.to(torch.float32)).to(torch.int64)  # (N,)

    slots = []
    for k in range(E):
        offset = torch.div(k * n_spawns_env, E, rounding_mode="floor")
        idx_k = (r + offset) % n_spawns_env  # (N,)
        gather_idx = idx_k.view(N, 1, 1).expand(N, 1, 2)
        slots.append(torch.gather(map_spawns, 1, gather_idx).squeeze(1))
    return torch.stack(slots, dim=1)  # (N,E,2)


def reset_envs(state, reset_mask: torch.Tensor, bank, params, cfg, gen, spec, params_hook=None) -> None:
    """MUTATES everything for masked rows (see module docstring for what zero_ covers). Order:
    zero_ -> resample_params -> params_hook -> map_id -> kinds -> personalities -> positions ->
    hp/max_hp/ammo/facing/alive -> target/death_step -> hunt_t -> spawn_boxes -> init_zone ->
    n_alive.

    `params_hook(params, reset_mask)` is an optional hook called immediately AFTER
    resample_params and BEFORE anything reads `params` back: `max_hp`/`max_ammo` below derive
    from `params`, so a later hook would spawn entities whose HP disagrees with their own
    `base_hp`. Contract: mutate `params` in place, torch.where-masked by `reset_mask` over the
    full batch, sync-free, no Python loop over envs. training/curriculum.py uses it for per-env
    difficulty tiers (a weighted discrete mixture, which resample_params' {low, high} range
    syntax cannot express); the simulator itself never sets it."""
    zero_(state, reset_mask)
    resample_params(params, reset_mask, cfg, gen, spec)
    if params_hook is not None:
        params_hook(params, reset_mask)

    map_ids = sample_map_ids(reset_mask, bank, cfg, gen)
    kinds = sample_enemy_kinds(reset_mask, cfg, gen)
    persons = sample_personalities(reset_mask, cfg, gen)
    positions = sample_spawn_positions(reset_mask, map_ids, bank, cfg, gen)

    mask_e = reset_mask.unsqueeze(-1)  # (N,1), broadcasts against (N,E)
    mask_e2 = mask_e.unsqueeze(-1)  # (N,1,1), broadcasts against (N,E,2)

    state.map_id.copy_(torch.where(reset_mask, map_ids, state.map_id))
    state.ent_kind.copy_(torch.where(mask_e, kinds, state.ent_kind))
    state.ent_person.copy_(torch.where(mask_e, persons, state.ent_person))
    state.ent_pos.copy_(torch.where(mask_e2, positions, state.ent_pos))

    zero_cubes = torch.zeros_like(kinds)
    max_hp = stats.effective_max_hp(kinds, zero_cubes, params)  # cubes=0 at spawn (N,E)
    max_ammo = stats.gather_kind(params.max_ammo, kinds)
    map_center = geo.vec2(cfg.map_w / 2.0, cfg.map_h / 2.0, positions.device, positions.dtype)
    facing = geo.angle_of(map_center - positions)  # (N,E)

    state.ent_hp.copy_(torch.where(mask_e, max_hp, state.ent_hp))
    state.ent_max_hp.copy_(torch.where(mask_e, max_hp, state.ent_max_hp))
    state.ent_ammo.copy_(torch.where(mask_e, max_ammo, state.ent_ammo))
    state.ent_facing.copy_(torch.where(mask_e, facing, state.ent_facing))
    state.ent_alive.copy_(torch.where(mask_e, torch.ones_like(state.ent_alive), state.ent_alive))

    # -1 sentinels where zero_'s 0 would lie: ent_target 0 would read as "targeting entity 0"
    # (bots/perception.select_target relies on this reset), and ent_death_step 0 as "died on
    # step 0", since step_count also starts at 0.
    state.ent_target.copy_(torch.where(mask_e, torch.full_like(state.ent_target, -1), state.ent_target))
    state.ent_death_step.copy_(torch.where(mask_e, torch.full_like(state.ent_death_step, -1), state.ent_death_step))

    # ent_hunt_t's 0 would mean "search timeout already expired", so a hunter's first waypoint
    # would count as searched and be skipped. The other personality fields keep zero_'s 0:
    # ent_hunt_seen's 0 is "nowhere visited yet", and bots/personality.advance_wander re-rolls
    # ent_wander_dir/ent_wander_t before anything reads them.
    hunt_t_fresh = torch.full_like(state.ent_hunt_t, cfg.bots_hunt_timeout_seconds)
    state.ent_hunt_t.copy_(torch.where(mask_e, hunt_t_fresh, state.ent_hunt_t))

    boxes.spawn_boxes(state, reset_mask, bank, params, cfg, gen)
    zone.init_zone(state, reset_mask, params, cfg)

    n_alive_fresh = torch.full_like(state.n_alive, cfg.n_entities)
    state.n_alive.copy_(torch.where(reset_mask, n_alive_fresh, state.n_alive))
