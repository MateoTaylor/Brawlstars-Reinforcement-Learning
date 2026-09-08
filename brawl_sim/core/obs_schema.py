"""OBS_SCHEMA: the single source of truth for every field build_obs (Step 25) can produce --
shape (symbolic dims, resolved per-cfg by obs_spec), dtype, units, range, description, and
whether it's gated under entities.privileged or a cfg toggle. See BRAWL_SIM_BUILD_PLAN.md
Step 26.

Kept as one declarative table (`_ROWS`, one row per leaf field) rather than ~130 hand-written
dict literals -- the same "declarative table + a small builder" shape config.py's
PER_KIND_FIELDS/PER_ENV_FIELDS and state.py's _ENTITY_FIELDS already use for a long,
mechanically similar list that has to stay in sync with real code. `_ROWS`' order matches
`core/observation.py`'s own dict-construction order exactly (hero, entities [+ privileged],
projectiles, boxes, pickups, zone, visibility, view, world, action_mask, meta) -- describe_obs
and dump_obs_schema.py both render top-to-bottom in this order for free, no separate sort key.

Shape dims are either "N" (batch size -- left unresolved even by obs_spec, since n_envs isn't
part of EnvConfig, it's a separate argument to env construction), a plain int (fixed regardless
of cfg), or one of the symbols in _DIM_RESOLVERS (resolved against a real EnvConfig by
obs_spec). `range` is `None` wherever a field's bound is either unbounded or cfg-dependent
(e.g. position, bounded by map_w/map_h but not by a fixed constant) -- it's best-effort
documentation, not something validate_obs enforces.
"""
from dataclasses import dataclass

import torch

_TORCH_DTYPE = {
    "float32": torch.float32,
    "bool": torch.bool,
    "int64": torch.int64,
    "int32": torch.int32,
    "uint8": torch.uint8,
}

_DIM_RESOLVERS = {
    "E": lambda cfg: cfg.n_entities,
    "P": lambda cfg: cfg.max_projectiles,
    "B": lambda cfg: cfg.max_boxes,
    "U": lambda cfg: cfg.max_pickups,
    "H": lambda cfg: cfg.map_h,
    "W": lambda cfg: cfg.map_w,
    "VH": lambda cfg: cfg.view_h,
    "VW": lambda cfg: cfg.view_w,
    "MOVE": lambda cfg: cfg.n_move_bins + 1,
}

_PI = 3.141592653589793


@dataclass(frozen=True)
class FieldSpec:
    shape: tuple
    dtype: str
    units: str
    range: tuple | None
    description: str
    privileged: bool = False
    conditional: str | None = None  # EnvConfig bool attribute name gating presence; None = always


# name, shape, dtype, units, range, description, privileged, conditional
_ROWS: tuple[tuple, ...] = (
    # ---- hero: mirror of entities[0] -----------------------------------------------
    ("hero.pos", ("N", 2), "float32", "tiles", None, "Hero world position (x, y).", False, None),
    ("hero.pos_norm", ("N", 2), "float32", "fraction", (0.0, 1.0), "Hero position / (map_w, map_h).", False, None),
    ("hero.vel", ("N", 2), "float32", "tiles/s", None, "Hero velocity.", False, None),
    ("hero.facing", ("N",), "float32", "radians", (-_PI, _PI), "Hero facing angle, atan2 convention.", False, None),
    ("hero.facing_vec", ("N", 2), "float32", "unitless", (-1.0, 1.0), "Unit vector for hero.facing.", False, None),
    ("hero.hp", ("N",), "float32", "hp", (0.0, None), "Hero current HP.", False, None),
    ("hero.max_hp", ("N",), "float32", "hp", (0.0, None), "Hero max HP (cube-scaled).", False, None),
    ("hero.hp_frac", ("N",), "float32", "fraction", (0.0, 1.0), "hp / max_hp.", False, None),
    ("hero.alive", ("N",), "bool", "bool", None, "Whether the hero is alive.", False, None),
    ("hero.cubes", ("N",), "int64", "count", (0, None), "Cubes held.", False, None),
    ("hero.ammo", ("N",), "float32", "count", (0.0, None), "Ammo (fractional, reloads continuously).", False, None),
    ("hero.ammo_frac", ("N",), "float32", "fraction", (0.0, 1.0), "ammo / max_ammo.", False, None),
    ("hero.ammo_whole", ("N",), "int64", "count", (0, None), "floor(ammo) -- whole shots available.", False, None),
    ("hero.attack_cd", ("N",), "float32", "seconds", (0.0, None), "Seconds left on the attack cooldown.", False, None),
    ("hero.can_attack", ("N",), "bool", "bool", None, "alive & ammo>=1 & attack_cd<=0 & dash_t<=0.", False, None),
    ("hero.dashing", ("N",), "bool", "bool", None, "dash_t > 0.", False, None),
    ("hero.dash_t", ("N",), "float32", "seconds", (0.0, None), "Seconds left in the current dash.", False, None),
    ("hero.dash_dir", ("N", 2), "float32", "unitless", None, "Current dash direction (0 if not dashing).", False, None),
    ("hero.super_ready", ("N",), "bool", "bool", None, "Super is charged and legal to fire this tick.", False, None),
    ("hero.super_charge", ("N",), "int32", "count", (0, None), "Hits landed on players since the super was last fired.", False, None),
    ("hero.super_charge_frac", ("N",), "float32", "fraction", (0.0, 1.0), "super_charge / super_charge_hits; 1.0 = ready.", False, None),
    ("hero.long_dash_ready", ("N",), "bool", "bool", None, "Next dash reaches long_dash_multiplier x its normal distance.", False, None),
    ("hero.long_dash_frac", ("N",), "float32", "fraction", (0.0, 1.0), "Progress toward the long dash; 1.0 = ready.", False, None),
    ("hero.attack_idle_t", ("N",), "float32", "seconds", (0.0, None), "Seconds since the hero last attacked (resets on attack only, not on damage).", False, None),
    ("hero.invuln", ("N",), "bool", "bool", None, "invuln_t > 0 (dash i-frames).", False, None),
    ("hero.in_bush", ("N",), "bool", "bool", None, "Standing on a BUSH tile.", False, None),
    ("hero.in_zone", ("N",), "bool", "bool", None, "Outside the safe rect (in the damaging area).", False, None),
    ("hero.tile", ("N", 2), "int64", "tiles", None, "floor(pos) tile index (col, row).", False, None),
    ("hero.damage_dealt", ("N",), "float32", "hp", (0.0, None), "Cumulative damage dealt this episode.", False, None),
    ("hero.damage_taken", ("N",), "float32", "hp", (0.0, None), "Cumulative damage taken this episode.", False, None),
    ("hero.kills", ("N",), "int32", "count", (0, None), "Kills this episode.", False, None),
    ("hero.shots_fired", ("N",), "int32", "count", (0, None), "Shots/dashes fired this episode.", False, None),
    ("hero.rank", ("N",), "int64", "index", (1, None), "1 = best placement so far; see obs_schema module docstring.", False, None),

    # ---- entities: all E, index-stable, never sorted/masked ------------------------
    ("entities.alive", ("N", "E"), "bool", "bool", None, "Whether this entity is alive.", False, None),
    ("entities.kind", ("N", "E"), "int64", "enum", (0, 7), "brawl_sim.constants.Kind value.", False, None),
    ("entities.kind_onehot", ("N", "E", 8), "uint8", "onehot", (0, 1), "One-hot of kind, N_KINDS=8 wide.", False, None),
    ("entities.team", ("N", "E"), "int64", "enum", None, "== kind; FFA placeholder, see observation.py docstring.", False, None),
    ("entities.pos", ("N", "E", 2), "float32", "tiles", None, "World position.", False, None),
    ("entities.pos_norm", ("N", "E", 2), "float32", "fraction", (0.0, 1.0), "pos / (map_w, map_h).", False, None),
    ("entities.vel", ("N", "E", 2), "float32", "tiles/s", None, "Velocity.", False, None),
    ("entities.speed", ("N", "E"), "float32", "tiles/s", (0.0, None), "norm(vel).", False, None),
    ("entities.facing", ("N", "E"), "float32", "radians", (-_PI, _PI), "Facing angle.", False, None),
    ("entities.hp", ("N", "E"), "float32", "hp", (0.0, None), "Current HP.", False, None),
    ("entities.max_hp", ("N", "E"), "float32", "hp", (0.0, None), "Max HP (cube-scaled).", False, None),
    ("entities.hp_frac", ("N", "E"), "float32", "fraction", (0.0, 1.0), "hp / max_hp.", False, None),
    ("entities.cubes", ("N", "E"), "int64", "count", (0, None), "Cubes held.", False, None),
    ("entities.ammo", ("N", "E"), "float32", "count", (0.0, None), "Ammo (fractional).", False, None),
    ("entities.ammo_frac", ("N", "E"), "float32", "fraction", (0.0, 1.0), "ammo / max_ammo.", False, None),
    ("entities.attack_cd", ("N", "E"), "float32", "seconds", (0.0, None), "Seconds left on cooldown.", False, None),
    ("entities.can_attack", ("N", "E"), "bool", "bool", None, "alive & ammo>=1 & attack_cd<=0 & dash_t<=0.", False, None),
    ("entities.dashing", ("N", "E"), "bool", "bool", None, "dash_t > 0.", False, None),
    ("entities.dash_t", ("N", "E"), "float32", "seconds", (0.0, None), "Seconds left in the current dash.", False, None),
    ("entities.dash_dir", ("N", "E", 2), "float32", "unitless", None, "Current dash direction.", False, None),
    ("entities.invuln", ("N", "E"), "bool", "bool", None, "invuln_t > 0.", False, None),
    ("entities.in_bush", ("N", "E"), "bool", "bool", None, "Standing on a BUSH tile.", False, None),
    ("entities.in_zone", ("N", "E"), "bool", "bool", None, "Outside the safe rect.", False, None),
    ("entities.tile", ("N", "E", 2), "int64", "tiles", None, "floor(pos) tile index.", False, None),
    ("entities.rel_pos", ("N", "E", 2), "float32", "tiles", None, "pos - hero.pos.", False, None),
    ("entities.dist", ("N", "E"), "float32", "tiles", (0.0, None), "Distance to the hero.", False, None),
    ("entities.bearing", ("N", "E"), "float32", "radians", (-_PI, _PI), "Angle to this entity relative to hero.facing.", False, None),
    ("entities.rel_vel", ("N", "E", 2), "float32", "tiles/s", None, "vel - hero.vel.", False, None),
    ("entities.closing_speed", ("N", "E"), "float32", "tiles/s", None, "Positive = closing in on the hero.", False, None),
    ("entities.in_view", ("N", "E"), "bool", "bool", None, "Inside the egocentric view crop window.", False, None),
    ("entities.dist_rank", ("N", "E"), "int64", "index", (0, None), "0 = nearest to hero; a permutation of 0..E-1.", False, None),
    ("entities.revealed_to_hero", ("N", "E"), "bool", "bool", None, "The hero currently sees this entity.", False, None),
    ("entities.hero_revealed_to", ("N", "E"), "bool", "bool", None, "This entity currently sees the hero.", False, None),
    ("entities.los_from_hero", ("N", "E"), "bool", "bool", None, "Clear physical (wall-only) line of sight from the hero.", False, None),
    ("entities.hidden_by_bush", ("N", "E"), "bool", "bool", None, "In a bush and not revealed to the hero.", False, None),
    ("entities.death_step", ("N", "E"), "int32", "ticks", (-1, None), "step_count at death, or -1 if never died.", False, None),
    ("entities.death_cause", ("N", "E"), "int32", "enum", (0, 2), "brawl_sim.constants.DeathCause value.", False, None),
    ("entities.damage_dealt", ("N", "E"), "float32", "hp", (0.0, None), "Cumulative damage dealt this episode.", False, None),
    ("entities.damage_taken", ("N", "E"), "float32", "hp", (0.0, None), "Cumulative damage taken this episode.", False, None),
    ("entities.kills", ("N", "E"), "int32", "count", (0, None), "Kills this episode.", False, None),
    ("entities.last_hit_by", ("N", "E"), "int64", "index", (-1, None), "Entity index that last damaged this one, or -1.", False, None),

    # ---- entities.privileged: bot internals, namespaced away from the agent --------
    ("entities.privileged.target_id", ("N", "E"), "int64", "index", (-1, None), "Bot's current AI target, or -1.", True, "obs_include_privileged"),
    ("entities.privileged.react_t", ("N", "E"), "float32", "seconds", (0.0, None), "Reaction-delay countdown before this bot may fire again.", True, "obs_include_privileged"),
    ("entities.privileged.reveal_t", ("N", "E"), "float32", "seconds", (0.0, None), "Seconds left forced-revealed after this entity last attacked.", True, "obs_include_privileged"),
    ("entities.privileged.decision_phase", ("N", "E"), "int64", "ticks", (0, None), "(step_count + slot) % decision_period; 0 = this is a fire-reconsideration tick.", True, "obs_include_privileged"),
    ("entities.privileged.move_intent", ("N", "E", 2), "float32", "unitless", None, "Bot's smoothed (post-reaction-delay) movement intent.", True, "obs_include_privileged"),

    # ---- projectiles: all P slots ---------------------------------------------------
    ("projectiles.alive", ("N", "P"), "bool", "bool", None, "Whether this slot holds a live projectile.", False, None),
    ("projectiles.kind", ("N", "P"), "int64", "enum", (0, 6), "brawl_sim.constants.Proj value.", False, None),
    ("projectiles.kind_onehot", ("N", "P", 7), "uint8", "onehot", (0, 1), "One-hot of kind, N_PROJ_KINDS=7 wide.", False, None),
    ("projectiles.pos", ("N", "P", 2), "float32", "tiles", None, "World position.", False, None),
    ("projectiles.pos_norm", ("N", "P", 2), "float32", "fraction", (0.0, 1.0), "pos / (map_w, map_h).", False, None),
    ("projectiles.vel", ("N", "P", 2), "float32", "tiles/s", None, "Velocity.", False, None),
    ("projectiles.speed", ("N", "P"), "float32", "tiles/s", (0.0, None), "norm(vel).", False, None),
    ("projectiles.heading", ("N", "P"), "float32", "radians", (-_PI, _PI), "atan2 of vel.", False, None),
    ("projectiles.owner", ("N", "P"), "int64", "index", (0, None), "Entity index that fired this projectile.", False, None),
    ("projectiles.owner_kind", ("N", "P"), "int64", "enum", (0, 7), "Kind of the owning entity.", False, None),
    ("projectiles.damage", ("N", "P"), "float32", "hp", (0.0, None), "Damage on hit.", False, None),
    ("projectiles.radius", ("N", "P"), "float32", "tiles", (0.0, None), "Collision radius.", False, None),
    ("projectiles.aoe", ("N", "P"), "float32", "tiles", (0.0, None), "AoE radius (lobbed projectiles only; 0 otherwise).", False, None),
    ("projectiles.class", ("N", "P"), "int64", "enum", (0, 2), "brawl_sim.constants.ProjClass: how it moves and damages.", False, None),
    ("projectiles.class_onehot", ("N", "P", 3), "uint8", "onehot", (0, 1), "One-hot of class, N_PROJ_CLASSES=3 wide (PROJECTILE/ARTILLERY/HAZARD).", False, None),
    ("projectiles.lobbed", ("N", "P"), "bool", "bool", None, "class == ARTILLERY; kept as a convenience alias.", False, None),
    ("projectiles.dist_left", ("N", "P"), "float32", "tiles", None, "Remaining travel distance before expiry.", False, None),
    ("projectiles.age", ("N", "P"), "float32", "seconds", (0.0, None), "Time since this projectile was spawned.", False, None),
    ("projectiles.rel_pos", ("N", "P", 2), "float32", "tiles", None, "pos - hero.pos.", False, None),
    ("projectiles.dist", ("N", "P"), "float32", "tiles", (0.0, None), "Distance to the hero.", False, None),
    ("projectiles.time_to_closest", ("N", "P"), "float32", "seconds", (0.0, None), "Time to closest future approach to the hero.", False, None),
    ("projectiles.closest_dist", ("N", "P"), "float32", "tiles", (0.0, None), "Distance at that closest approach.", False, None),
    ("projectiles.threatens_hero", ("N", "P"), "bool", "bool", None, "Alive, not hero-owned, and on a path that hits the hero.", False, None),
    ("projectiles.in_view", ("N", "P"), "bool", "bool", None, "Inside the egocentric view crop window.", False, None),

    # ---- boxes: all B slots ----------------------------------------------------------
    ("boxes.alive", ("N", "B"), "bool", "bool", None, "Whether this box is still standing.", False, None),
    ("boxes.pos", ("N", "B", 2), "float32", "tiles", None, "World position (static after spawn).", False, None),
    ("boxes.pos_norm", ("N", "B", 2), "float32", "fraction", (0.0, 1.0), "pos / (map_w, map_h).", False, None),
    ("boxes.hp", ("N", "B"), "float32", "hp", (0.0, None), "Current HP.", False, None),
    ("boxes.max_hp", ("N", "B"), "float32", "hp", (0.0, None), "Max HP.", False, None),
    ("boxes.hp_frac", ("N", "B"), "float32", "fraction", (0.0, 1.0), "hp / max_hp.", False, None),
    ("boxes.rel_pos", ("N", "B", 2), "float32", "tiles", None, "pos - hero.pos.", False, None),
    ("boxes.dist", ("N", "B"), "float32", "tiles", (0.0, None), "Distance to the hero.", False, None),
    ("boxes.in_view", ("N", "B"), "bool", "bool", None, "Inside the egocentric view crop window.", False, None),

    # ---- pickups: all U slots ---------------------------------------------------------
    ("pickups.alive", ("N", "U"), "bool", "bool", None, "Whether this pickup is still on the ground.", False, None),
    ("pickups.pos", ("N", "U", 2), "float32", "tiles", None, "World position.", False, None),
    ("pickups.pos_norm", ("N", "U", 2), "float32", "fraction", (0.0, 1.0), "pos / (map_w, map_h).", False, None),
    ("pickups.cubes", ("N", "U"), "int64", "count", (0, None), "Cubes this pickup holds.", False, None),
    ("pickups.age", ("N", "U"), "float32", "seconds", (0.0, None), "Time since this pickup was dropped.", False, None),
    ("pickups.rel_pos", ("N", "U", 2), "float32", "tiles", None, "pos - hero.pos.", False, None),
    ("pickups.dist", ("N", "U"), "float32", "tiles", (0.0, None), "Distance to the hero.", False, None),
    ("pickups.in_view", ("N", "U"), "bool", "bool", None, "Inside the egocentric view crop window.", False, None),

    # ---- zone --------------------------------------------------------------------------
    ("zone.lo", ("N", 2), "float32", "tiles", None, "Safe rect lower corner.", False, None),
    ("zone.hi", ("N", 2), "float32", "tiles", None, "Safe rect upper corner.", False, None),
    ("zone.lo_norm", ("N", 2), "float32", "fraction", (0.0, 1.0), "lo / (map_w, map_h).", False, None),
    ("zone.hi_norm", ("N", 2), "float32", "fraction", (0.0, 1.0), "hi / (map_w, map_h).", False, None),
    ("zone.active", ("N",), "bool", "bool", None, "cfg.zone_enabled, broadcast per env.", False, None),
    ("zone.step", ("N",), "int32", "count", (0, None), "Number of shrinks so far.", False, None),
    ("zone.dps", ("N",), "float32", "hp/s", (0.0, None), "Damage-per-second the HERO takes outside the rect (the zone's rate is a fraction of max HP, so it differs per entity).", False, None),
    ("zone.next_shrink_in", ("N",), "float32", "seconds", (0.0, None), "Seconds until the next shrink (0 if disabled/overdue).", False, None),
    ("zone.hero_margin", ("N", 4), "float32", "tiles", None, "(x-lo.x, hi.x-x, y-lo.y, hi.y-y); negative = outside on that side.", False, None),
    ("zone.hero_margin_local", ("N", 4), "float32", "tiles", None, "hero_margin clamped to +/- cfg.zone_margin_horizon_tiles -- the same four distances as a bounded sensor sees them, which is what brawl_deployment can supply from observed gas. Deploy specs take this; full-information specs take hero_margin.", False, None),
    ("zone.safe_area_frac", ("N",), "float32", "fraction", (0.0, 1.0), "Safe rect area / map area.", False, None),

    # ---- visibility ----------------------------------------------------------------------
    ("visibility.vis", ("N", "E", "E"), "bool", "bool", None, "vis[i,j]: i sees j (bush-aware targeting visibility).", False, None),
    ("visibility.los", ("N", "E", "E"), "bool", "bool", None, "los[i,j]: clear physical (wall-only) line of sight i -> j.", False, None),
    ("visibility.dist_matrix", ("N", "E", "E"), "float32", "tiles", (0.0, None), "Pairwise entity distances.", False, None),

    # ---- grids -----------------------------------------------------------------------------
    ("view", ("N", 12, "VH", "VW"), "uint8", "count", (0, 255), "Egocentric 12-channel occupancy/terrain grid, hero-centered.", False, None),
    ("world", ("N", 12, "H", "W"), "uint8", "count", (0, 255), "Full-map 12-channel occupancy/terrain grid.", False, "obs_include_world_grid"),

    # ---- action_mask -----------------------------------------------------------------------
    ("action_mask.move", ("N", "MOVE"), "bool", "bool", None, "Legal move bins (idle + n_move_bins directions); always all-True today.", False, None),
    ("action_mask.attack", ("N", 3), "bool", "bool", None, "[no-fire, attack, super] legal for the hero this tick.", False, None),

    # ---- meta ------------------------------------------------------------------------------
    ("meta.map_id", ("N",), "int64", "index", (0, None), "Index into cfg.map_names for this env.", False, None),
    ("meta.time", ("N",), "float32", "seconds", (0.0, None), "Elapsed episode time.", False, None),
    ("meta.step_count", ("N",), "int32", "ticks", (0, None), "Elapsed episode ticks.", False, None),
    ("meta.time_frac", ("N",), "float32", "fraction", (0.0, None), "step_count / max_episode_steps.", False, None),
    ("meta.n_alive", ("N",), "int32", "count", (0, None), "Entities alive (including the hero).", False, None),
    ("meta.n_enemies_alive", ("N",), "int32", "count", (0, None), "Non-hero entities alive.", False, None),
    ("meta.episode_step_limit", ("N",), "int64", "ticks", (0, None), "cfg.max_episode_steps, broadcast per env.", False, None),
    ("meta.map_h", ("N",), "int64", "tiles", (0, None), "cfg.map_h, broadcast per env.", False, None),
    ("meta.map_w", ("N",), "int64", "tiles", (0, None), "cfg.map_w, broadcast per env.", False, None),
    ("meta.view_h", ("N",), "int64", "tiles", (0, None), "cfg.view_h, broadcast per env.", False, None),
    ("meta.view_w", ("N",), "int64", "tiles", (0, None), "cfg.view_w, broadcast per env.", False, None),
)

OBS_SCHEMA: dict = {
    name: FieldSpec(shape, dtype, units, rng, desc, privileged, conditional)
    for name, shape, dtype, units, rng, desc, privileged, conditional in _ROWS
}


def _resolve_shape(shape: tuple, cfg) -> tuple:
    return tuple(d if d == "N" or isinstance(d, int) else _DIM_RESOLVERS[d](cfg) for d in shape)


def obs_spec(cfg) -> dict:
    """OBS_SCHEMA with every symbolic dim resolved against `cfg`, and any field whose
    `conditional` cfg attribute is falsy dropped entirely -- the dict this specific cfg's
    build_obs output should structurally match. "N" is left unresolved (see module
    docstring)."""
    out = {}
    for name, spec in OBS_SCHEMA.items():
        if spec.conditional is not None and not getattr(cfg, spec.conditional):
            continue
        out[name] = FieldSpec(
            _resolve_shape(spec.shape, cfg), spec.dtype, spec.units, spec.range,
            spec.description, spec.privileged, spec.conditional,
        )
    return out


def _flatten(node, prefix: str = "") -> dict:
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            sub = f"{prefix}.{key}" if prefix else key
            out.update(_flatten(value, sub))
        return out
    return {prefix: node}


def validate_obs(obs: dict, cfg) -> None:
    """Raises ValueError with the specific field(s) and reason on any mismatch: missing/extra
    fields (relative to obs_spec(cfg)), wrong shape, wrong dtype, or non-finite floats. Passes
    silently on a well-formed observation. Not sync-free (does host-transferring .item()-free
    checks only, but raises via Python control flow) -- a validation/debug utility, not part of
    the hot path."""
    spec = obs_spec(cfg)
    flat = _flatten(obs)

    expected_names = set(spec.keys())
    actual_names = set(flat.keys())
    if expected_names != actual_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"obs does not match obs_spec(cfg) -- missing: {missing}, extra: {extra}")

    for name, field in spec.items():
        value = flat[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name}: expected a Tensor, got {type(value).__name__}")

        n = value.shape[0] if value.dim() > 0 else None
        expected_shape = tuple(n if d == "N" else d for d in field.shape)
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{name}: expected shape {expected_shape}, got {tuple(value.shape)}")

        expected_dtype = _TORCH_DTYPE[field.dtype]
        if value.dtype != expected_dtype:
            raise ValueError(f"{name}: expected dtype {expected_dtype}, got {value.dtype}")

        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{name}: contains NaN/Inf")


def _fmt_scalar(v: torch.Tensor) -> str:
    if v.dtype == torch.bool:
        return "T" if bool(v) else "F"
    if v.is_floating_point():
        return f"{v.item():.3g}"
    return str(v.item())


def _fmt_vec(v: torch.Tensor) -> str:
    return "(" + ", ".join(f"{x:.3g}" for x in v.tolist()) + ")"


_HERO_DESCRIBE_FIELDS = (
    "pos", "hp_frac", "ammo", "cubes", "alive", "in_bush", "in_zone",
    "dashing", "can_attack", "invuln", "rank",
)
_MAX_ROWS_PER_GROUP = 8


def describe_obs(obs: dict, env_index: int = 0) -> str:
    """Pretty-prints ONE env's observation for a human, with names/values/units -- a debug/
    logging utility, like state.snapshot and to_numpy below: never call this from the hot path
    (it does plenty of host-transferring .item()/.tolist() calls by design). Summarizes rather
    than dumping the big per-slot/grid/matrix fields in full, so a debug_tiny observation fits
    on one screen."""
    meta = obs["meta"]
    lines = [
        f"=== env {env_index}  map={int(meta['map_id'][env_index])}  "
        f"t={meta['time'][env_index].item():.2f}s  step={int(meta['step_count'][env_index])}  "
        f"n_alive={int(meta['n_alive'][env_index])} ==="
    ]

    hero = obs["hero"]
    lines.append("-- hero --")
    for name in _HERO_DESCRIBE_FIELDS:
        v = hero[name][env_index]
        lines.append(f"  {name}: {_fmt_vec(v) if v.dim() > 0 else _fmt_scalar(v)}")

    ent = obs["entities"]
    E = ent["alive"].shape[1]
    lines.append(f"-- entities (E={E}) --")
    for e in range(E):
        if not bool(ent["alive"][env_index, e]):
            lines.append(f"  [{e}] dead (death_step={int(ent['death_step'][env_index, e])})")
            continue
        lines.append(
            f"  [{e}] kind={int(ent['kind'][env_index, e])} pos={_fmt_vec(ent['pos'][env_index, e])} "
            f"hp_frac={ent['hp_frac'][env_index, e].item():.2f} dist={ent['dist'][env_index, e].item():.1f} "
            f"revealed={bool(ent['revealed_to_hero'][env_index, e])}"
        )

    for group_name in ("projectiles", "boxes", "pickups"):
        group = obs[group_name]
        alive = group["alive"][env_index]
        idx = torch.nonzero(alive, as_tuple=False).squeeze(-1).tolist()
        lines.append(f"-- {group_name} ({len(idx)}/{alive.shape[0]} alive) --")
        for i in idx[:_MAX_ROWS_PER_GROUP]:
            lines.append(f"  [{i}] pos={_fmt_vec(group['pos'][env_index, i])} dist={group['dist'][env_index, i].item():.1f}")
        if len(idx) > _MAX_ROWS_PER_GROUP:
            lines.append(f"  ... and {len(idx) - _MAX_ROWS_PER_GROUP} more")

    zone = obs["zone"]
    lines.append(
        f"-- zone -- active={bool(zone['active'][env_index])} step={int(zone['step'][env_index])} "
        f"dps={zone['dps'][env_index].item():.0f} safe_frac={zone['safe_area_frac'][env_index].item():.2f}"
    )

    vis = obs["visibility"]["vis"][env_index]
    n_seen = max(int(vis[0].sum()) - 1, 0)  # exclude self
    lines.append(f"-- visibility -- hero sees {n_seen} other entities")

    if "view" in obs:
        v = obs["view"][env_index]
        lines.append(f"-- view -- shape={tuple(v.shape)} wall_frac={(v[0] > 0).float().mean().item():.2f}")
    if "world" in obs:
        lines.append(f"-- world -- shape={tuple(obs['world'][env_index].shape)}")

    am = obs["action_mask"]
    lines.append(
        f"-- action_mask -- move_legal={int(am['move'][env_index].sum())}/{am['move'].shape[1]} "
        f"attack={am['attack'][env_index].tolist()}"
    )

    return "\n".join(lines)


def to_numpy(obs):
    """Recursively converts every tensor leaf to a detached CPU numpy array, preserving
    structure. Host-transfer debug/logging utility ONLY -- like state.snapshot, never call
    this from the sim's hot path."""
    if isinstance(obs, dict):
        return {k: to_numpy(v) for k, v in obs.items()}
    return obs.detach().cpu().numpy()
