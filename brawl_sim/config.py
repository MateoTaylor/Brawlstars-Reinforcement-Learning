"""Config loading and per-env parameter tensors.

EnvConfig holds static Python values (map size, episode length, feature toggles -- anything
that shapes tensors or branches Python control flow). SimParams holds the numeric stats that
can be randomized per env (brawler stats, cube/zone/regen numbers) as leading-(N,) tensors,
built from configs/default.yaml + configs/brawlers.yaml (the "spec") with an optional
configs/randomization.yaml overlay (RandomizationSpec) layered on top via apply_randomization.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from .constants import HUNT_WAYPOINT_BITS, AimModel, Kind, Proj
# No import cycle: core.projectiles imports only torch, ..constants and sibling core modules
# (geometry/stats/terrain), none of which import config.
from .core.projectiles import BOX_RADIUS, MAX_SPLITS

_MISSING = object()
# Distinct from _MISSING: `_dget(d, k, default=_MISSING)` RAISES when the key is absent
# (apply_randomization relies on that), while `_ABSENT` is RETURNED, so load_config can keep the
# EnvConfig default for a field the file omits.
_ABSENT = object()

# Kind 0 is always the hero; this order matches brawl_sim.constants.Kind's values 0..7 and
# is also the order used for the leading dim of every per-kind SimParams tensor's K axis.
KIND_YAML_NAMES = ("hero_mortis", "bot_sniper", "bot_artillery", "bot_melee", "bot_rifle",
                   "bot_edgar", "bot_spike", "bot_bull")
# Bot kinds in Kind-value order (1..7), used for entities.enemy_type_weights / fixed_enemy_types.
# "edgar", "spike" and "bull" are brawler names where the other four are role names -- see
# constants.Kind.
ARCHETYPE_SHORT_NAMES = ("sniper", "artillery", "melee", "rifle", "edgar", "spike", "bull")
# constants.Person order (values 0..4), used for bots.personality_weights. Bot PERSONALITY is
# orthogonal to bot ARCHETYPE: both are drawn independently per entity on every reset.
PERSON_SHORT_NAMES = ("rush", "camper", "hunter", "trapper", "kite")

_KNOWN_MAP_NAMES = (
    "blank", "open", "bushy", "walled", "skull_creek", "feast_or_famine",
    "scorched_stone", "island_invasion",
    # The ten generated maps; seeds and families in brawl_sim/maps/README.md. A new CSV under
    # brawl_sim/maps/csv/ must be listed here before world.maps / fixed_map may name it.
    "broken_wall", "stone_fort", "twin_ponds", "cross_creek", "split_river", "narrow_pass",
    "dry_gulch", "thorn_field", "reed_marsh", "hollow_ring",
    # Five maps transcribed from Solo Showdown screenshots, then fifteen generated from the five
    # generator families modeled on them; both tables in brawl_sim/maps/README.md.
    "hot_maze", "ghost_point", "shadow_spirits", "crescent_lakes", "twisting_vines",
    "pond_maze", "canal_maze", "picket_maze", "lagoon_ring", "square_lakes", "bush_halo",
    "bramble_ponds", "bramble_bend", "bramble_stars", "moon_gate", "half_moon", "moon_pools",
    "vine_springs", "vine_canal", "vine_hollow",
)

# A RandomizationSpec maps a "section.field" dotted key (matching configs/*.yaml section and
# leaf key names) to {"low": float, "high": float, "mode": "additive"|"multiplicative"}.
RandomizationSpec = dict


@dataclass(frozen=True)
class EnvConfig:
    map_h: int = 60
    map_w: int = 60
    map_names: tuple[str, ...] = ("open", "bushy", "walled")
    map_selection: str = "uniform"
    fixed_map: str = "open"
    view_h: int = 20
    view_w: int = 40
    # The camera model (BRAWL_SIM_DESIGN.md §9): the ground quad the screen shows, in tiles
    # relative to the hero's nominal screen anchor, corners TL/TR/BR/BL with y down (re-derived
    # from the shipped homography by tests/test_sim_camera.py); how far from each map edge (west,
    # east, north, south) the game camera stops following the hero; and the hero-to-camera offset
    # past which `hero.near_edge` is set. The quad as the view window and the 2-tile edge flag are
    # user decisions (2026-09-24). Defaults mirror configs/default.yaml.
    camera_quad: tuple[tuple[float, float], ...] = (
        (-14.11, -10.59), (14.77, -10.93), (11.73, 7.45), (-11.55, 7.01),
    )
    camera_clamp_onset: tuple[float, float, float, float] = (12.1, 12.8, 8.9, 5.5)
    camera_edge_flag_tiles: float = 2.0
    # Enemy observation slots, `EntityTracker`'s rule in DECISIONS: sightings before a slot is
    # granted, unseen decisions before it is released (core/slots.py).
    slots_promote_hits: int = 2
    slots_max_misses: int = 3
    n_enemies: int = 6
    randomize_enemy_types: bool = True
    enemy_type_weights: tuple[float, ...] = (1 / 7,) * 7
    fixed_enemy_types: tuple[str, ...] = ()
    max_projectiles: int = 128
    max_boxes: int = 16
    max_pickups: int = 32
    dt: float = 0.05
    max_episode_steps: int = 3000
    action_repeat: int = 1
    action_latency_seconds: float = 0.001
    n_move_bins: int = 16
    dash_on_idle: str = "facing"
    # A fifth attack value, 4 = auto-aimed attack (user decision, 2026-09-26): the dash goes at
    # the nearest alive enemy or unbroken crate within the dash's reach, visibility ignored, and
    # along the move bin (or `facing`) when nothing is in reach. STRUCTURAL: it widens
    # `action_nvec`, `hist.attack_onehot` and the action mask, so it is a per-run setting in
    # train.yaml's `env_overrides` and every run trained before it keeps `false`.
    auto_aim: bool = False
    obs_include_world_grid: bool = True
    obs_include_privileged: bool = True
    # `visibility.los` and `entities.los_from_hero`: a full-ray wall march over every entity pair
    # on every obs build. Nothing in training reads them, so train.yaml turns them off.
    obs_include_raw_los: bool = True
    # --- observation history. The sim keeps the last `history_frames` DECISIONS of the hero's
    # hp, ammo and position, the action it took, and each enemy's position plus whether the hero
    # saw it (core/state `hist_*`, written by core/history.push at the top of env.step).
    # `history_radius_tiles` is how far around the hero's current tile the `view` grid's
    # enemy-history channels reach. Both are structural -- they shape state buffers and the
    # observation -- so they live here, not in SimParams, and changing them breaks checkpoints.
    history_frames: int = 3
    history_radius_tiles: int = 4
    bots_break_boxes: bool = True
    bots_collect_cubes: bool = True
    bots_avoid_zone: bool = True
    # --- bot personalities. See bots/personality.py.
    bots_personalities: bool = True
    bots_personality_weights: tuple[float, ...] = (0.28, 0.12, 0.24, 0.16, 0.20)
    bots_min_aggressive: int = 1
    bots_attack_boxes: bool = True
    bots_sight_tiles: float = 14.0
    bots_bush_search_tiles: int = 4
    bots_zone_avoid_tiles: float = 4.0
    bots_camper_zone_flee_tiles: float = 2.0
    bots_wander_seconds: float = 2.5
    bots_hunt_cell_tiles: int = 10
    bots_hunt_arrive_tiles: float = 2.0
    bots_hunt_timeout_seconds: float = 12.0
    zone_enabled: bool = True
    # Sensing horizon for zone.hero_margin_local ONLY -- it changes no dynamics, just how far that
    # one observation field can see. Must equal the clamp the deployed gas estimator uses; see
    # BRAWL_DEPLOYMENT_DESIGN.md 9.14.
    zone_margin_horizon_tiles: float = 10.0
    iframes_block_zone: bool = False
    regen_enabled: bool = True
    drop_victim_cubes: bool = True
    # How far from its crate a broken crate's cube lands: a uniform distance in [min, max] tiles,
    # in a uniform direction (core/boxes.resolve_broken_boxes). 0 and 0 land it on the crate
    # itself.
    box_scatter_min_tiles: float = 0.0
    box_scatter_max_tiles: float = 0.0
    los_step_tiles: float = 0.5
    max_ray_tiles: float = 24.0
    debug_checks: bool = False
    compile: bool = False
    device: str = "cuda"

    @property
    def n_entities(self) -> int:
        return 1 + self.n_enemies

    @property
    def action_nvec(self) -> tuple[int, int]:
        """(move bins + idle, attack). The attack dimension is 4-valued: 0 = nothing, 1 = attack,
        2 = super, 3 = gadget, and 5-valued under `action.auto_aim` with 4 = auto-aimed attack
        (user decision, 2026-09-26). Attack, super and gadget are one masked choice (user's rule,
        2026-09-21), so the column widens rather than gaining a third dimension and the action
        stays (N,2) -- see core/hero.action_mask."""
        return (self.n_move_bins + 1, 5 if self.auto_aim else 4)

    @property
    def ray_steps(self) -> int:
        return int(math.ceil(self.max_ray_tiles / self.los_step_tiles))

    @property
    def agent_dt(self) -> float:
        """Seconds of simulated time per AGENT DECISION. `dt` is the SIM tick; one decision
        covers `action_repeat` of them (env.py's `_run_decision`), so the agent's decision rate
        is `1 / agent_dt` Hz -- 4 Hz at default.yaml's `dt: 0.05, action_repeat: 5`."""
        return self.dt * self.action_repeat

    @property
    def max_agent_steps(self) -> int:
        """`max_episode_steps` counted in DECISIONS rather than sim ticks -- the correct loop
        bound for anything that drives the env through `step()` (evaluation rollouts, watch,
        smoke tests). Rounded UP: with a remainder, the final decision is short (the sim
        truncates mid-window and `env.py` latches it) rather than being dropped."""
        return -(-self.max_episode_steps // self.action_repeat)

    @property
    def action_latency_ticks(self) -> int:
        return int(round(self.action_latency_seconds / self.dt))

    @property
    def latency_buf_len(self) -> int:
        return max(1, self.action_latency_ticks + 1)


# ---------------------------------------------------------------------------
# Nested-dict helpers. `spec` and the raw parsed YAML are plain dicts shaped exactly like
# configs/default.yaml + configs/brawlers.yaml; dotted paths address a leaf ("world.map_h",
# "bot_sniper.aim_noise_std_rad").
# ---------------------------------------------------------------------------

def _dget(d: dict, dotted: str, default=_MISSING):
    node = d
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(dotted)
            return default
        node = node[part]
    return node


def _dset(d: dict, dotted: str, value) -> dict:
    """Returns a new dict with `value` set at `dotted`; does not mutate `d`."""
    parts = dotted.split(".")
    out = dict(d)
    node = out
    for part in parts[:-1]:
        node[part] = dict(node.get(part, {}))
        node = node[part]
    node[parts[-1]] = value
    return out


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# EnvConfig loading
# ---------------------------------------------------------------------------

def _weights_transform(d: dict) -> tuple[float, ...]:
    return tuple(float(d[name]) for name in ARCHETYPE_SHORT_NAMES)


def _person_weights_transform(d: dict) -> tuple[float, ...]:
    return tuple(float(d[name]) for name in PERSON_SHORT_NAMES)


def _tuple_transform(x) -> tuple:
    return tuple(x)


_ONSET_SIDES = ("west", "east", "north", "south")


def _quad_transform(v) -> tuple[tuple[float, float], ...]:
    """Four `[x, y]` corners, TL/TR/BR/BL. The count is checked here and the orientation in
    `validate`, so a yaml with the corners in the wrong order fails at load, not by revealing
    nothing."""
    corners = list(v) if isinstance(v, (list, tuple)) else None
    if corners is None or len(corners) != 4 or any(len(c) != 2 for c in corners):
        raise ValueError(f"camera.quad needs exactly 4 corners of 2 numbers, got {v!r}")
    return tuple((float(x), float(y)) for x, y in corners)


def _onset_transform(d) -> tuple[float, float, float, float]:
    """`{west, east, north, south}` -> that order. A fragment naming only some sides is refused:
    a missing side would otherwise read as 0, a camera that never clamps there. (An override on
    top of default.yaml is deep-merged first, so overriding one side that way is fine.)"""
    if not isinstance(d, dict):
        raise ValueError(f"camera.clamp_onset must be a mapping of sides, got {d!r}")
    missing = [s for s in _ONSET_SIDES if s not in d]
    unknown = sorted(set(d) - set(_ONSET_SIDES))
    if missing or unknown:
        raise ValueError(
            f"camera.clamp_onset needs exactly the sides {', '.join(_ONSET_SIDES)}; "
            f"missing: {', '.join(missing) or 'none'}; unknown: {', '.join(unknown) or 'none'}"
        )
    return tuple(float(d[s]) for s in _ONSET_SIDES)


# (dotted YAML path, EnvConfig field name, type coercion)
_ENV_CONFIG_FIELDS = (
    ("world.map_h", "map_h", int),
    ("world.map_w", "map_w", int),
    ("world.maps", "map_names", _tuple_transform),
    ("world.map_selection", "map_selection", str),
    ("world.fixed_map", "fixed_map", str),
    ("view.height", "view_h", int),
    ("view.width", "view_w", int),
    ("camera.quad", "camera_quad", _quad_transform),
    ("camera.clamp_onset", "camera_clamp_onset", _onset_transform),
    ("camera.edge_flag_tiles", "camera_edge_flag_tiles", float),
    ("slots.promote_hits", "slots_promote_hits", int),
    ("slots.max_misses", "slots_max_misses", int),
    ("entities.n_enemies", "n_enemies", int),
    ("entities.randomize_enemy_types", "randomize_enemy_types", bool),
    ("entities.enemy_type_weights", "enemy_type_weights", _weights_transform),
    ("entities.fixed_enemy_types", "fixed_enemy_types", _tuple_transform),
    ("limits.max_projectiles", "max_projectiles", int),
    ("limits.max_boxes", "max_boxes", int),
    ("limits.max_pickups", "max_pickups", int),
    ("sim.dt", "dt", float),
    ("sim.max_episode_steps", "max_episode_steps", int),
    ("sim.action_repeat", "action_repeat", int),
    ("sim.action_latency_seconds", "action_latency_seconds", float),
    ("action.n_move_bins", "n_move_bins", int),
    ("action.dash_on_idle", "dash_on_idle", str),
    ("action.auto_aim", "auto_aim", bool),
    ("observation.include_world_grid", "obs_include_world_grid", bool),
    ("observation.include_privileged", "obs_include_privileged", bool),
    ("observation.include_raw_los", "obs_include_raw_los", bool),
    ("observation.history_frames", "history_frames", int),
    ("observation.history_radius_tiles", "history_radius_tiles", int),
    ("bots.break_boxes", "bots_break_boxes", bool),
    ("bots.collect_cubes", "bots_collect_cubes", bool),
    ("bots.avoid_zone", "bots_avoid_zone", bool),
    ("bots.personalities", "bots_personalities", bool),
    ("bots.personality_weights", "bots_personality_weights", _person_weights_transform),
    ("bots.min_aggressive", "bots_min_aggressive", int),
    ("bots.attack_boxes", "bots_attack_boxes", bool),
    ("bots.sight_tiles", "bots_sight_tiles", float),
    ("bots.bush_search_tiles", "bots_bush_search_tiles", int),
    ("bots.zone_avoid_tiles", "bots_zone_avoid_tiles", float),
    ("bots.camper_zone_flee_tiles", "bots_camper_zone_flee_tiles", float),
    ("bots.wander_seconds", "bots_wander_seconds", float),
    ("bots.hunt_cell_tiles", "bots_hunt_cell_tiles", int),
    ("bots.hunt_arrive_tiles", "bots_hunt_arrive_tiles", float),
    ("bots.hunt_timeout_seconds", "bots_hunt_timeout_seconds", float),
    ("zone.enabled", "zone_enabled", bool),
    ("zone.margin_horizon_tiles", "zone_margin_horizon_tiles", float),
    ("zone.iframes_block_zone", "iframes_block_zone", bool),
    ("regen.enabled", "regen_enabled", bool),
    ("cubes.drop_victim_cubes", "drop_victim_cubes", bool),
    ("cubes.box_scatter_min_tiles", "box_scatter_min_tiles", float),
    ("cubes.box_scatter_max_tiles", "box_scatter_max_tiles", float),
    ("perception.los_step_tiles", "los_step_tiles", float),
    ("perception.max_ray_tiles", "max_ray_tiles", float),
    ("engine.debug_checks", "debug_checks", bool),
    ("engine.compile", "compile", bool),
    ("device", "device", str),
)


def load_config(path, overrides: dict | None = None) -> EnvConfig:
    """`overrides` is deep-merged onto the file first. A field absent from both keeps its
    EnvConfig default rather than raising, so a partial YAML fragment loads."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)

    kwargs = {}
    for dotted, field_name, coerce in _ENV_CONFIG_FIELDS:
        value = _dget(raw, dotted, default=_ABSENT)
        if value is _ABSENT:
            continue  # keep the EnvConfig default
        kwargs[field_name] = coerce(value)
    return EnvConfig(**kwargs)


# ---------------------------------------------------------------------------
# RandomizationSpec
# ---------------------------------------------------------------------------

def _parse_range_entry(value) -> dict:
    if not isinstance(value, dict) or "low" not in value or "high" not in value:
        raise ValueError(f"randomization entry must be a {{low, high[, mode]}} mapping, got {value!r}")
    low, high = float(value["low"]), float(value["high"])
    mode = value.get("mode", "additive")
    if mode not in ("additive", "multiplicative"):
        raise ValueError(f"unknown randomization mode {mode!r}")
    if low > high:
        raise ValueError(f"randomization range has low > high: {value!r}")
    return {"low": low, "high": high, "mode": mode}


def load_randomization(path) -> RandomizationSpec:
    raw = yaml.safe_load(Path(path).read_text())
    if not raw:
        return {}
    return {key: _parse_range_entry(value) for key, value in raw.items()}


def apply_randomization(spec: dict, randomization: RandomizationSpec) -> dict:
    """Layer a RandomizationSpec (configs/randomization.yaml) onto a base spec (configs/
    default.yaml + configs/brawlers.yaml, already merged), since build_params/resample_params
    take no randomization argument of their own. Returns a new dict; `spec` is not mutated.

    mode="additive" (default) replaces the base value outright: final ~ U(low, high).
    mode="multiplicative" scales the base value: final ~ U(low, high) * base, so the base
    must resolve to a plain scalar (it's the nominal being jittered, not itself a range).
    """
    out = spec
    for dotted, r in randomization.items():
        if "." not in dotted:
            raise KeyError(f"randomization key must be 'section.field', got {dotted!r}")
        # `mode` defaults to "additive", as in _parse_range_entry, so a HAND-WRITTEN spec dict
        # passed straight to `BrawlVecEnv(..., randomization={"bot_sniper.base_hp": {"low": ...,
        # "high": ...}})` works without one.
        if r.get("mode", "additive") == "multiplicative":
            try:
                base_value = _dget(spec, dotted)
            except KeyError:
                base_value = _MISSING
            if base_value is _MISSING or isinstance(base_value, dict):
                raise ValueError(
                    f"multiplicative randomization for {dotted!r} needs an existing scalar "
                    f"base value in the base spec, got {base_value!r}"
                )
            new_value = {"low": r["low"] * base_value, "high": r["high"] * base_value}
        else:
            new_value = {"low": r["low"], "high": r["high"]}
        out = _dset(out, dotted, new_value)
    return out


# ---------------------------------------------------------------------------
# SimParams
# ---------------------------------------------------------------------------

_F32 = torch.float32
_I64 = torch.int64

# (SimParams attribute, brawlers.yaml leaf key, dtype) -- one column per KIND_YAML_NAMES entry.
# Enum-valued fields live in PER_KIND_ENUM_FIELDS below: they're name strings, not scalar-or-range
# numbers, so they can't ride _resolve_value.
PER_KIND_FIELDS = (
    ("base_hp", "base_hp", _F32),
    ("base_damage", "base_damage", _F32),
    ("move_speed", "move_speed", _F32),
    ("attack_range", "attack_range", _F32),
    ("attack_cooldown", "attack_cooldown", _F32),
    ("max_ammo", "max_ammo", _F32),
    ("reload_seconds", "reload_seconds", _F32),
    ("proj_speed", "proj_speed", _F32),
    # Lobbed shells only, and only when > 0: the shell takes this many seconds to reach its
    # landing point NO MATTER how far away it is, instead of flying at proj_speed (which then
    # describes only that kind's split shards). 0: the shell flies at proj_speed.
    ("proj_flight_seconds", "proj_flight_seconds", _F32),
    ("proj_radius", "proj_radius", _F32),
    ("proj_count", "proj_count", _I64),
    ("proj_spread_rad", "proj_spread_rad", _F32),
    ("aoe_radius", "aoe_radius", _F32),
    # The ring a detonating shell splits into (core/projectiles.SPLIT_DIR_TABLE). All three
    # default to 0, "no split", which is what every non-splitting kind resolves to.
    ("split_distance", "split_distance", _F32),
    ("split_damage_fraction", "split_damage_fraction", _F32),
    # How many shards, evenly spaced around the circle: 4 is Grom's world-axis cross, 6 is Spike's
    # hexagonal star. The DIRECTIONS are derived from this (core/projectiles._ring). validate()
    # caps it at core/projectiles.MAX_SPLITS, which sizes the (N,P,K) grid _spawn_splits
    # allocates; arms beyond it would be dropped in silence.
    ("split_count", "split_count", _I64),
    # How long a shard takes to fly `split_distance`; the shard's SPEED is derived from the two,
    # so the config states the observable duration. 0 falls back to `proj_speed`.
    ("split_seconds", "split_seconds", _F32),
    ("attack_arc_rad", "attack_arc_rad", _F32),
    # Swept melee (core/melee_sweep.py). `hitscan_count` <= 1, what every kind that does not set
    # it resolves to, is a single instantaneous cone, so these default to "not swept".
    ("hitscan_count", "hitscan_count", _I64),
    ("hitscan_sweep_rad", "hitscan_sweep_rad", _F32),
    # MELEE lifesteal (Edgar): heal this fraction of the damage a melee cone lands on PLAYERS; 0,
    # what every other kind resolves to, means none. Named for its source because only
    # `env._attack_phase` reads it: a kind that set it without a melee cone would heal nothing.
    # Not `super_heal`, the other lifesteal: a flat HP figure per player a super connects with
    # (Mortis), independent of damage dealt.
    ("melee_lifesteal_fraction", "melee_lifesteal_fraction", _F32),
    # Lingering damage areas (constants.ProjClass.HAZARD). A projectile whose owner's kind sets
    # on_hit_area_radius > 0 leaves a stationary sphere where it dies. All default to 0, "leaves
    # nothing", which is what every kind but Brock resolves to.
    ("on_hit_area_radius", "on_hit_area_radius", _F32),
    # A FRACTION of the projectile's own damage, never an absolute figure -- so it inherits
    # the cube bonus, enemy_damage_mult and the curriculum's tier multiplier, exactly as
    # split_damage_fraction does. See core/projectiles._spawn_hazards.
    ("on_hit_area_damage_fraction", "on_hit_area_damage_fraction", _F32),
    ("on_hit_area_ticks", "on_hit_area_ticks", _I64),
    ("on_hit_area_interval", "on_hit_area_interval", _F32),
    ("aim_noise_std_rad", "aim_noise_std_rad", _F32),
    ("aim_noise_tiles", "aim_noise_tiles", _F32),
    ("reaction_delay", "reaction_delay", _F32),
    ("lead_target_fraction", "lead_target_fraction", _F32),
    ("decision_period", "decision_period_ticks", _I64),
    ("dash_distance", "dash_distance", _F32),
    ("dash_duration", "dash_duration", _F32),
    ("dash_radius", "dash_radius", _F32),
    # Long dash: after `long_dash_seconds` without attacking, the next dash reaches
    # `long_dash_multiplier` times as far. 0 seconds means "this kind has no long dash", which is
    # what every kind but the hero resolves to.
    ("long_dash_seconds", "long_dash_seconds", _F32),
    ("long_dash_multiplier", "long_dash_multiplier", _F32),
    # SUPER. Per-kind, not hero-only: `super_charge_hits: 0` means "this kind has no super", which
    # is what every bot resolves to; giving a bot one is a brawlers.yaml block plus a `super_fire`
    # decision in its combat rule, with no plumbing change.
    ("super_charge_hits", "super_charge_hits", _I64),
    ("super_range", "super_range", _F32),
    ("super_damage", "super_damage", _F32),
    ("super_heal", "super_heal", _F32),
    ("super_proj_speed", "super_proj_speed", _F32),
    ("super_radius", "super_radius", _F32),
    # GADGET. Per-kind like the super: `gadget_cooldown: 0` means "this kind has no gadget", which
    # is what every bot resolves to. The five numbers are the whole mechanic: a Proj.GADGET_SPINNER
    # (ARTILLERY-class) flies up to `gadget_range` tiles toward the nearest enemy its thrower can
    # see (along the facing if none; hero.gadget_target), lands after `gadget_flight_seconds`, and
    # deals `gadget_damage` to everything within `gadget_radius`. The timer is state.ent_gadget_cd
    # (0 = ready, so a fresh episode starts charged); `validate()` rejects a cooldown with no
    # geometry.
    ("gadget_cooldown", "gadget_cooldown", _F32),
    ("gadget_range", "gadget_range", _F32),
    ("gadget_flight_seconds", "gadget_flight_seconds", _F32),
    ("gadget_damage", "gadget_damage", _F32),
    ("gadget_radius", "gadget_radius", _F32),
    # --- FIRE RULE. These four numbers plus `aim_model` below are a kind's whole fire rule, read
    # by bots/combat_rules.py, so a new brawler is a brawlers.yaml block, not a Python module.
    #
    # Every one of them is off/neutral at 0, which is what a kind that never mentions them gets:
    #   fire_needs_los 0                  fire without a physical line to the target (artillery
    #                                     arcs over walls; melee_hitscan checks LOS itself)
    #   fire_range_fraction 0             read as 1.0, i.e. no restriction beyond attack_range
    #   fire_lateral_speed_limit 0        never hold fire on a fast lateral mover
    # "Does this kind need its target inside a swing cone" is NOT a field: it is exactly
    # `attack_arc_rad > 0`, which cone_ray_tiles and combat.melee_hitscan already read; a second
    # field could disagree with it.
    ("fire_needs_los", "fire_needs_los", _I64),
    # Hold fire past this fraction of attack_range. Shelly's and Bull's 0.9: past it the fan has
    # spread wide enough that a small sidestep clears it. 0 means 1.0 -- see the block above.
    ("fire_range_fraction", "fire_range_fraction", _F32),
    # Hold fire on a target crossing faster than this many tiles/s, but only past
    # `fire_lateral_hold_range_fraction` of attack_range (close in, the fan is tight enough to
    # land anyway). 0 disables the rule: the limit is checked for > 0 first, since a 0 read
    # literally would hold fire on ANY moving target.
    ("fire_lateral_speed_limit", "fire_lateral_speed_limit", _F32),
    ("fire_lateral_hold_range_fraction", "fire_lateral_hold_range_fraction", _F32),
    # Preferred engagement distance as a fraction of this kind's own attack_range -- what the KITE
    # personality holds. Read by bots/policy.targeting into Targeting.desired_range, capped at the
    # kind's fire reach (fire_range_fraction x attack_range; user decision, 2026-09-21) so no
    # aggression tier holds its target out of range; the hero's 0 is never read.
    ("desired_range_fraction", "desired_range_fraction", _F32),
    # --- DIFFICULTY AXES. Both are per kind so the curriculum can scale them per (env, kind) like
    # `hp` and `damage`; both are authored on every bot block in brawlers.yaml because a tier
    # MULTIPLIES them and 0 times anything is 0.
    #
    # hero_focus in [0, 1]: how much a bot prefers the hero as a target when the hero is in view.
    # bots/perception.select_target discounts the hero's distance by (1 - hero_focus) and switches
    # to the hero, sticky target or not, when the discounted distance wins. 0 is the plain
    # nearest-visible-target rule. The hero's own value is never read.
    ("hero_focus", "hero_focus", _F32),
    # aggression, 1.0 = the bot as authored, gathered by core/stats.aggression_of for exactly three
    # uses: the HUNTER/KITE retreat HP threshold and the CAMPER fire veto (bots/personality.py) and
    # the KITE hold distance (bots/policy.targeting). 0, what a hand-built partial spec resolves
    # to, is read as 1.0, so it is neutral like every other optional field here; a NEGATIVE value
    # is rejected by validate().
    ("aggression", "aggression", _F32),
)

# (SimParams attribute, brawlers.yaml leaf key, enum class, default member name) -- one column per
# KIND_YAML_NAMES entry, same (N,K) i64 shape as PER_KIND_FIELDS produces. These are enum NAME
# STRINGS in the config ("SNIPER_BOLT", "LEAD"), so they resolve to a constant column of that
# member's value rather than through _resolve_value's scalar-or-range sampling -- an enum has no
# meaningful midpoint to randomize between.
#
# A misspelled name raises KeyError from the enum lookup, naming the bad string. Defaults are
# MEMBER names rather than 0: `Proj.NONE` and `AimModel.DIRECT` are both 0, but naming them keeps
# the table readable against enums where 0 is not the neutral member.
PER_KIND_ENUM_FIELDS = (
    ("proj_kind", "proj_kind", Proj, "NONE"),
    ("aim_model", "aim_model", AimModel, "DIRECT"),
)

# (SimParams attribute, dotted spec path, dtype). zone_start_time is handled separately: it's
# derived from zone.start_fraction * max_episode_steps * dt, not a direct copy.
PER_ENV_FIELDS = (
    ("enemy_hp_mult", "entities.enemy_hp_mult", _F32),
    ("enemy_damage_mult", "entities.enemy_damage_mult", _F32),
    ("unit_radius", "entities.unit_radius", _F32),
    ("pickup_radius", "cubes.pickup_radius", _F32),
    # FLAT HP per cube, not a fraction of base_hp -- see core/stats.effective_max_hp. The old
    # fractional key `hp_bonus_per_cube` is refused (_REMOVED_KEYS) rather than left to resolve
    # this to 0.
    ("cube_hp_flat", "cubes.hp_per_cube", _F32),
    ("cube_damage_bonus", "cubes.damage_bonus_per_cube", _F32),
    ("max_cubes", "cubes.max_cubes", _I64),
    ("cubes_per_box", "cubes.cubes_per_box", _I64),
    ("cubes_on_kill_base", "cubes.cubes_on_kill_base", _I64),
    ("box_hp", "boxes.hp", _F32),
    ("n_boxes", "boxes.n_boxes", _I64),
    ("zone_step_seconds", "zone.step_seconds", _F32),
    ("zone_tiles_per_step", "zone.tiles_per_step", _F32),
    # A FRACTION of each entity's own max HP per second, not a flat HP/s. The old flat key
    # `zone.dps` is refused (_REMOVED_KEYS), and validate() rejects a 0 here while the zone is on,
    # so the zone can never silently deal no damage.
    ("zone_hp_fraction", "zone.max_hp_fraction_per_second", _F32),
    ("zone_fraction_growth", "zone.fraction_growth_per_step", _F32),
    ("bush_reveal_radius", "perception.bush_reveal_radius", _F32),
    ("reveal_after_attack", "perception.reveal_after_attack", _F32),
    ("regen_delay", "regen.delay_seconds", _F32),
    # A FRACTION of the entity's own max HP per second, not a flat HP/s. The old flat key
    # `regen.per_second` is refused (_REMOVED_KEYS): its HP figure read as a fraction would be an
    # instant full heal.
    ("regen_fraction_per_second", "regen.max_hp_fraction_per_second", _F32),
)


def _spec_upper(value) -> float:
    """The largest value a scalar-or-range spec entry can ever resolve to. A {low, high} range
    resolves somewhere in [low, high] on every resample, so `high` is the static upper bound."""
    if isinstance(value, dict):
        return float(value["high"])
    return float(value)


def _spec_lower(value) -> float:
    """The smallest value a scalar-or-range spec entry can ever resolve to; `_spec_upper`'s twin."""
    if isinstance(value, dict):
        return float(value["low"])
    return float(value)


def cone_ray_tiles(spec: dict) -> float:
    """A STATIC upper bound (in tiles) on how far a melee CONE attack can reach, across every
    kind and every value its randomization range can produce, so `combat.melee_hitscan` can hand
    `terrain.march` a short ray budget instead of the full `cfg.ray_steps` for a short swing.

    **Derived from the SPEC, not from sampled `params`.** `params.attack_range.max()` would be a
    host sync, and `resample_params` runs inside `step()` via autoreset, so there is no safe
    moment to take one. Reading `high` off the spec instead gives a bound that is valid for every
    reset this env will ever do, computed once in pure Python.

    **Only kinds that can actually swing a cone count.** A kind with `attack_arc_rad == 0` can
    never satisfy `geo.point_in_cone`, so `melee_hitscan` discards its rows regardless of what
    the LOS check says; counting it would raise the bound to the longest RANGED range for
    nothing.
    """
    best = 0.0
    for kind in KIND_YAML_NAMES:
        blob = spec.get(kind, {})
        if _spec_upper(blob.get("attack_arc_rad", 0)) > 0:
            best = max(best, _spec_upper(blob.get("attack_range", 0)))
    return best


def attack_ray_tiles(spec: dict) -> float:
    """A STATIC upper bound (in tiles) on any kind's `attack_range`, across every value its
    randomization range can produce. Same spec-derived, host-sync-free construction as
    `cone_ray_tiles` -- see that function for why it reads the spec rather than sampled params.

    Used by bots/policy.targeting to bound the loot-box LOS march. Sound for the same reason the
    cone bound is: a box further away than `attack_range` is rejected by `fire_gate`'s own
    `dist <= attack_range` term, so a shortened ray giving it a wrong LOS answer cannot change any
    fire decision. Unlike `cone_ray_tiles` this spans ALL kinds, since any archetype can shoot a
    box.
    """
    return max(
        (_spec_upper(spec.get(kind, {}).get("attack_range", 0)) for kind in KIND_YAML_NAMES),
        default=0.0,
    )


def dash_ray_tiles(spec: dict) -> float:
    """A STATIC upper bound (in tiles) on how far any dash can go: `dash_distance` times the long
    dash's `long_dash_multiplier` (floored at 1, as `hero.long_dash_scale` floors it), across
    every kind and every value its randomization range can produce. Budgets `hero.start_dash`'s
    body march (`terrain.body_travel`), which otherwise pays the full `cfg.ray_steps` circle
    tests for a dash of a few tiles. Same spec-derived, host-sync-free construction as
    `cone_ray_tiles`; the curriculum's tier multipliers never touch dash stats.
    """
    return max(
        (_spec_upper(spec.get(kind, {}).get("dash_distance", 0))
         * max(1.0, _spec_upper(spec.get(kind, {}).get("long_dash_multiplier", 1.0)))
         for kind in KIND_YAML_NAMES),
        default=0.0,
    )


def shot_step_tiles(spec: dict, dt: float) -> float:
    """A STATIC upper bound (in tiles) on how far any shot that a wall or a box can stop moves in
    one tick, across every kind and every value its randomization range can produce: the fastest
    such shot's speed times `dt`. Budgets `projectiles.step_projectiles`' per-tick wall march,
    which otherwise pays the full `cfg.ray_steps` to check a move of a fraction of a tile, and
    feeds `box_reach_tiles`. Same spec-derived, host-sync-free construction as `cone_ray_tiles`.

    Walls and boxes only ever stop an alive, non-lobbed, non-piercing projectile --
    `step_projectiles` masks every other slot out of both answers -- so two speeds count:
    - `proj_speed`, which every volley flies at and which a shard falls back to when
      `split_seconds` is 0. Taken over every kind; a lobbed kind's shell never meets a wall, but
      counting it costs nothing.
    - a shard's `split_distance / split_seconds`. A `split_seconds` range reaching down to 0 has no
      finite bound, so the result is infinite and the caller falls back to the full ray.

    Two speeds stay out on purpose: a super always pierces (`spawn_supers`), and a timed lob's
    speed comes from its landing distance, but an ARTILLERY shell is never tested against walls
    or boxes in flight. The curriculum's tier multipliers never touch any of these stats.
    """
    fastest = 0.0
    for kind in KIND_YAML_NAMES:
        blob = spec.get(kind, {})
        fastest = max(fastest, _spec_upper(blob.get("proj_speed", 0)))
        if _spec_upper(blob.get("split_distance", 0)) > 0 and _spec_upper(blob.get("split_seconds", 0)) > 0:
            shortest = _spec_lower(blob["split_seconds"])
            shard = _spec_upper(blob["split_distance"]) / shortest if shortest > 0 else math.inf
            fastest = max(fastest, shard)
    return fastest * dt


def box_reach_tiles(spec: dict, shot_step: float) -> float:
    """A STATIC upper bound (in tiles) on how far a box's centre can be from the point
    `step_projectiles` gathers box candidates around and still take damage this tick, so its box
    tests skip every box on a tile further away (`projectiles._box_candidates`). The larger of:
    - a shot's sweep: the widest `proj_radius` any kind can roll (a shard inherits its shell's),
      plus BOX_RADIUS, plus half of `shot_step` -- every point of a tick's path lies within half
      its length of the path's midpoint, which is where the candidates are gathered.
    - a blast's radius, from the blast point: the widest `aoe_radius` (a shell), `gadget_radius`
      (the spinner) or `on_hit_area_radius` (a hazard) any kind can roll.
    Supers stay out for the reason `shot_step_tiles` gives: a piercing bolt never hits a box.
    """
    widest_shot = max((_spec_upper(spec.get(kind, {}).get("proj_radius", 0)) for kind in KIND_YAML_NAMES),
                      default=0.0)
    widest_blast = max(
        (_spec_upper(spec.get(kind, {}).get(field, 0))
         for kind in KIND_YAML_NAMES for field in ("aoe_radius", "gadget_radius", "on_hit_area_radius")),
        default=0.0,
    )
    return max(widest_shot + BOX_RADIUS + shot_step / 2.0, widest_blast)


class SimParams:
    """Per-env numerics. Every field is a tensor with a leading (N,) dim, even when the
    config value is a scalar, so downstream code stays uniform. Per-kind fields are (N, K);
    per-env fields are (N,).

    **`SCALAR_FIELDS` is the explicit exception list to that rule**, an enumerated allowlist
    rather than a convention: a field here is a plain Python number, so anything that walks
    `__slots__` expecting tensors (`.clone()`, row indexing, `torch.where`) breaks on it. Naming
    them lets such a walk skip exactly these and still fail loudly on a non-tensor field added
    by accident. Both halves are pinned by
    tests/test_spawn.py::test_sim_params_fields_are_tensors_except_the_declared_scalars.

    A field earns a place here only by being STRUCTURAL: something that sizes a tensor dimension
    or drives Python control flow (the ray budgets), and so can never live on the device without
    forcing a host sync in the hot path. They are spec-derived and constant for the env's whole
    lifetime; `resample_params` does not touch them.
    """

    SCALAR_FIELDS = ("cone_ray_tiles", "attack_ray_tiles", "dash_ray_tiles", "shot_step_tiles", "box_reach_tiles")

    __slots__ = (
        tuple(attr for attr, _, _ in PER_KIND_FIELDS)
        + tuple(attr for attr, _, _, _ in PER_KIND_ENUM_FIELDS)
        + tuple(attr for attr, _, _ in PER_ENV_FIELDS)
        + ("zone_start_time",) + SCALAR_FIELDS
    )


def _resolve_value(value, gen: torch.Generator, device, n_envs: int, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, dict):
        low, high = float(value["low"]), float(value["high"])
        if low > high:
            raise ValueError(f"range has low > high: {value!r}")
        u = torch.rand((n_envs,), generator=gen, device=device, dtype=torch.float32)
        sampled = low + u * (high - low)
    else:
        sampled = torch.full((n_envs,), float(value), device=device, dtype=torch.float32)
    if dtype == torch.int64:
        return torch.round(sampled).to(torch.int64)
    return sampled


def _resolve_all(cfg: EnvConfig, n_envs: int, device, gen: torch.Generator, spec: dict) -> SimParams:
    params = SimParams()

    for attr, yaml_key, dtype in PER_KIND_FIELDS:
        cols = [
            _resolve_value(spec.get(kind, {}).get(yaml_key, 0), gen, device, n_envs, dtype)
            for kind in KIND_YAML_NAMES
        ]
        setattr(params, attr, torch.stack(cols, dim=1))

    for attr, yaml_key, enum_cls, default_name in PER_KIND_ENUM_FIELDS:
        cols = [
            torch.full(
                (n_envs,), int(enum_cls[spec.get(kind, {}).get(yaml_key, default_name)]),
                device=device, dtype=torch.int64,
            )
            for kind in KIND_YAML_NAMES
        ]
        setattr(params, attr, torch.stack(cols, dim=1))

    for attr, dotted, dtype in PER_ENV_FIELDS:
        value = _dget(spec, dotted, default=0)
        setattr(params, attr, _resolve_value(value, gen, device, n_envs, dtype))

    start_fraction = _dget(spec, "zone.start_fraction", default=0.0)
    frac = _resolve_value(start_fraction, gen, device, n_envs, _F32)
    params.zone_start_time = frac * (cfg.max_episode_steps * cfg.dt)

    # Python floats, spec-derived, identical for every reset -- see cone_ray_tiles' docstring and
    # SimParams.SCALAR_FIELDS. resample_params deliberately does NOT re-derive these (the spec
    # cannot change mid-episode, so there is nothing to resample).
    params.cone_ray_tiles = cone_ray_tiles(spec)
    params.attack_ray_tiles = attack_ray_tiles(spec)
    params.dash_ray_tiles = dash_ray_tiles(spec)
    # Capped at the full ray: a march never takes more than cfg.ray_steps samples, and an
    # unbounded shard speed (see shot_step_tiles) lands exactly there.
    params.shot_step_tiles = min(shot_step_tiles(spec, cfg.dt), cfg.max_ray_tiles)
    params.box_reach_tiles = box_reach_tiles(spec, params.shot_step_tiles)

    return params


# Retired keys, and what replaced them. Each is a rename where the new key means something
# NUMERICALLY DIFFERENT, so a config left on the old spelling would not fail: it would resolve the
# new key to `_dget`'s default of 0 and silently disable the mechanic -- invisible at
# construction, and recognisable only much later as "the agent learned something strange".
#
# Checked in `build_params`, not `_resolve_all`, which `resample_params` also runs inside `step()`
# on every autoreset; the spec cannot change mid-run, so once at construction is enough.
_REMOVED_KEYS = {
    "cubes.hp_bonus_per_cube": (
        "cubes.hp_per_cube -- a power cube now adds a FLAT number of HP (400, the real game's "
        "value), not a FRACTION of base_hp. A 0.15 left under the new key would give each cube "
        "0.15 HP; the old key under the new schema gives cubes no HP at all."
    ),
    "zone.dps": (
        "zone.max_hp_fraction_per_second -- the zone deals a FRACTION of each entity's own max HP "
        "per second (0.2 = dead in 5s), not flat HP/s."
    ),
    "regen.per_second": (
        "regen.max_hp_fraction_per_second -- regen is a FRACTION of max HP per second (0.13), not "
        "flat HP/s. The old 400.0 read under the new key would be a full heal every tick."
    ),
}


def check_removed_keys(spec: dict) -> None:
    """Raises if `spec` still carries a key this schema has renamed. See _REMOVED_KEYS."""
    stale = [(key, replacement) for key, replacement in _REMOVED_KEYS.items()
             if _dget(spec, key, default=_ABSENT) is not _ABSENT]
    if stale:
        lines = "\n".join(f"  {key!r} -> {replacement}" for key, replacement in stale)
        raise ValueError(
            f"config uses {len(stale)} key(s) this schema has renamed, which would be silently "
            f"ignored:\n{lines}"
        )


def build_params(cfg: EnvConfig, n_envs: int, device, gen: torch.Generator, spec: dict) -> SimParams:
    check_removed_keys(spec)
    return _resolve_all(cfg, n_envs, device, gen, spec)


def resample_params(
    params: SimParams, reset_mask: torch.Tensor, cfg: EnvConfig, gen: torch.Generator, spec: dict,
) -> None:
    """MUTATES: every SimParams field, via torch.where over the full batch -- masked rows get
    a freshly sampled value, unmasked rows are untouched. Sync-free."""
    device = reset_mask.device
    n_envs = reset_mask.shape[0]
    fresh = _resolve_all(cfg, n_envs, device, gen, spec)
    mask_k = reset_mask.unsqueeze(1)

    for attr, _, _ in PER_KIND_FIELDS:
        setattr(params, attr, torch.where(mask_k, getattr(fresh, attr), getattr(params, attr)))
    for attr, _, _, _ in PER_KIND_ENUM_FIELDS:
        setattr(params, attr,
                torch.where(mask_k, getattr(fresh, attr), getattr(params, attr)))

    for attr, _, _ in PER_ENV_FIELDS:
        setattr(params, attr, torch.where(reset_mask, getattr(fresh, attr), getattr(params, attr)))
    params.zone_start_time = torch.where(reset_mask, fresh.zone_start_time, params.zone_start_time)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def peak_projectile_demand(cfg: EnvConfig, params: SimParams) -> int:
    """Worst-case CONCURRENT projectile slots this roster can occupy. One-time check, not hot
    path -- host syncs are fine here.

    Projectiles persist for many ticks, so what fills the buffer is burst x residency:

        lifetime      = proj_flight_seconds, or attack_range / proj_speed for a normal shot
        residency     = lifetime + dwell (see below)
        volleys_alive = min(max_ammo, floor(residency / attack_cooldown) + 1)
        per_entity    = volleys_alive * proj_count * (split_count if this kind splits else 1)

    i.e. how many volleys a full clip can put in the air before the first one expires. The roster
    worst case is the hero plus `n_enemies` copies of the most demanding BOT kind, since
    `randomize_enemy_types` can legitimately draw an all-one-archetype lobby (and
    `fixed_enemy_types` can ask for one outright).

    **This is a bound, not a prediction.** Real peaks run well under it -- bots need a target in
    range, are staggered by `decision_period_ticks`, and die -- but the buffer has to be sized
    for what CAN happen, because overflow is silent (`alloc_slots` drops the shot rather than
    raising) and a thinned shotgun blast is a subtle, hard-to-attribute behavior bug.
    """
    per_kind_max = lambda t: t.max(dim=0).values.to(torch.float32)  # noqa: E731  -- over envs

    count = per_kind_max(params.proj_count)
    ammo = per_kind_max(params.max_ammo)
    cooldown = torch.clamp(per_kind_max(params.attack_cooldown), min=1e-3)
    attack_range = per_kind_max(params.attack_range)
    speed = torch.clamp(per_kind_max(params.proj_speed), min=1e-3)
    flight = per_kind_max(params.proj_flight_seconds)
    # Each kind is budgeted against its OWN ring width, not core/projectiles.MAX_SPLITS, so Spike's
    # 6 arms do not raise the buffer Grom's 4-arm cross must reserve.
    split_count = per_kind_max(params.split_count)
    splits = torch.where(per_kind_max(params.split_distance) > 0, split_count, torch.ones_like(split_count))

    flight_life = torch.where(flight > 0, flight, attack_range / speed)

    # **DWELL: how long a shot keeps occupying slots AFTER its own projectile ends.** Two mechanics
    # produce it, treated as one concept:
    #   - a HAZARD lingers on_hit_area_ticks * on_hit_area_interval seconds
    #   - SHARDS fly for split_seconds after the shell detonates
    # In both cases the successor INHERITS the parent's slot (`_spawn_hazards` / `_spawn_splits`
    # both run after prj_alive is updated), so the slot COUNT is already covered by `splits` above;
    # dwell extends the RESIDENCY, which decides how many of a kind's shots are in the buffer at
    # once. max(), not a sum: no kind does both.
    haz_life = (
        per_kind_max(params.on_hit_area_ticks).to(torch.float32)
        * per_kind_max(params.on_hit_area_interval)
    )
    shard_seconds = per_kind_max(params.split_seconds)
    shard_life = torch.where(
        per_kind_max(params.split_distance) > 0,
        torch.where(shard_seconds > 0, shard_seconds, per_kind_max(params.split_distance) / speed),
        torch.zeros_like(shard_seconds),
    )
    dwell = torch.maximum(haz_life, shard_life)

    residency = flight_life + dwell
    volleys = torch.minimum(ammo, torch.floor(residency / cooldown) + 1.0)
    per_entity = volleys * count * splits  # (K,)

    hero = float(per_entity[0])
    worst_bot = float(per_entity[1:].max()) if per_entity.numel() > 1 else 0.0
    return int(math.ceil(hero + cfg.n_enemies * worst_bot))


def validate(cfg: EnvConfig, params: SimParams) -> None:
    """One-time sanity check, not called from the sim's hot path -- host syncs here are fine.
    The "any range with low > high" error from the spec table is raised earlier, inside
    _resolve_value, since a sampled tensor no longer carries its low/high metadata."""
    if cfg.view_h > cfg.map_h or cfg.view_w > cfg.map_w:
        raise ValueError(
            f"view ({cfg.view_h}x{cfg.view_w}) is larger than the map ({cfg.map_h}x{cfg.map_w})"
        )
    if cfg.n_enemies < 1:
        raise ValueError(f"n_enemies must be >= 1, got {cfg.n_enemies}")

    if len(cfg.bots_personality_weights) != len(PERSON_SHORT_NAMES):
        raise ValueError(
            f"bots_personality_weights has {len(cfg.bots_personality_weights)} entries, expected "
            f"{len(PERSON_SHORT_NAMES)} ({', '.join(PERSON_SHORT_NAMES)})"
        )
    if any(w < 0 for w in cfg.bots_personality_weights):
        raise ValueError(f"bots_personality_weights must be >= 0, got {cfg.bots_personality_weights}")
    if sum(cfg.bots_personality_weights) <= 0:
        raise ValueError("bots_personality_weights sum to 0; at least one personality must be drawable")
    # core/spawn.sample_personalities forces this many slots to an AGGRESSIVE_PERSONS draw, and it
    # only has cfg.n_enemies bot slots to work with (slot 0 is the hero and is never assigned one).
    if not 0 <= cfg.bots_min_aggressive <= cfg.n_enemies:
        raise ValueError(
            f"bots_min_aggressive must be in [0, n_enemies={cfg.n_enemies}], got "
            f"{cfg.bots_min_aggressive}"
        )
    if cfg.bots_bush_search_tiles < 1:
        raise ValueError(f"bots_bush_search_tiles must be >= 1, got {cfg.bots_bush_search_tiles}")
    if cfg.bots_hunt_cell_tiles < 1:
        raise ValueError(f"bots_hunt_cell_tiles must be >= 1, got {cfg.bots_hunt_cell_tiles}")
    # One bush waypoint per hunt cell, and a bot's visited-set is a 63-bit mask
    # (state.ent_hunt_seen). maps/loader.bush_waypoints subsamples rather than failing if a map
    # exceeds this, but a config that guarantees truncation everywhere is a mistake worth naming.
    n_cells = -(-cfg.map_w // cfg.bots_hunt_cell_tiles) * -(-cfg.map_h // cfg.bots_hunt_cell_tiles)
    if n_cells > HUNT_WAYPOINT_BITS:
        raise ValueError(
            f"bots_hunt_cell_tiles={cfg.bots_hunt_cell_tiles} splits a "
            f"{cfg.map_w}x{cfg.map_h} map into {n_cells} cells, more than the "
            f"{HUNT_WAYPOINT_BITS} bush waypoints a hunter's int64 visited-mask can track. "
            "Raise bots_hunt_cell_tiles."
        )
    if cfg.bots_wander_seconds <= 0:
        raise ValueError(f"bots_wander_seconds must be > 0, got {cfg.bots_wander_seconds}")
    if cfg.bots_hunt_timeout_seconds <= 0:
        raise ValueError(
            f"bots_hunt_timeout_seconds must be > 0, got {cfg.bots_hunt_timeout_seconds}"
        )
    if not 0 <= cfg.box_scatter_min_tiles <= cfg.box_scatter_max_tiles:
        raise ValueError(
            f"cubes.box_scatter_min_tiles ({cfg.box_scatter_min_tiles}) and box_scatter_max_tiles "
            f"({cfg.box_scatter_max_tiles}) must satisfy 0 <= min <= max"
        )
    for name in (*cfg.map_names, cfg.fixed_map):
        if name not in _KNOWN_MAP_NAMES:
            raise ValueError(f"unknown map name {name!r}; known maps are {_KNOWN_MAP_NAMES}")

    demand = peak_projectile_demand(cfg, params)
    if cfg.max_projectiles < demand:
        raise ValueError(
            f"max_projectiles ({cfg.max_projectiles}) is below the worst-case concurrent demand "
            f"({demand}) for this roster -- volleys would be silently thinned (alloc_slots returns "
            f"ok=False and the shot just loses projectiles). See config.peak_projectile_demand."
        )

    episode_seconds = cfg.max_episode_steps * cfg.dt
    if bool((params.zone_start_time >= episode_seconds).any()):
        raise ValueError(
            f"zone_start_time >= max_episode_steps*dt ({episode_seconds}) for at least one env"
        )
    if bool((cfg.dt >= params.zone_step_seconds).any()):
        raise ValueError(f"dt ({cfg.dt}) >= zone_step_seconds for at least one env")

    if cfg.action_latency_seconds < 0:
        raise ValueError(f"action_latency_seconds must be >= 0, got {cfg.action_latency_seconds}")
    # The history rings are allocated K deep and `history.push` shifts K-1 slots; 0 would allocate
    # an empty ring and index slot 0 of it. 1 is the legitimate minimum: "just the last decision".
    # The radius is the `view` grid's enemy-history window; 0 would leave only the hero's tile.
    if cfg.history_frames < 1:
        raise ValueError(f"observation.history_frames must be >= 1, got {cfg.history_frames}")
    if cfg.history_radius_tiles < 1:
        raise ValueError(
            f"observation.history_radius_tiles must be >= 1, got {cfg.history_radius_tiles}"
        )

    # The camera quad must be convex, wound the way core/camera.in_camera assumes (clockwise on
    # screen, y down) and contain the hero's anchor at the origin. The predicate is the same edge
    # cross product `in_camera` evaluates, so a mis-ordered yaml fails HERE with a message instead
    # of passing every entity through as "off screen".
    quad = cfg.camera_quad
    if len(quad) != 4:
        raise ValueError(f"camera.quad needs 4 corners, got {len(quad)}")
    for i in range(4):
        (ax, ay), (bx, by) = quad[i], quad[(i + 1) % 4]
        (cx, cy) = quad[(i + 2) % 4]
        ex, ey = bx - ax, by - ay
        if ex * (cy - by) - ey * (cx - bx) <= 0:
            raise ValueError(
                f"camera.quad is not convex and clockwise at corner {i + 1} -> {i + 2}: "
                f"corners must run TL, TR, BR, BL with y down, got {quad}"
            )
        if ex * (0.0 - ay) - ey * (0.0 - ax) <= 0:
            raise ValueError(
                f"camera.quad does not contain the hero anchor (origin) on the side of edge "
                f"{i + 1} -> {i + 2}: {quad}"
            )
    if len(cfg.camera_clamp_onset) != 4 or any(o < 0 for o in cfg.camera_clamp_onset):
        raise ValueError(
            f"camera.clamp_onset needs four non-negative distances (west, east, north, south), "
            f"got {cfg.camera_clamp_onset}"
        )
    if cfg.camera_edge_flag_tiles <= 0:
        raise ValueError(f"camera.edge_flag_tiles must be > 0, got {cfg.camera_edge_flag_tiles}")
    if cfg.slots_promote_hits < 1:
        raise ValueError(f"slots.promote_hits must be >= 1, got {cfg.slots_promote_hits}")
    if cfg.slots_max_misses < 0:
        raise ValueError(f"slots.max_misses must be >= 0, got {cfg.slots_max_misses}")

    if cfg.action_repeat < 1:
        raise ValueError(f"action_repeat must be >= 1, got {cfg.action_repeat}")
    # Not an error: a remainder just makes the LAST decision of an episode short, which env.py
    # handles correctly (truncation is latched at whichever sub-tick crosses the cap). Worth
    # naming anyway, since it makes episode length in decisions non-obvious and is almost always
    # an accident rather than an intent.
    if cfg.max_episode_steps % cfg.action_repeat:
        print(
            f"[config] max_episode_steps ({cfg.max_episode_steps}) is not a multiple of "
            f"action_repeat ({cfg.action_repeat}); the last decision of each episode covers "
            f"{cfg.max_episode_steps % cfg.action_repeat} sim tick(s) instead of "
            f"{cfg.action_repeat}. Episodes run {cfg.max_agent_steps} decisions."
        )

    # --- per-kind mechanic sanity. Each of these is a config that raises NO error and produces
    # subtly wrong behavior for an entire training run, visible only much later as "the agent
    # learned something strange".
    for k, kind_name in enumerate(KIND_YAML_NAMES):
        # Swept melee: the sub-swing schedule is a floor() crossing on elapsed time, so it can fire
        # at most ONE sub-swing per tick. An interval below dt silently DROPS swings -- a Buzz
        # configured for 5 hitscans would land 4, with nothing anywhere saying so.
        count = int(params.hitscan_count[:, k].max())
        if count > 1:
            interval = float(params.attack_cooldown[:, k].min()) / count
            if interval < cfg.dt:
                raise ValueError(
                    f"{kind_name}: hitscan_count {count} over attack_cooldown "
                    f"{float(params.attack_cooldown[:, k].min())}s gives a {interval:.4f}s sub-swing "
                    f"interval, below dt {cfg.dt}. core/melee_sweep fires at most one sub-swing per "
                    f"tick, so swings would be silently dropped. Raise attack_cooldown or lower "
                    f"hitscan_count."
                )
        # `hitscan_sweep_rad == 0` with `hitscan_count > 1` is legitimate and not rejected: Edgar's
        # two hits land at ONE angle, each re-testing cone and LOS against a target that has had
        # time to move and lifestealing on its own, so it is not one double-damage swing.

        # Splits: `split_count` decides both how many shards spawn and which ring they fly, and
        # BOTH failure modes here are silent. A count of 0 with a split_distance set loads fine and
        # produces a shell that detonates into nothing -- the shape of "added the distance, forgot
        # the count" -- and a count above MAX_SPLITS is truncated by the fixed-width grid
        # _spawn_splits allocates against, so a 8-arm star would quietly fire 6 arms forever.
        if float(params.split_distance[:, k].max()) > 0:
            count = int(params.split_count[:, k].max())
            if count <= 0:
                raise ValueError(
                    f"{kind_name}: split_distance is set but split_count is 0, so a detonating "
                    f"shell would spawn no shards at all. Set the number of arms in the ring."
                )
            if count > MAX_SPLITS:
                raise ValueError(
                    f"{kind_name}: split_count {count} exceeds core/projectiles.MAX_SPLITS "
                    f"({MAX_SPLITS}), which sizes the grid _spawn_splits allocates against -- the "
                    f"extra arms would be dropped in silence. Raise MAX_SPLITS (and re-check "
                    f"max_projectiles, which is budgeted per kind from this count)."
                )

        # Lingering hazards: a 0 interval makes `floor(age / interval)` explode on the first tick,
        # so the sphere expires immediately having dealt nothing. 0 ticks is the same, more
        # obviously. Both leave `on_hit_area_radius` configured and doing nothing.
        if float(params.on_hit_area_radius[:, k].max()) > 0:
            if float(params.on_hit_area_interval[:, k].min()) <= 0:
                raise ValueError(
                    f"{kind_name}: on_hit_area_radius is set but on_hit_area_interval is 0, so the "
                    f"lingering area expires on its first tick without ever dealing damage."
                )
            if int(params.on_hit_area_ticks[:, k].min()) <= 0:
                raise ValueError(
                    f"{kind_name}: on_hit_area_radius is set but on_hit_area_ticks is 0, so the "
                    f"lingering area never damages anything."
                )

        # --- fire rule. Each check catches a brawlers.yaml block that loads without complaint and
        # then produces a bot that quietly never shoots, or shoots wrong, for a whole training run.
        frac = float(params.fire_range_fraction[:, k].max())
        if frac > 1.0 or float(params.fire_range_fraction[:, k].min()) < 0:
            raise ValueError(
                f"{kind_name}: fire_range_fraction must be in [0, 1] -- it is a fraction OF "
                f"attack_range, and 0 means 'no restriction' (read as 1.0). Got {frac}."
            )
        if (float(params.fire_lateral_hold_range_fraction[:, k].max()) > 0
                and float(params.fire_lateral_speed_limit[:, k].max()) <= 0):
            raise ValueError(
                f"{kind_name}: fire_lateral_hold_range_fraction is set but "
                f"fire_lateral_speed_limit is 0, which disables the lateral hold entirely -- the "
                f"range fraction it is paired with does nothing. Set the speed limit or drop both."
            )
        desired = float(params.desired_range_fraction[:, k].max())
        if desired > 1.0 or float(params.desired_range_fraction[:, k].min()) < 0:
            raise ValueError(
                f"{kind_name}: desired_range_fraction must be in [0, 1] -- it is a fraction OF "
                f"attack_range. Got {desired}."
            )
        # 0 is legitimate for the hero (his movement never reads this) and a mistake for a bot: a
        # KITE bot of that kind would hold range 0, i.e. charge to point blank. It is the one
        # fire-rule field with no usable default, so a new brawler block MUST name it. Gated on
        # attack_range, of which it is a FRACTION: a kind with no range can never pass fire_gate,
        # so demanding one would only reject hand-built partial specs.
        if (k != int(Kind.HERO_MORTIS) and desired <= 0
                and float(params.attack_range[:, k].max()) > 0):
            raise ValueError(
                f"{kind_name}: attack_range is set but desired_range_fraction is 0, so a KITE bot "
                f"of this kind would hold range 0 -- it would charge to point blank instead of "
                f"keeping distance. Set the fraction of attack_range this kind fights at."
            )

        # --- difficulty axes. hero_focus is a discount FACTOR on a distance, so outside [0, 1] it
        # either inflates the hero's distance (a bot that avoids the hero) or makes it negative
        # (argmin picks the hero at any range, including out of sight). aggression divides a
        # threshold, so a negative value flips retreat into charge. 0 is neutral for both (see
        # PER_KIND_FIELDS) and is not rejected: hand-built partial specs in tests resolve to it.
        # tests/test_configs_files.py pins the shipped brawlers.yaml to author both on every bot.
        focus_hi = float(params.hero_focus[:, k].max())
        focus_lo = float(params.hero_focus[:, k].min())
        if focus_lo < 0 or focus_hi > 1.0:
            raise ValueError(
                f"{kind_name}: hero_focus must be in [0, 1] -- it discounts the hero's distance "
                f"by (1 - hero_focus) in bots/perception.select_target. Got {focus_lo}..{focus_hi}."
            )
        if float(params.aggression[:, k].min()) < 0:
            raise ValueError(
                f"{kind_name}: aggression must be >= 0 (0 is read as 1.0; a negative value would "
                f"turn the retreat threshold into a charge threshold). Got "
                f"{float(params.aggression[:, k].min())}."
            )

        # --- gadget. `gadget_cooldown > 0` is what "this kind has a gadget" means
        # (core/hero.gadget_ready), so a block that sets it and leaves the geometry at 0 loads fine
        # and ships a spinner that reaches nowhere, lands never and hurts nothing. Damage is not
        # checked: 0 is a legitimate "utility" gadget, the geometry is not.
        if float(params.gadget_cooldown[:, k].min()) < 0:
            raise ValueError(f"{kind_name}: gadget_cooldown must be >= 0 (0 = no gadget).")
        if float(params.gadget_cooldown[:, k].max()) > 0:
            for field in ("gadget_range", "gadget_flight_seconds", "gadget_radius"):
                if float(getattr(params, field)[:, k].min()) <= 0:
                    raise ValueError(
                        f"{kind_name}: gadget_cooldown is set but {field} is 0, so the gadget "
                        f"would fire a spinner that cannot reach, land or hit anything. Set all "
                        f"three of gadget_range, gadget_flight_seconds and gadget_radius, or "
                        f"drop the cooldown."
                    )

        # A projectile with no speed and no fixed flight time never moves and never expires by
        # distance -- it would sit on the shooter forever, holding a slot.
        if int(params.proj_count[:, k].max()) > 0:
            has_speed = float(params.proj_speed[:, k].min()) > 0
            has_flight = float(params.proj_flight_seconds[:, k].min()) > 0
            if not (has_speed or has_flight):
                raise ValueError(
                    f"{kind_name}: proj_count > 0 but both proj_speed and proj_flight_seconds are "
                    f"0, so its projectiles would never move or expire."
                )

    if cfg.zone_enabled:
        # An ENABLED zone that deals no damage is always a mistake: a spec without this key
        # resolves it to _dget's default of 0, and the zone silently becomes inert, which shows up
        # only much later as "episodes never resolve". Fail loudly here instead.
        frac = params.zone_hp_fraction
        if bool((frac <= 0).any()):
            raise ValueError(
                "zone.max_hp_fraction_per_second resolved to 0 for at least one env while "
                "zone.enabled is true, so the zone would deal no damage. If your config still "
                "sets the old flat `zone.dps`, rename it: the new key is a FRACTION of each "
                "entity's max HP per second (0.2 = dead in 5s), not HP/s."
            )
        if bool((frac > 1.0).any()):
            raise ValueError(
                f"zone.max_hp_fraction_per_second must be in (0, 1] (it is a FRACTION of max HP "
                f"per second, not flat HP/s), got max {float(frac.max())}"
            )
        if bool((params.zone_fraction_growth < 0).any()):
            raise ValueError("zone.fraction_growth_per_step must be >= 0")

    # Checked whether or not the zone is enabled: it is an observation setting, and a zero or
    # negative horizon would collapse zone.hero_margin_local to a constant column -- exactly the
    # "trained on a constant" failure configs/agent_obs_deploy*.yaml exist to prevent.
    if cfg.zone_margin_horizon_tiles <= 0:
        raise ValueError(
            f"zone.margin_horizon_tiles must be > 0 (it is the sensing horizon "
            f"zone.hero_margin_local is clamped to), got {cfg.zone_margin_horizon_tiles}"
        )

    if cfg.regen_enabled:
        if bool((params.regen_delay < 0).any()):
            raise ValueError("regen.delay_seconds must be >= 0")
        # The key is a FRACTION of max HP per second, so a flat HP figure put under it (400, say)
        # would be a full heal every tick. A missing key resolves to 0, no regen, which is allowed.
        frac = params.regen_fraction_per_second
        if bool((frac < 0).any()) or bool((frac > 1.0).any()):
            raise ValueError(
                f"regen.max_hp_fraction_per_second must be in [0, 1] (it is a FRACTION of max HP "
                f"per second, not flat HP/s), got min {float(frac.min())} / max {float(frac.max())}"
            )
