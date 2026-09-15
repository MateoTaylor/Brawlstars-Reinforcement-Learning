"""Config loading and per-env parameter tensors. See BRAWL_SIM_BUILD_PLAN.md Step 3/4.

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
# Safe despite config.py being imported by nearly everything: core.projectiles' only imports are
# torch, ..constants and its sibling core modules (geometry/stats/terrain), none of which import
# config -- there is no cycle to create here.
from .core.projectiles import MAX_SPLITS

_MISSING = object()
# Distinct from _MISSING on purpose. `_dget(d, k, default=_MISSING)` means "RAISE if absent"
# (apply_randomization relies on that, via try/except KeyError); `_ABSENT` is a value _dget will
# happily RETURN, which is what lets load_config tell "the file didn't mention this field" apart
# from "the file set it to something". See load_config for why that distinction matters.
_ABSENT = object()

# Kind 0 is always the hero; this order matches brawl_sim.constants.Kind's values 0..4 and
# is also the order used for the leading dim of every per-kind SimParams tensor's K axis.
KIND_YAML_NAMES = ("hero_mortis", "bot_sniper", "bot_artillery", "bot_melee", "bot_rifle",
                   "bot_edgar", "bot_spike", "bot_bull")
# BOT_KINDS order (Kind values 1..7), used for entities.enemy_type_weights / fixed_enemy_types.
# "edgar", "spike" and "bull" are brawler names where the other four are role names -- see
# constants.Kind.
ARCHETYPE_SHORT_NAMES = ("sniper", "artillery", "melee", "rifle", "edgar", "spike", "bull")
# constants.Person order (values 0..4), used for bots.personality_weights. Bot PERSONALITY is
# orthogonal to bot ARCHETYPE: both are drawn independently per entity on every reset.
PERSON_SHORT_NAMES = ("rush", "camper", "hunter", "trapper", "kite")

_KNOWN_MAP_NAMES = (
    "blank", "open", "bushy", "walled", "skull_creek", "feast_or_famine",
    "scorched_stone", "island_invasion",
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
    obs_include_world_grid: bool = True
    obs_include_privileged: bool = True
    bots_break_boxes: bool = True
    bots_collect_cubes: bool = True
    bots_avoid_zone: bool = True
    bots_fight_each_other: bool = True
    # --- bot personalities (Step 41). See bots/personality.py.
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
    zone_mode: str = "rect"
    # Sensing horizon for zone.hero_margin_local ONLY -- it changes no dynamics, just how far that
    # one observation field can see. Must equal the clamp the deployed gas estimator uses; see
    # BRAWL_DEPLOYMENT_DESIGN.md 9.14.
    zone_margin_horizon_tiles: float = 10.0
    iframes_block_zone: bool = False
    regen_enabled: bool = True
    drop_victim_cubes: bool = True
    # How far from its crate a broken crate's cube lands: a uniform distance in [min, max] tiles,
    # in a uniform direction (core/boxes.resolve_broken_boxes). 0 and 0, the defaults, keep the
    # old rule of landing on the crate itself.
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
        """(move bins + idle, attack). The attack dimension is 3-valued as of Step D2:
        0 = nothing, 1 = attack, 2 = super. Widened rather than joined by a third dimension so the
        action stays (N,2) -- see core/hero.action_mask."""
        return (self.n_move_bins + 1, 3)

    @property
    def ray_steps(self) -> int:
        return int(math.ceil(self.max_ray_tiles / self.los_step_tiles))

    @property
    def agent_dt(self) -> float:
        """Seconds of simulated time per AGENT DECISION. `dt` is the SIM tick; one decision
        covers `action_repeat` of them (env.py's `_run_decision`), so the agent's decision rate
        is `1 / agent_dt` Hz -- 4 Hz at the default `dt=0.05, action_repeat=5`."""
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


# (dotted YAML path, EnvConfig field name, type coercion)
_ENV_CONFIG_FIELDS = (
    ("world.map_h", "map_h", int),
    ("world.map_w", "map_w", int),
    ("world.maps", "map_names", _tuple_transform),
    ("world.map_selection", "map_selection", str),
    ("world.fixed_map", "fixed_map", str),
    ("view.height", "view_h", int),
    ("view.width", "view_w", int),
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
    ("observation.include_world_grid", "obs_include_world_grid", bool),
    ("observation.include_privileged", "obs_include_privileged", bool),
    ("bots.break_boxes", "bots_break_boxes", bool),
    ("bots.collect_cubes", "bots_collect_cubes", bool),
    ("bots.avoid_zone", "bots_avoid_zone", bool),
    ("bots.fight_each_other", "bots_fight_each_other", bool),
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
    ("zone.mode", "zone_mode", str),
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
    """A field absent from the YAML keeps its EnvConfig default rather than raising.

    This is what the "keep the EnvConfig default" branch below always meant to do, but it was
    unreachable until Step 41: it was written as `_dget(raw, dotted, default=_MISSING)`, and
    `_MISSING` is precisely the sentinel that makes `_dget` RAISE instead of return, so every
    config file was in fact required to spell out all ~40 fields. Adding a field to
    `_ENV_CONFIG_FIELDS` therefore broke every previously-valid YAML that didn't have it --
    which is exactly what Step 41's nine new `bots.*` keys did to the hand-written fragments in
    tests/test_config.py. The dataclass defaults exist to be the fallback; now they are one."""
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
    default.yaml + configs/brawlers.yaml, already merged). Returns a new dict; `spec` is not
    mutated. Not one of the Step 3 pseudocode's five named functions -- it's the minimal glue
    needed to turn a loaded RandomizationSpec into something build_params/resample_params can
    read, since neither of those takes a randomization argument separately.

    mode="additive" (default) replaces the base value outright: final ~ U(low, high).
    mode="multiplicative" scales the base value: final ~ U(low, high) * base, so the base
    must resolve to a plain scalar (it's the nominal being jittered, not itself a range).
    """
    out = spec
    for dotted, r in randomization.items():
        if "." not in dotted:
            raise KeyError(f"randomization key must be 'section.field', got {dotted!r}")
        # `mode` defaults to "additive" here, matching _parse_range_entry's own default, so a
        # HAND-WRITTEN spec dict works: README/BrawlVecEnv both advertise passing one straight to
        # `BrawlVecEnv(..., randomization={"bot_sniper.base_hp": {"low": ..., "high": ...}})`,
        # but only `load_randomization` filled `mode` in, so that documented form used to raise
        # KeyError on this line.
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
    # describes only that kind's split shards). 0 keeps the old constant-speed arc.
    ("proj_flight_seconds", "proj_flight_seconds", _F32),
    ("proj_radius", "proj_radius", _F32),
    ("proj_count", "proj_count", _I64),
    ("proj_spread_rad", "proj_spread_rad", _F32),
    ("aoe_radius", "aoe_radius", _F32),
    # The ring a detonating shell splits into (core/projectiles.SPLIT_DIR_TABLE). All three
    # default to 0, which means "no split" -- that is what every non-splitting kind resolves to,
    # and what a config written before these fields existed keeps doing.
    ("split_distance", "split_distance", _F32),
    ("split_damage_fraction", "split_damage_fraction", _F32),
    # How many shards, evenly spaced around the circle: 4 is Grom's world-axis cross, 6 is Spike's
    # hexagonal star. The DIRECTIONS are derived from this rather than configured -- see
    # core/projectiles._ring. Capped at core/projectiles.MAX_SPLITS by validate(), because the
    # (N,P,K) grid _spawn_splits allocates against is sized by that constant and arms beyond it
    # would be dropped in silence.
    ("split_count", "split_count", _I64),
    # How long a shard takes to fly `split_distance` (Step C4). The shard's SPEED is derived from
    # these two, so the config states the observable duration rather than a speed someone has to
    # solve for. 0 falls back to `proj_speed`, the pre-C4 constant-speed behavior.
    ("split_seconds", "split_seconds", _F32),
    ("attack_arc_rad", "attack_arc_rad", _F32),
    # Swept melee (core/melee_sweep.py, Step C2). `hitscan_count` <= 1 -- which is what every kind
    # that does not set it resolves to -- is the single instantaneous cone melee always had, so
    # these default to "not swept" and cost nothing.
    ("hitscan_count", "hitscan_count", _I64),
    ("hitscan_sweep_rad", "hitscan_sweep_rad", _F32),
    # MELEE lifesteal (Edgar): heal this fraction of the damage a melee cone lands on PLAYERS.
    # 0 -- what every other kind resolves to -- means no lifesteal and costs one masked multiply.
    #
    # Named for the damage source it actually reads rather than the generic `lifesteal_fraction`,
    # because it is wired into `env._attack_phase` only: a kind that set it without a melee cone
    # would silently heal nothing, and a name that cannot describe that failure is a name that
    # lies. Extending it to dash or projectile damage means reading it in those phases too (both
    # already produce the same (N,E,E) attacker x victim matrix this needs) and renaming it then.
    #
    # Deliberately NOT `super_heal`, which is the OTHER lifesteal in the sim: that one is a flat
    # HP figure per player a super connects with (Mortis, Step D2), independent of damage dealt.
    # Two different mechanics that would fight over one field.
    ("melee_lifesteal_fraction", "melee_lifesteal_fraction", _F32),
    # Lingering damage areas (constants.ProjClass.HAZARD, Step C3b). A projectile whose owner's
    # kind sets on_hit_area_radius > 0 leaves a stationary sphere where it dies. All default to 0,
    # which means "leaves nothing" -- what every kind but Brock resolves to.
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
    # Long dash (Step D1): after `long_dash_seconds` without attacking, the next dash reaches
    # `long_dash_multiplier` times as far. 0 seconds means "this kind has no long dash", which is
    # what every kind but the hero resolves to.
    ("long_dash_seconds", "long_dash_seconds", _F32),
    ("long_dash_multiplier", "long_dash_multiplier", _F32),
    # SUPER (Step D2). Per-kind, and deliberately not hero-only: `super_charge_hits: 0` means "this
    # kind has no super", which is what every bot resolves to today -- but giving a bot one is a
    # brawlers.yaml block plus a `super_fire` decision in its combat rule, with no plumbing change.
    ("super_charge_hits", "super_charge_hits", _I64),
    ("super_range", "super_range", _F32),
    ("super_damage", "super_damage", _F32),
    ("super_heal", "super_heal", _F32),
    ("super_proj_speed", "super_proj_speed", _F32),
    ("super_radius", "super_radius", _F32),
    # --- FIRE RULE (Step E1). These four numbers plus `aim_model` below are the entirety of what
    # used to be bots/{sniper,artillery,melee,rifle}.py: adding a sixth brawler is now a
    # brawlers.yaml block, not a fifth Python module plus a dispatcher entry. See
    # bots/combat_rules.py, which reads exactly these.
    #
    # Every one of them is off/neutral at 0, which is what a kind that never mentions them gets:
    #   fire_needs_los 0                  fire on a bush-only target through a wall (artillery,
    #                                     melee -- melee_hitscan does its own LOS check in phase 6)
    #   fire_range_fraction 0             read as 1.0, i.e. no restriction beyond attack_range
    #   fire_lateral_speed_limit 0        never hold fire on a fast lateral mover
    # The fifth axis, "does this kind need its target inside a swing cone", is NOT a field: it is
    # exactly `attack_arc_rad > 0`, which is already what cone_ray_tiles and combat.melee_hitscan
    # use to mean "this kind swings a cone". A second field saying the same thing could disagree
    # with the first.
    ("fire_needs_los", "fire_needs_los", _I64),
    # Hold fire past this fraction of attack_range. Shelly's 0.9: her fan diverges too much at the
    # rim to land. 0 means 1.0 -- see the block comment above.
    ("fire_range_fraction", "fire_range_fraction", _F32),
    # Hold fire on a target crossing faster than this many tiles/s, but only past
    # `fire_lateral_hold_range_fraction` of attack_range (close in, the fan is tight enough to
    # land anyway). 0 disables the whole rule, which is why the limit is checked for > 0 rather
    # than just compared against: a 0 limit read literally would hold fire on ANY moving target.
    ("fire_lateral_speed_limit", "fire_lateral_speed_limit", _F32),
    ("fire_lateral_hold_range_fraction", "fire_lateral_hold_range_fraction", _F32),
    # Preferred engagement distance as a fraction of this kind's own attack_range -- what the KITE
    # personality holds. Was bots/policy.RANGE_FRACTION_BY_KIND, a Python tuple with an
    # `assert len(...) == N_KINDS` that a sixth brawler would have tripped. Read by
    # bots/policy.targeting into Targeting.desired_range; the hero's 0 is never read (its movement
    # comes from decode_action).
    ("desired_range_fraction", "desired_range_fraction", _F32),
)

# (SimParams attribute, brawlers.yaml leaf key, enum class, default member name) -- one column per
# KIND_YAML_NAMES entry, same (N,K) i64 shape as PER_KIND_FIELDS produces. These are enum NAME
# STRINGS in the config ("SNIPER_BOLT", "LEAD"), so they resolve to a constant column of that
# member's value rather than through _resolve_value's scalar-or-range sampling -- an enum has no
# meaningful midpoint to randomize between.
#
# A misspelled name raises KeyError from the enum lookup naming the bad string, which is the
# behavior proj_kind has always had and the reason these get a default MEMBER rather than a
# default 0: `Proj.NONE` and `AimModel.DIRECT` are both deliberately 0, but saying so by name
# keeps the table readable against enums where 0 is not the neutral member.
PER_KIND_ENUM_FIELDS = (
    ("proj_kind", "proj_kind", Proj, "NONE"),
    ("aim_model", "aim_model", AimModel, "DIRECT"),
)

# (SimParams attribute, dotted spec path, dtype). zone_start_time is handled separately: it's
# derived from zone.start_fraction * max_episode_steps * dt (Step 23), not a direct copy.
PER_ENV_FIELDS = (
    ("enemy_hp_mult", "entities.enemy_hp_mult", _F32),
    ("enemy_damage_mult", "entities.enemy_damage_mult", _F32),
    ("unit_radius", "entities.unit_radius", _F32),
    ("pickup_radius", "cubes.pickup_radius", _F32),
    # FLAT HP per cube (400), not a fraction of base_hp -- see core/stats.effective_max_hp.
    # Renamed, not reinterpreted: a config still carrying `hp_bonus_per_cube: 0.15` would
    # otherwise resolve to 0 here and silently give cubes no HP at all. See _REMOVED_KEYS.
    ("cube_hp_flat", "cubes.hp_per_cube", _F32),
    ("cube_damage_bonus", "cubes.damage_bonus_per_cube", _F32),
    ("max_cubes", "cubes.max_cubes", _I64),
    ("cubes_per_box", "cubes.cubes_per_box", _I64),
    ("cubes_on_kill_base", "cubes.cubes_on_kill_base", _I64),
    ("box_hp", "boxes.hp", _F32),
    ("n_boxes", "boxes.n_boxes", _I64),
    ("zone_step_seconds", "zone.step_seconds", _F32),
    ("zone_tiles_per_step", "zone.tiles_per_step", _F32),
    # A FRACTION of each entity's own max HP per second, not a flat HP/s (which is what
    # `zone.dps` used to be). Renamed rather than reinterpreted, for the same reason
    # `regen.per_second` -> `regen.max_hp_fraction_per_second` was: a config still carrying
    # `dps: 1000.0` under the OLD key would otherwise be silently ignored, resolve to 0, and
    # produce a zone that deals no damage at all -- see validate(), which rejects exactly that.
    ("zone_hp_fraction", "zone.max_hp_fraction_per_second", _F32),
    ("zone_fraction_growth", "zone.fraction_growth_per_step", _F32),
    ("bush_reveal_radius", "perception.bush_reveal_radius", _F32),
    ("reveal_after_attack", "perception.reveal_after_attack", _F32),
    ("regen_delay", "regen.delay_seconds", _F32),
    # A FRACTION of the entity's own max HP per second, not a flat HP/s (which is what
    # `regen.per_second` used to be). Renamed rather than reinterpreted in place: a config still
    # carrying the old `per_second: 400.0` would otherwise be read as "heal 400x max HP per
    # second", i.e. instant full heal, with nothing anywhere to flag it.
    ("regen_fraction_per_second", "regen.max_hp_fraction_per_second", _F32),
)


def _spec_upper(value) -> float:
    """The largest value a scalar-or-range spec entry can ever resolve to. A {low, high} range
    resolves somewhere in [low, high] on every resample, so `high` is the static upper bound."""
    if isinstance(value, dict):
        return float(value["high"])
    return float(value)


def cone_ray_tiles(spec: dict) -> float:
    """A STATIC upper bound (in tiles) on how far a melee CONE attack can reach, across every
    kind and every value its randomization range can produce.

    Exists so `combat.melee_hitscan` can hand `terrain.march` a per-call ray budget instead of
    paying the full `cfg.ray_steps` (48 at the defaults) for a 3-tile swing -- measured at
    4.15 -> 1.95 ms/tick at n_envs=1024. See bot_overhaul.md Step A1.

    **Derived from the SPEC, not from sampled `params`.** `params.attack_range.max()` would be a
    host sync, and `resample_params` runs inside `step()` via autoreset, so there is no safe
    moment to take one. Reading `high` off the spec instead gives a bound that is valid for every
    reset this env will ever do, computed once in pure Python.

    **Only kinds that can actually swing a cone count.** A kind with `attack_arc_rad == 0` can
    never satisfy `geo.point_in_cone`, so `melee_hitscan` discards its rows regardless of what
    the LOS check says -- including it would raise the bound to the longest RANGED range (8.67)
    and give back most of the saving for nothing.
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


class SimParams:
    """Per-env numerics. Every field is a tensor with a leading (N,) dim, even when the
    config value is a scalar -- this keeps downstream code uniform (N08). Per-kind fields are
    (N, K); per-env fields are (N,).

    **`SCALAR_FIELDS` is the explicit exception list to that rule**, and it is deliberately an
    enumerated allowlist rather than a convention: a field here is a plain Python number, so
    anything that walks `__slots__` expecting tensors (`.clone()`, row indexing, `torch.where`)
    breaks on it. Naming them means such a walk can skip exactly these and still fail loudly on a
    non-tensor field someone adds by accident later -- see
    tests/test_spawn.py::test_sim_params_fields_are_tensors_except_the_declared_scalars, which
    pins both halves.

    A field earns a place here only by being STRUCTURAL: something that sizes a tensor dimension
    or drives Python control flow, and therefore can never live on the device without forcing a
    host sync in the hot path. `cone_ray_tiles` (the melee LOS ray budget, see the module-level
    function) is the first and currently only one. Scalars here are spec-derived and constant for
    the env's whole lifetime -- `resample_params` does not touch them, because the spec they come
    from cannot change mid-run.
    """

    SCALAR_FIELDS = ("cone_ray_tiles", "attack_ray_tiles")

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

    return params


# Keys this schema used to read, and what replaced them. Every entry is a rename where the new
# key means something NUMERICALLY DIFFERENT from the old one, so a config left on the old spelling
# does not fail -- it resolves the new key to `_dget`'s default of 0 and silently disables the
# mechanic (or, worse, keeps a number that now means something else). That is the single most
# expensive failure mode this project has: invisible at construction, and only recognisable
# 300M steps later as "the agent learned something strange".
#
# Checked in `build_params` rather than in `_resolve_all`, because `_resolve_all` also runs from
# `resample_params` -- i.e. inside `step()` on every autoreset -- and a Python dict walk has no
# business there. The spec cannot change mid-run, so once at construction is exactly enough.
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

    Replaces (Step B2) a bound that was wrong in both directions:
    `n_entities * max(proj_count) * 4` (4 being the fixed arm count of the day). That demanded 200
    slots for a 10-entity roster by
    assuming every entity fires a max-width volley AND that every projectile splits -- no kind
    does both (only artillery splits, and it fires proj_count 1) -- while never modelling the
    thing that actually fills the buffer: **projectiles persist for many ticks**. It would have
    rejected a perfectly workable config and accepted a broken one.

    What actually fills the buffer is burst x lifetime:

        lifetime      = proj_flight_seconds, or attack_range / proj_speed for a normal shot
        volleys_alive = min(max_ammo, floor(lifetime / attack_cooldown) + 1)
        per_entity    = volleys_alive * proj_count * (split_count if this kind splits else 1)

    i.e. how many volleys a full clip can put in the air before the first one expires. The roster
    worst case is the hero plus `n_enemies` copies of the most demanding BOT kind, since
    `randomize_enemy_types` can legitimately draw an all-one-archetype lobby (and
    `fixed_enemy_types` can ask for one outright).

    **This is a bound, not a prediction.** Measured peaks run well under it -- bots need a target
    in range, are staggered by `decision_period_ticks`, and die -- but the buffer has to be sized
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
    # Each kind is budgeted against its OWN ring width, not core/projectiles.MAX_SPLITS: charging
    # every splitting kind for the widest ring in the roster would have made adding Spike's 6 arms
    # silently raise the buffer Grom's 4-arm cross is required to reserve.
    split_count = per_kind_max(params.split_count)
    splits = torch.where(per_kind_max(params.split_distance) > 0, split_count, torch.ones_like(split_count))

    flight_life = torch.where(flight > 0, flight, attack_range / speed)

    # **DWELL: how long a shot keeps occupying slots AFTER its own projectile ends.** Two mechanics
    # produce it, and they are treated as one concept rather than two special cases -- a third
    # would otherwise want a third branch:
    #   - a HAZARD (Step C3b) lingers on_hit_area_ticks * on_hit_area_interval seconds
    #   - SHARDS (Step C4) fly for split_seconds after the shell detonates
    # In both cases the successor INHERITS the parent's slot (`_spawn_hazards` / `_spawn_splits`
    # both run after prj_alive is updated), so the slot COUNT is already covered by `splits` above
    # -- what dwell extends is the RESIDENCY, which is what decides how many of a kind's shots are
    # in the buffer at once. They are combined with max() rather than summed: no kind does both,
    # and a hypothetical one would chain them rather than overlap.
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
    # subtly wrong behavior for an entire training run, which is the worst failure mode available:
    # by the time it shows up as "the agent learned something strange", 300M steps have burned.
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
        # There used to be a second check here rejecting `hitscan_sweep_rad == 0` alongside
        # `hitscan_count > 1`, on the grounds that identical-angle sub-swings are "a slower single
        # swing dealing count x damage, not a sweep". **Edgar is the counterexample that retired
        # it**: his attack is deliberately two hits 0.25 s apart at ONE angle, and it is not
        # equivalent to a single double-damage swing, for three reasons that all live in
        # core/melee_sweep and core/combat rather than in the angle:
        #
        #   - Each sub-swing re-tests `point_in_cone` and its own LOS march at the moment it
        #     lands, against a target that has had 0.25 s to move. Half a combo whiffing because
        #     the victim walked out is the mechanic, not a rounding error.
        #   - Each sub-swing lifesteals separately (`melee_lifesteal_fraction`), so landing one
        #     hit of two heals half as much.
        #   - The victim's i-frames are re-checked per sub-swing, so a dash timed into the gap
        #     eats one hit and not the other.
        #
        # The mistake the check was built to catch -- "set hitscan_count, forgot the sweep" -- is
        # now indistinguishable from a legitimate config, so it cannot be caught here. The
        # interval check above still catches the failure that IS silent (dropped swings).

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

        # --- fire rule (Step E1). Each of these is a brawlers.yaml block that loads without
        # complaint and then produces a bot that quietly never shoots, or shoots wrong, for a
        # whole training run -- the failure mode the four hand-written archetype modules used to
        # make impossible by construction and that moving the rules into data reintroduces.
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
        # 0 is legitimate for the hero (its movement never reads this) and a mistake for a bot: a
        # KITE bot of that kind would hold range 0, i.e. charge to point blank, which is the exact
        # opposite of the behavior it drew. It is the one fire-rule field with no usable default,
        # so it is the one a new brawler block MUST name.
        #
        # Gated on attack_range because the field is a FRACTION of it: a kind with no attack range
        # is not a functioning bot at all (it can never pass fire_gate), so demanding a fraction of
        # zero would only reject hand-built partial specs without protecting anything.
        if (k != int(Kind.HERO_MORTIS) and desired <= 0
                and float(params.attack_range[:, k].max()) > 0):
            raise ValueError(
                f"{kind_name}: attack_range is set but desired_range_fraction is 0, so a KITE bot "
                f"of this kind would hold range 0 -- it would charge to point blank instead of "
                f"keeping distance. Set the fraction of attack_range this kind fights at."
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
        # An ENABLED zone that deals no damage is always a mistake, and it is the exact shape a
        # config left on the pre-B3 `zone.dps: 1000.0` spelling takes: the old key is simply not
        # read any more, the new one resolves to _dget's default of 0, and the zone silently
        # becomes inert. That failure mode is invisible at construction and shows up much later as
        # "episodes never resolve" -- which is the very symptom the proportional zone exists to
        # fix, so it would be maximally confusing to debug. Fail loudly here instead.
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
        # Catches a config left on the pre-rename `regen.per_second: 400.0` spelling: the new key
        # is a FRACTION of max HP, so 400 would mean a full heal every tick. A missing key resolves
        # to 0 (no regen), which is safe but worth naming too.
        frac = params.regen_fraction_per_second
        if bool((frac < 0).any()) or bool((frac > 1.0).any()):
            raise ValueError(
                f"regen.max_hp_fraction_per_second must be in [0, 1] (it is a FRACTION of max HP "
                f"per second, not flat HP/s), got min {float(frac.min())} / max {float(frac.max())}"
            )
