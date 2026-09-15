"""AgentObsSpec: turns the full observation package (Step 25/26) into the AGENT's observation
-- a flat `gymnasium.spaces.Dict` of `Box` subspaces, with fairness gating applied. See
BRAWL_SIM_BUILD_PLAN.md Step 32.

Requires `gymnasium` (the project's "sb3" extra) -- unlike every `core/` module before it, this
one is not on the pure-simulation import path (nothing in Steps 1-31 imports `obs_select`), so
this is an acceptable, opt-in dependency, not a break of `core/`'s previous gymnasium-free
discipline.

**Shapes are computed, not authored.** `configs/agent_obs.yaml` declares `fields`/`per_entity`/
`max_slots`/`view_channels` per group; `load_agent_spec(path, cfg)` resolves each group's
`GroupSpec.shape` from those against a real `cfg` (via `obs_schema.obs_spec(cfg)`'s already-
resolved field shapes) -- the yaml itself never states a shape.

**Two structurally different kinds of "per-entity" group exist,** both flagged `per_entity=True`:
`entities.*`-prefixed fields (the "enemies" group in the default spec) drop hero slot 0 (`E - 1`
rows: entity axis has a hero, the agent obs never should) and are gated by
`entities.revealed_to_hero` under `fair: true`; every other axis (`projectiles.*`, and
generically any other per-entity-shaped group a spec author adds) keeps every slot and, if
`fair: true`, is gated by `<prefix>.in_view` when that field exists in the schema (silently
ungated otherwise -- there is no schema-wide convention for what "fair" means on an axis this
module wasn't told about, e.g. `boxes.*`/`pickups.*`; a future step can extend
`_FAIRNESS_MASK_FIELD` when a concrete need shows up). `_entity_prefix` derives this axis from
the group's own `fields` (all of one group's fields must share one dotted prefix) rather than
from the group's `name`, so nothing here is hardcoded to the literal string "projectiles".

**`max_slots` (the projectile group's top-K-by-`time_to_closest` selection) is likewise generic**
by prefix, not hardcoded to "projectiles" -- any per-entity group may set it, as long as its
prefix has a `<prefix>.time_to_closest` field to sort by (checked at load time). Selected slots
beyond however many are actually alive are zero-padded (multiplied by the gathered `alive`
mask) -- this happens unconditionally, independent of `fair`, since it's slot-padding, not
fairness. **This is the one group shape that is NOT index-stable tick to tick** -- unlike every
other group, a given output row is not "the same real entity" across steps.

**`normalize: true` is a real but intentionally narrow transform**, not full z-scoring. Four
units are covered, each by one constant chosen the same way -- so that the largest value the sim
can actually produce lands just inside 1.0, with headroom:

    tiles    / max(cfg.map_w, cfg.map_h)   the scale `*_norm` fields already use, applied to the
                                           raw-tile fields with no `_norm` counterpart
    tiles/s  / _NORM_SPEED_SCALE   (20.0)
    hp       / _NORM_HP_SCALE   (20000.0)
    count    / _NORM_COUNT_SCALE   (20.0)

`seconds`/`fraction`/`bool`/`onehot`/`unitless`/`radians`/`enum`/`index` still pass through
unnormalized. Fractions and bools are already bounded; the rest have a bounded `_frac`/`_whole`/
`onehot` alternative a spec author can pick instead. Grid (`uint8`) groups are never normalized
-- Hard Constraint 3 fixes their space at `[0, 255]` regardless.

**`hp` and `count` were added when the deployed spec dropped `hp_frac`** (see
`configs/agent_obs_deploy.yaml`). Power cubes make max HP unobservable to CV, so the deployed
agent reads the HP *numeral* and there is no denominator to divide by -- which turns two fields
that had been bounded fractions into raw magnitudes of 20500 and 9. Both now have a divisor
rather than a special case. Note this also rescales `projectiles.damage` in `agent_obs.yaml`,
the only other `hp`-unit field in any shipped spec, and the `cubes` fields in `agent_obs.yaml`/
`agent_obs_lowinfo.yaml`: those specs' observations change value (not width), so a checkpoint
trained under them before this change cannot be resumed after it.

One limitation worth stating rather than discovering: `count` also covers CUMULATIVE counters
(`hero.shots_fired`, `hero.kills`), which no constant bounds. Dividing them by 20 makes them
smaller, not bounded. They are not in any shipped spec -- they exist for `info`/eval logging --
and if one starts using them it needs a bounded alternative, not a bigger divisor.

**Normalizing a counter cannot disturb the reward, and that is worth knowing before someone
adds `hero.kills` to a spec and goes looking for the reward term to compensate.** Nothing here
touches `full_obs`: `build_agent_obs` gathers into its own `out_buffers` and `_normalize`
returns a new tensor, so the dict `env.py` builds `info` from keeps raw units. And the kill
reward does not read a counter at all -- `training/reward.py` takes `info["kills_tick"]`, which
`core/events.py` recomputes each tick from `newly_dead & (ent_last_hit_by >= 0)` precisely
because the cumulative `ent_kills` has no "before this tick" snapshot to diff against. The two
paths never meet.
"""
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from . import obs_schema

_MAX_GROUPS = 6  # Hard Constraint 4: each group is a separate device->host transfer (Step 33)

# The 12 `_build_grid` channels (core/observation.py), by name -> index. "enemy_any"/
# "enemy_hidden" are deliberately excluded from `fair: true` specs (checked at load time) --
# only "enemy_revealed" is a legitimate agent-facing channel under fairness gating.
_CHANNEL_INDEX = {
    "blocks_unit": 0, "blocks_projectile": 1, "is_bush": 2, "is_water": 3, "in_zone": 4,
    "enemy_any": 5, "enemy_revealed": 6, "enemy_hidden": 7, "hero": 8, "box": 9,
    "pickup": 10, "projectile": 11,
}
_UNFAIR_GRID_CHANNELS = ("enemy_any", "enemy_hidden")

_HERO_AXIS_PREFIX = "entities"  # the only per-entity axis with a hero slot to drop
_FAIRNESS_MASK_FIELD = {"entities": "entities.revealed_to_hero", "projectiles": "projectiles.in_view"}

_NORM_DIVISOR_TILES = lambda cfg: float(max(cfg.map_w, cfg.map_h))
# Reference velocity scale for `normalize: true`'s "tiles/s" fields -- grounded in
# configs/brawlers.yaml's actual stat ranges, not an arbitrary guess: dash speed tops out
# ~16.7 tiles/s (dash_distance / dash_duration, e.g. 5.0 / 0.30), the fastest projectile listed
# is 14 tiles/s (bot_sniper). 20.0 keeps every observed velocity comfortably inside (-1, 1) with
# headroom for randomization, while raw entity move_speed (~2.3-2.6 tiles/s) lands at a small,
# well-behaved ~0.12.
_NORM_SPEED_SCALE = 20.0

# Reference HP scale for `normalize: true`'s "hp" fields. The largest HP the sim can produce is
# an enemy at the roster's highest base (10000, bot_melee/bot_bull) carrying `cubes.max_cubes: 16`
# at `cubes.hp_per_cube: 400` flat, scaled by the top of `enemy_hp_mult`'s randomization range:
# (10000 + 6400) * 1.25 = 20500. The hero cannot exceed 8000 + 6400 = 14400 (enemy_hp_mult never
# touches the hero -- see core/stats.effective_max_hp). 20000 puts that ceiling at 1.03 and a
# fresh Mortis at 0.40, the same "small but well-behaved" placement _NORM_SPEED_SCALE gives a
# walking entity. Re-derive it, don't nudge it, if base_hp or the cube numbers change.
_NORM_HP_SCALE = 20000.0

# Reference scale for `normalize: true`'s "count" fields. The largest BOUNDED count in
# obs_schema is `cubes` at `cubes.max_cubes: 16`; `meta.n_enemies_alive` reaches cfg.n_enemies
# (9) and `hero.ammo_whole` reaches max_ammo (3). 20.0 covers all three with headroom -- the
# same number as _NORM_SPEED_SCALE by coincidence, not by sharing a meaning.
_NORM_COUNT_SCALE = 20.0

_tensor_cache: dict = {}


def _cached_tensor(key, values: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    """Memoizes a small constant tensor by (key, device) -- same shape as reward.ZeroReward's
    own cache, used here so per-call grid-channel-selection / normalization never does a
    `torch.tensor(python_list, device=...)` host sync (see core/geometry.vec2's docstring for
    why that specific call is a real sync)."""
    cache_key = (key, device)
    cached = _tensor_cache.get(cache_key)
    if cached is None:
        cached = torch.tensor(values, dtype=dtype, device=device)
        _tensor_cache[cache_key] = cached
    return cached


@dataclass(frozen=True)
class GroupSpec:
    name: str
    dtype: str                                    # "float32" | "uint8"
    shape: tuple                                   # resolved against cfg by load_agent_spec
    fields: tuple = ()
    per_entity: bool = False
    max_slots: int | None = None
    view_channels: tuple | None = None
    channel_idx: tuple | None = None               # resolved view-channel indices, grid groups only
    entity_prefix: str | None = None                # shared dotted prefix of `fields`, per_entity groups only
    norm_scale: tuple | None = None                 # per-output-column normalize divisor, if spec.normalize


@dataclass(frozen=True)
class AgentObsSpec:
    fair: bool
    groups: tuple
    normalize: bool


def _entity_prefix(fields: tuple) -> str:
    prefixes = {f.split(".", 1)[0] for f in fields}
    if len(prefixes) != 1:
        raise ValueError(f"a per_entity group's fields must all share one dotted prefix, got {sorted(prefixes)}")
    return next(iter(prefixes))


def _field_width(shape: tuple, per_entity: bool) -> int:
    """shape is a resolved (from obs_schema.obs_spec) field shape, e.g. ("N","E",5) or ("N",2).
    Strips the ("N",) or ("N","<axis>") prefix and returns the product of what's left (1 for a
    bare scalar field)."""
    trailing = shape[2:] if per_entity else shape[1:]
    width = 1
    for d in trailing:
        width *= d
    return width


def load_agent_spec(path, cfg) -> AgentObsSpec:
    raw = yaml.safe_load(Path(path).read_text())
    fair = bool(raw["fair"])
    normalize = bool(raw["normalize"])
    spec_fields = obs_schema.obs_spec(cfg)

    groups = []
    seen_names = set()
    for g in raw["groups"]:
        name = g["name"]
        if name in seen_names:
            raise ValueError(f"duplicate agent_obs group name {name!r}")
        seen_names.add(name)
        dtype = g.get("dtype", "float32")
        if dtype not in ("float32", "uint8"):
            raise ValueError(f"group {name!r}: dtype must be 'float32' or 'uint8', got {dtype!r}")

        view_channels = g.get("view_channels")
        if view_channels is not None:
            groups.append(_load_grid_group(name, dtype, view_channels, fair, cfg))
            continue

        fields = tuple(g["fields"])
        for f in fields:
            if f.startswith("entities.privileged."):
                raise ValueError(
                    f"group {name!r}: field {f!r} is under entities.privileged and must NEVER "
                    "be exposed to the agent (this is R04's guardrail)"
                )
            if f not in spec_fields:
                raise ValueError(f"group {name!r}: unknown field {f!r} (not present in obs_spec(cfg))")

        per_entity = bool(g.get("per_entity", False))
        max_slots = g.get("max_slots")
        entity_prefix = _entity_prefix(fields) if per_entity else None
        if max_slots is not None:
            if not per_entity:
                raise ValueError(f"group {name!r}: max_slots is only meaningful when per_entity: true")
            sort_field = f"{entity_prefix}.time_to_closest"
            if sort_field not in spec_fields:
                raise ValueError(f"group {name!r}: max_slots requires a {sort_field!r} field to sort by")

        width = sum(_field_width(spec_fields[f].shape, per_entity) for f in fields)
        if per_entity:
            axis_full = spec_fields[fields[0]].shape[1]
            n_slots = max_slots if max_slots is not None else (
                axis_full - 1 if entity_prefix == _HERO_AXIS_PREFIX else axis_full
            )
            shape = (n_slots, width)
        else:
            shape = (width,)

        norm_scale = None
        if normalize and dtype == "float32":
            scale = []
            for f in fields:
                divisor = _norm_divisor(spec_fields[f].units, cfg)
                scale.extend([divisor] * _field_width(spec_fields[f].shape, per_entity))
            norm_scale = tuple(scale)

        groups.append(GroupSpec(
            name=name, dtype=dtype, shape=shape, fields=fields, per_entity=per_entity,
            max_slots=max_slots, entity_prefix=entity_prefix, norm_scale=norm_scale,
        ))

    if len(groups) > _MAX_GROUPS:
        raise ValueError(f"agent_obs spec has {len(groups)} groups; keep it <= {_MAX_GROUPS} (Hard Constraint 4)")
    if not groups:
        raise ValueError("agent_obs spec has no groups")

    return AgentObsSpec(fair=fair, groups=tuple(groups), normalize=normalize)


def _load_grid_group(name: str, dtype: str, view_channels, fair: bool, cfg) -> GroupSpec:
    for ch in view_channels:
        if ch not in _CHANNEL_INDEX:
            raise ValueError(f"group {name!r}: unknown view channel {ch!r}; valid: {sorted(_CHANNEL_INDEX)}")
    if fair:
        forbidden = [ch for ch in view_channels if ch in _UNFAIR_GRID_CHANNELS]
        if forbidden:
            raise ValueError(
                f"group {name!r}: fair:true forbids {forbidden} (use 'enemy_revealed', which "
                "already respects visibility -- 'enemy_any'/'enemy_hidden' would leak hidden "
                "enemy positions to the agent)"
            )
    channel_idx = tuple(_CHANNEL_INDEX[c] for c in view_channels)
    shape = (len(view_channels), cfg.view_h, cfg.view_w)
    return GroupSpec(name=name, dtype=dtype, shape=shape, view_channels=tuple(view_channels), channel_idx=channel_idx)


def _norm_divisor(units: str, cfg) -> float:
    if units == "tiles":
        return _NORM_DIVISOR_TILES(cfg)
    if units == "tiles/s":
        return _NORM_SPEED_SCALE
    if units == "hp":
        return _NORM_HP_SCALE
    if units == "count":
        return _NORM_COUNT_SCALE
    return 1.0


def agent_space(spec: AgentObsSpec, cfg) -> gym.spaces.Dict:
    spaces = {}
    for g in spec.groups:
        if g.dtype == "uint8":
            spaces[g.name] = gym.spaces.Box(low=0, high=255, shape=g.shape, dtype=np.uint8)
        else:
            spaces[g.name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=g.shape, dtype=np.float32)
    return gym.spaces.Dict(spaces)


def make_agent_obs_buffers(spec: AgentObsSpec, cfg, n_envs: int, device) -> dict:
    """Preallocated `out_buffers` for `build_agent_obs`, built ONCE by the caller (typically
    `wrappers/sb3_vecenv.py`, Step 33) and reused every call so `build_agent_obs` allocates
    nothing per call (steady-state memory doesn't grow -- Step 21's established reading of that
    phrase, see observation.py's own docstring)."""
    torch_dtype = {"float32": torch.float32, "uint8": torch.uint8}
    return {g.name: torch.zeros((n_envs, *g.shape), dtype=torch_dtype[g.dtype], device=device) for g in spec.groups}


def agent_obs_index_map(spec: AgentObsSpec, cfg) -> dict:
    """{group_name: {field_name: (start, end)}} column ranges within each group's flattened
    last dim -- documentation-only (used by dump_obs_schema.py to render docs/AGENT_OBS.md),
    not read by build_agent_obs itself (which reconstructs the same concatenation directly)."""
    spec_fields = obs_schema.obs_spec(cfg)
    out = {}
    for g in spec.groups:
        if g.view_channels is not None:
            out[g.name] = {f"view[{ch}]": (i, i + 1) for i, ch in enumerate(g.view_channels)}
            continue
        cols = {}
        pos = 0
        for f in g.fields:
            w = _field_width(spec_fields[f].shape, g.per_entity)
            cols[f] = (pos, pos + w)
            pos += w
        out[g.name] = cols
    return out


def _get_field(full_obs: dict, dotted: str) -> torch.Tensor:
    node = full_obs
    for part in dotted.split("."):
        node = node[part]
    return node


def _ensure_trailing_dim(t: torch.Tensor, per_entity: bool) -> torch.Tensor:
    min_dims = 3 if per_entity else 2
    return t.unsqueeze(-1) if t.dim() < min_dims else t


def _build_flat_group(full_obs: dict, g: GroupSpec) -> torch.Tensor:
    parts = [_ensure_trailing_dim(_get_field(full_obs, f), per_entity=False).to(torch.float32) for f in g.fields]
    return torch.cat(parts, dim=-1)


def _build_entity_group(full_obs: dict, g: GroupSpec, fair: bool) -> torch.Tensor:
    drop_hero = g.entity_prefix == _HERO_AXIS_PREFIX
    parts = []
    for f in g.fields:
        t = _ensure_trailing_dim(_get_field(full_obs, f), per_entity=True)
        if drop_hero:
            t = t[:, 1:]
        parts.append(t.to(torch.float32))
    out = torch.cat(parts, dim=-1)  # (N, n_slots, W)

    idx = None  # set below only when max_slots selected a K-of-full-axis subset of slots
    if g.max_slots is not None:
        ttc = _get_field(full_obs, f"{g.entity_prefix}.time_to_closest")
        alive = _get_field(full_obs, f"{g.entity_prefix}.alive")
        priority = torch.where(alive, ttc, torch.full_like(ttc, float("inf")))
        _, idx = torch.topk(priority, k=g.max_slots, largest=False, dim=-1)  # (N,K)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, out.shape[-1])
        out = torch.gather(out, 1, idx_exp)
        slot_alive = torch.gather(alive, 1, idx).to(torch.float32).unsqueeze(-1)
        out = out * slot_alive  # zero-pad slots with fewer than max_slots alive occupants

    if fair:
        mask_field = _FAIRNESS_MASK_FIELD.get(g.entity_prefix)
        if mask_field is not None:
            mask = _get_field(full_obs, mask_field)
            if drop_hero:
                mask = mask[:, 1:]
            if g.max_slots is not None:
                mask = torch.gather(mask, 1, idx)
            out = out * mask.to(torch.float32).unsqueeze(-1)

    return out


def _build_grid_group(full_obs: dict, g: GroupSpec) -> torch.Tensor:
    view = full_obs["view"]
    # Keyed on the channel tuple as well as the name, for the reason _normalize gives below. Every
    # spec calls its view group "grid", so once agent_obs_deploy3.yaml (10 planes) existed, building
    # it and then agent_obs_deploy.yaml (8) in one process handed the second the first's indices.
    idx = _cached_tensor(("grid_channels", g.name, g.channel_idx), g.channel_idx,
                         torch.int64, view.device)
    return view.index_select(1, idx)


def _normalize(raw: torch.Tensor, g: GroupSpec) -> torch.Tensor:
    # The scale tuple is part of the cache key, not just the group NAME. Keying on the name alone
    # was wrong the moment a second spec existed: `configs/agent_obs{,_lowinfo,_deploy}.yaml` all
    # have a group called "self", at 25/25/24 columns, so loading two of them in one process
    # handed the second the first's scale vector -- a broadcast error at best, and silently wrong
    # divisors whenever the widths happened to match. Tuples of floats are hashable, so this
    # costs one extra cache entry per distinct scale vector and nothing else.
    scale = _cached_tensor(("norm_scale", g.name, g.norm_scale), g.norm_scale,
                           torch.float32, raw.device)
    return raw / scale


def build_agent_obs(full_obs: dict, spec: AgentObsSpec, cfg, out_buffers: dict) -> dict:
    for g in spec.groups:
        if g.view_channels is not None:
            out_buffers[g.name].copy_(_build_grid_group(full_obs, g))
            continue
        raw = _build_entity_group(full_obs, g, spec.fair) if g.per_entity else _build_flat_group(full_obs, g)
        if g.norm_scale is not None:
            raw = _normalize(raw, g)
        out_buffers[g.name].copy_(raw)
    return out_buffers
