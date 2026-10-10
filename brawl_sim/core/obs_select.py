"""AgentObsSpec: turns the full observation (core/observation.py) into the AGENT's observation,
a flat `gymnasium.spaces.Dict` of `Box` subspaces with fairness gating applied. Needs
`gymnasium` (the project's "sb3" extra). brawl_deployment/perception/assemble.py calls the same
`build_agent_obs` live, so column order, normalization and row selection are shared with
deployment.

**Shapes are computed, not authored.** A spec yaml (`configs/agent_obs*.yaml`) names
`fields`/`per_entity`/`max_slots`/`slots`/`view_channels` per group; `load_agent_spec(path, cfg)`
resolves each `GroupSpec.shape` against a real `cfg` through `obs_schema.obs_spec(cfg)`.

**Per-entity groups** take their axis from the fields' one shared dotted prefix, never from the
group name. `entities.*` drops hero slot 0 (`E - 1` rows); any other axis keeps every slot.
Under `fair: true` only the prefixes in `_FAIRNESS_MASK_FIELD` are gated: `entities.*` by
`entities.revealed_to_hero`, `projectiles.*` by `projectiles.in_view`. Any other axis
(`boxes.*`, `pickups.*`) passes ungated, although it has an `in_view` field.

**`max_slots`** keeps the K rows with the smallest `<prefix>.time_to_closest` (it and
`<prefix>.rel_pos` must exist; checked at load). Ties are the common case, since everything
moving away from the hero reads 0, and go nearest the hero first (`_nearest_first`). Slots past
the alive count are zero-padded whatever `fair` says. The ranking covers every alive occupant,
on screen or not, so under `fair` an off-screen projectile still takes a slot and then has its
row blanked (user decision, 2026-09-24: noise in the same direction as the live detector's
misses; the live tracker never holds an off-screen projectile, so it fills those frames
differently). These rows are NOT index-stable: row k is not the same projectile step to step.

**`observation.projectile_static_speed`** (0 = off, 2.0 in configs/train.yaml): a projectile
slower than this reads as still, `vel` (0, 0) and `time_to_closest` 0. That is what the sim gives
a hazard, and what the live tracker reports for any track under its STATIC_TILES_S, where a
velocity is detector noise. It applies before `max_slots` ranks, so a crawling lob ranks among
the other zeros, nearest first, as it does live. Only those two fields change, the two velocity
fields the live tracker supplies, and `full_obs` itself keeps the true values (user decision,
2026-10-07: a timed lob aimed short crawls, and its straight-line time reached hundreds of
seconds, unscaled).

**`slots: tracked`** orders an `entities.*` group by the sim's tracker-style slots
(`core/slots.py`, via `full_obs["slots"]["entity"]`): row k is the entity holding slot k, zero
while the slot is empty, and the fairness mask is gathered the same way. Row identity follows
the live `EntityTracker` (a slot is taken on the `cfg.slots_promote_hits`-th consecutive
sighting and kept through `cfg.slots_max_misses` unseen decisions); deployment writes the
identity permutation (`assemble._put_entities`). Hero axis only, and never with `max_slots`.

**`normalize: true`** divides four units by one constant each, chosen so the largest value the
sim produces lands near 1.0 (the derivations sit beside the constants):

    tiles    / max(cfg.map_w, cfg.map_h)   the scale the `*_norm` fields already use
    tiles/s  / _NORM_SPEED_SCALE
    hp       / _NORM_HP_SCALE
    count    / _NORM_COUNT_SCALE

Every other unit passes through: fractions and bools are bounded, and the rest have a bounded
`_frac`/`_whole`/`onehot` alternative. uint8 (grid) groups are never normalized; their Box is
fixed at [0, 255]. `hp` needs a divisor because the deploy specs read the HP numeral: power
cubes make max HP unobservable to CV, so there is no denominator. Cumulative counters
(`hero.kills`, `hero.shots_fired`) are `count` too, and no divisor bounds them: a spec that
wants one needs a bounded alternative. Normalizing never touches `full_obs`, so `info` and the
reward keep raw units.
"""
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from . import obs_schema

_MAX_GROUPS = 6  # each group costs its own device->host copy (wrappers/sb3_vecenv.py)

# The 12 base `_build_grid` channels (core/observation.py), by name -> index; `channel_index(cfg)`
# appends "enemy_hist1".."enemy_histK" (K = cfg.history_frames). `fair: true` specs may not name
# "enemy_any"/"enemy_hidden" (checked at load); the enemy_hist planes are fair, since they only
# ever draw enemies that were revealed to the hero.
_CHANNEL_INDEX = {
    "blocks_unit": 0, "blocks_projectile": 1, "is_bush": 2, "is_water": 3, "in_zone": 4,
    "enemy_any": 5, "enemy_revealed": 6, "enemy_hidden": 7, "hero": 8, "box": 9,
    "pickup": 10, "projectile": 11,
}
_UNFAIR_GRID_CHANNELS = ("enemy_any", "enemy_hidden")
_HISTORY_CHANNEL_PREFIX = "enemy_hist"  # brawl_deployment/perception/grid.py imports it

_HERO_AXIS_PREFIX = "entities"  # the only per-entity axis with a hero slot to drop
_FAIRNESS_MASK_FIELD = {"entities": "entities.revealed_to_hero", "projectiles": "projectiles.in_view"}

_NORM_DIVISOR_TILES = lambda cfg: float(max(cfg.map_w, cfg.map_h))
# Divisor for `normalize: true`'s "tiles/s" fields, from configs/brawlers.yaml: Mortis's charged
# dash (5.34 tiles in 0.30 s, 17.8 tiles/s) is the fastest mover, his Super bolt (12.0) the
# fastest projectile, and a walk (2.17-2.73) lands at ~0.12. A relative velocity passes 1.0 only
# slightly, when a charged dash meets a walker head-on.
_NORM_SPEED_SCALE = 20.0

# Divisor for `normalize: true`'s "hp" fields. The largest HP the sim produces is a 10000-HP bot
# (bot_melee/bot_bull) at train.yaml's elite curriculum tier (`hp: 1.5` scales base_hp) holding
# `cubes.max_cubes: 16` at a flat `cubes.hp_per_cube: 400`: 15000 + 6400 = 21400, i.e. 1.07. The
# hero, which the curriculum never scales, tops out at 8000 + 6400 = 14400 and starts at 0.40.
# `entities.enemy_hp_mult` (1.0 by default) would multiply an enemy's total. Re-derive it, don't
# nudge it, if base_hp, the tiers or the cube numbers change.
# Except for one change, kept on purpose: configs/train.yaml lifts `cubes.max_cubes` to the game's
# 99 (2026-10-06), which raises the ceiling to 15000 + 39600 = 54600 (2.73). Every finished
# checkpoint, the deployed one included, was trained against 20000, and a value past 1.0 is not
# clipped anywhere: the float Box is unbounded and every run so far trains with `normalize.obs:
# false`, so the network reads it as is. The supply keeps it far below 2.73: a 2026-10-07 GPU
# smoke (18,805 episodes) saw no entity past 42 cubes, which caps the value at 1.59.
_NORM_HP_SCALE = 20000.0

# Divisor for `normalize: true`'s "count" fields. The largest count a shipped spec reads is
# `cubes` (`cubes.max_cubes: 16`, or 99 in configs/train.yaml, kept for the same reason as
# _NORM_HP_SCALE); `meta.n_enemies_alive` reaches cfg.n_enemies (9) and `hero.ammo_whole`
# max_ammo (3). Equal to _NORM_SPEED_SCALE by coincidence.
_NORM_COUNT_SCALE = 20.0

_tensor_cache: dict = {}


def _cached_tensor(key, values: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    """Memoizes a small constant tensor by (key, device), so grid-channel selection and
    normalization never call `torch.tensor(python_list, device=...)` per call: on CUDA that is a
    host sync (core/geometry.vec2)."""
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
    slots: str | None = None                       # None | "tracked" (rows by full_obs["slots"])
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
                    "be exposed to the agent"
                )
            if f.startswith("slots."):
                raise ValueError(
                    f"group {name!r}: field {f!r} is the slot bookkeeping `slots: tracked` reads "
                    "(core/slots.py), not an observation"
                )
            if f not in spec_fields:
                raise ValueError(f"group {name!r}: unknown field {f!r} (not present in obs_spec(cfg))")

        per_entity = bool(g.get("per_entity", False))
        max_slots = g.get("max_slots")
        entity_prefix = _entity_prefix(fields) if per_entity else None
        slots = g.get("slots")
        if slots is not None:
            if slots != "tracked":
                raise ValueError(f"group {name!r}: slots must be omitted or 'tracked', got {slots!r}")
            if not per_entity or entity_prefix != _HERO_AXIS_PREFIX:
                raise ValueError(
                    f"group {name!r}: slots: tracked is only defined for a per_entity "
                    f"{_HERO_AXIS_PREFIX}.* group (core/slots.py slots enemies, nothing else)"
                )
            if max_slots is not None:
                raise ValueError(f"group {name!r}: slots: tracked and max_slots are both row orders; pick one")
        if max_slots is not None:
            if not per_entity:
                raise ValueError(f"group {name!r}: max_slots is only meaningful when per_entity: true")
            for key in ("time_to_closest", "rel_pos"):  # the sort key, then its tie-break
                sort_field = f"{entity_prefix}.{key}"
                if sort_field not in spec_fields:
                    raise ValueError(
                        f"group {name!r}: max_slots requires a {sort_field!r} field to sort by")

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
            max_slots=max_slots, slots=slots, entity_prefix=entity_prefix, norm_scale=norm_scale,
        ))

    if len(groups) > _MAX_GROUPS:
        raise ValueError(f"agent_obs spec has {len(groups)} groups; keep it <= {_MAX_GROUPS} (each group is one host copy per step)")
    if not groups:
        raise ValueError("agent_obs spec has no groups")

    return AgentObsSpec(fair=fair, groups=tuple(groups), normalize=normalize)


def channel_index(cfg) -> dict:
    """Every `view` channel for this cfg, by name -> index: the 12 of `_CHANNEL_INDEX`, then
    "enemy_hist1".."enemy_hist{K}" at 12..11+K for K = cfg.history_frames. "enemy_histk" draws
    the enemies seen k decisions before the current observation (core/observation.py)."""
    index = dict(_CHANNEL_INDEX)
    base = len(_CHANNEL_INDEX)
    for k in range(1, cfg.history_frames + 1):
        index[f"{_HISTORY_CHANNEL_PREFIX}{k}"] = base + k - 1
    return index


def _load_grid_group(name: str, dtype: str, view_channels, fair: bool, cfg) -> GroupSpec:
    index = channel_index(cfg)
    for ch in view_channels:
        if ch not in index:
            raise ValueError(f"group {name!r}: unknown view channel {ch!r}; valid: {sorted(index)}")
    if fair:
        forbidden = [ch for ch in view_channels if ch in _UNFAIR_GRID_CHANNELS]
        if forbidden:
            raise ValueError(
                f"group {name!r}: fair:true forbids {forbidden} (use 'enemy_revealed', which "
                "already respects visibility -- 'enemy_any'/'enemy_hidden' would leak hidden "
                "enemy positions to the agent)"
            )
    channel_idx = tuple(index[c] for c in view_channels)
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
    """Preallocated `out_buffers` for `build_agent_obs`, built ONCE by the caller
    (`wrappers/sb3_vecenv.py`, `wrappers/gym_single.py`, brawl_deployment's assemble.py) and
    reused every call, so the output tensors are never reallocated."""
    torch_dtype = {"float32": torch.float32, "uint8": torch.uint8}
    return {g.name: torch.zeros((n_envs, *g.shape), dtype=torch_dtype[g.dtype], device=device) for g in spec.groups}


def agent_obs_index_map(spec: AgentObsSpec, cfg) -> dict:
    """{group_name: {field_name: (start, end)}} column ranges within each group's flattened
    last dim -- documentation-only (used by dump_obs_schema.py to render docs/AGENT_OBS*.md),
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


def _flat_columns(t: torch.Tensor) -> torch.Tensor:
    """(N,) -> (N,1), and (N, d1, d2, ...) -> (N, d1*d2*...) in row-major order: the width
    `_field_width` counts and the column order `agent_obs_index_map` documents. `hist.move_onehot`
    (N,K,17) becomes slot 0's 17 columns, then slot 1's, then slot 2's."""
    return t.unsqueeze(-1) if t.dim() == 1 else t.flatten(1)


def _build_flat_group(full_obs: dict, g: GroupSpec) -> torch.Tensor:
    parts = [_flat_columns(_get_field(full_obs, f)).to(torch.float32) for f in g.fields]
    return torch.cat(parts, dim=-1)


def _nearest_first(priority: torch.Tensor, rel_pos: torch.Tensor, k: int) -> torch.Tensor:
    """(N,k) indices of the k smallest `priority`, ties broken by distance to the hero, nearest
    first (user decision, 2026-09-30).

    `topk` leaves the order of equal values unspecified, and the sim keeps its projectiles in
    scattered slots while deployment keeps a packed list, so a slot-order tie-break would feed
    the policy tied projectiles in a different order live than in training. Sorting by distance,
    then STABLY by priority, orders by (priority, distance) wherever the projectiles sit; only a
    tie on both keys is left in slot order."""
    by_dist = torch.argsort(torch.linalg.vector_norm(rel_pos, dim=-1), dim=-1, stable=True)
    by_priority = torch.argsort(torch.gather(priority, 1, by_dist), dim=-1, stable=True)
    return torch.gather(by_dist, 1, by_priority[:, :k])


def _build_entity_group(full_obs: dict, g: GroupSpec, fair: bool) -> torch.Tensor:
    drop_hero = g.entity_prefix == _HERO_AXIS_PREFIX
    parts = []
    for f in g.fields:
        t = _ensure_trailing_dim(_get_field(full_obs, f), per_entity=True)
        if drop_hero:
            t = t[:, 1:]
        parts.append(t.to(torch.float32))
    out = torch.cat(parts, dim=-1)  # (N, n_slots, W)

    idx = None  # (N,K) row gather, set when max_slots or `slots: tracked` reorders the rows
    if g.max_slots is not None:
        ttc = _get_field(full_obs, f"{g.entity_prefix}.time_to_closest")
        alive = _get_field(full_obs, f"{g.entity_prefix}.alive")
        priority = torch.where(alive, ttc, torch.full_like(ttc, float("inf")))
        rel_pos = _get_field(full_obs, f"{g.entity_prefix}.rel_pos")
        idx = _nearest_first(priority, rel_pos, g.max_slots)  # (N,K)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, out.shape[-1])
        out = torch.gather(out, 1, idx_exp)
        slot_alive = torch.gather(alive, 1, idx).to(torch.float32).unsqueeze(-1)
        out = out * slot_alive  # zero-pad slots with fewer than max_slots alive occupants
    elif g.slots == "tracked":
        ent = full_obs["slots"]["entity"]                       # (N,K): entity + 1, 0 = empty
        idx = (ent - 2).clamp(min=0)                            # entity e -> row e - 1 (hero dropped)
        valid = (ent > 0).to(torch.float32).unsqueeze(-1)
        out = torch.gather(out, 1, idx.unsqueeze(-1).expand(-1, -1, out.shape[-1])) * valid

    if fair:
        mask_field = _FAIRNESS_MASK_FIELD.get(g.entity_prefix)
        if mask_field is not None:
            mask = _get_field(full_obs, mask_field)
            if drop_hero:
                mask = mask[:, 1:]
            if idx is not None:
                mask = torch.gather(mask, 1, idx)
            out = out * mask.to(torch.float32).unsqueeze(-1)

    return out


def _build_grid_group(full_obs: dict, g: GroupSpec) -> torch.Tensor:
    view = full_obs["view"]
    # Keyed on the channel tuple, not just the name: every spec calls its view group "grid", with
    # different channels, and one process can build several specs.
    idx = _cached_tensor(("grid_channels", g.name, g.channel_idx), g.channel_idx,
                         torch.int64, view.device)
    return view.index_select(1, idx)


def _normalize(raw: torch.Tensor, g: GroupSpec) -> torch.Tensor:
    # The scale tuple is part of the cache key, not just the group name: the specs share a "self"
    # group name at different widths and divisors, and one process can load several specs.
    scale = _cached_tensor(("norm_scale", g.name, g.norm_scale), g.norm_scale,
                           torch.float32, raw.device)
    return raw / scale


def _slow_projectiles_still(full_obs: dict, cfg) -> dict:
    """`full_obs` with every projectile slower than `cfg.obs_projectile_static_speed` read as
    still: `vel` (0, 0) and `time_to_closest` 0, exactly what `closest_approach` gives a zero
    velocity. A new dict around new tensors, so `full_obs`, which `info` and the reward read, keeps
    the true velocity. Off, or with no projectiles in it, it is `full_obs` itself.

    Live it changes nothing: `ProjectileTracker.snapshot` has already zeroed every track under
    STATIC_TILES_S, by the same norm."""
    floor = cfg.obs_projectile_static_speed
    if floor <= 0 or "projectiles" not in full_obs:
        return full_obs
    prj = full_obs["projectiles"]
    moving = torch.linalg.vector_norm(prj["vel"], dim=-1) >= floor             # (N, P)
    still = {"vel": torch.where(moving.unsqueeze(-1), prj["vel"], 0.0),
             "time_to_closest": torch.where(moving, prj["time_to_closest"], 0.0)}
    return {**full_obs, "projectiles": {**prj, **still}}


def build_agent_obs(full_obs: dict, spec: AgentObsSpec, cfg, out_buffers: dict) -> dict:
    full_obs = _slow_projectiles_still(full_obs, cfg)
    for g in spec.groups:
        if g.view_channels is not None:
            out_buffers[g.name].copy_(_build_grid_group(full_obs, g))
            continue
        raw = _build_entity_group(full_obs, g, spec.fair) if g.per_entity else _build_flat_group(full_obs, g)
        if g.norm_scale is not None:
            raw = _normalize(raw, g)
        out_buffers[g.name].copy_(raw)
    return out_buffers
