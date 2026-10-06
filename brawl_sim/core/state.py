"""The simulation's per-env tensor buffers (entities, projectiles, boxes, pickups, zone/episode,
action latency, observation history, enemy slots). Everything is preallocated once in allocate()
and mutated in place thereafter -- nothing here is ever reallocated per step.
"""
import torch

F32, I64, I32, BOOL = torch.float32, torch.int64, torch.int32, torch.bool

# (field name, shape(N,E,P,B,U,L,K) -> tuple, dtype). Declarative so allocate() and the memory
# report share one source of truth for shapes/dtypes. K = cfg.history_frames, read only by the
# _HISTORY_FIELDS rings below; every other lambda ignores it.
_ENTITY_FIELDS = (
    ("ent_pos", lambda N, E, P, B, U, L, K: (N, E, 2), F32),
    ("ent_vel", lambda N, E, P, B, U, L, K: (N, E, 2), F32),
    ("ent_facing", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_hp", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_max_hp", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_alive", lambda N, E, P, B, U, L, K: (N, E), BOOL),
    ("ent_kind", lambda N, E, P, B, U, L, K: (N, E), I64),
    ("ent_cubes", lambda N, E, P, B, U, L, K: (N, E), I64),
    ("ent_ammo", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_attack_cd", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_dash_t", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_dash_dir", lambda N, E, P, B, U, L, K: (N, E, 2), F32),
    ("ent_dash_speed", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_dash_hits", lambda N, E, P, B, U, L, K: (N, E, E), BOOL),
    ("ent_invuln_t", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_reveal_t", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_react_t", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_out_of_combat_t", lambda N, E, P, B, U, L, K: (N, E), F32),
    # Seconds since this entity last ATTACKED (attack or super): a stopwatch counting UP, like
    # ent_out_of_combat_t. Drives Mortis's long dash. Deliberately NOT ent_out_of_combat_t, which
    # taking damage also resets: a Mortis under fire must still charge his long dash.
    ("ent_attack_idle_t", lambda N, E, P, B, U, L, K: (N, E), F32),
    # Hits landed on living PLAYERS since this entity's super was last fired. Per-entity so a bot
    # super needs no new state; only `hero_mortis` configures one today.
    ("ent_super_charge", lambda N, E, P, B, U, L, K: (N, E), I32),
    # Seconds until this entity's gadget is usable again; 0 means READY, so zero_'s blanket reset
    # is also "starts charged". hero.tick_timers counts it down and env._attack_phase sets it to
    # `gadget_cooldown` on a throw. Per-entity like ent_super_charge (only `hero_mortis` has one).
    ("ent_gadget_cd", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_target", lambda N, E, P, B, U, L, K: (N, E), I64),
    ("ent_move_smooth", lambda N, E, P, B, U, L, K: (N, E, 2), F32),
    # --- bot personality: per-entity and episode-scoped. core/spawn.py initializes them, only
    # bots/ reads them (bots/personality.py advances them). The hero's slot 0 carries them unread,
    # like ent_target/ent_move_smooth.
    ("ent_person", lambda N, E, P, B, U, L, K: (N, E), I64),
    ("ent_wander_dir", lambda N, E, P, B, U, L, K: (N, E, 2), F32),
    ("ent_wander_t", lambda N, E, P, B, U, L, K: (N, E), F32),
    # BITMASK, not a count: bit w set means "already searched bush waypoint w"
    # (maps/loader.bush_waypoints, capped at 63 so one int64 covers them). 0 is the fresh value.
    ("ent_hunt_seen", lambda N, E, P, B, U, L, K: (N, E), I64),
    ("ent_hunt_t", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_death_step", lambda N, E, P, B, U, L, K: (N, E), I32),
    ("ent_death_cause", lambda N, E, P, B, U, L, K: (N, E), I32),
    ("ent_last_hit_by", lambda N, E, P, B, U, L, K: (N, E), I64),
    ("ent_damage_dealt", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_damage_taken", lambda N, E, P, B, U, L, K: (N, E), F32),
    ("ent_kills", lambda N, E, P, B, U, L, K: (N, E), I32),
    ("ent_shots_fired", lambda N, E, P, B, U, L, K: (N, E), I32),
)

_PROJECTILE_FIELDS = (
    ("prj_pos", lambda N, E, P, B, U, L, K: (N, P, 2), F32),
    ("prj_vel", lambda N, E, P, B, U, L, K: (N, P, 2), F32),
    ("prj_target", lambda N, E, P, B, U, L, K: (N, P, 2), F32),
    ("prj_dist_left", lambda N, E, P, B, U, L, K: (N, P), F32),
    ("prj_damage", lambda N, E, P, B, U, L, K: (N, P), F32),
    ("prj_radius", lambda N, E, P, B, U, L, K: (N, P), F32),
    ("prj_aoe", lambda N, E, P, B, U, L, K: (N, P), F32),
    ("prj_age", lambda N, E, P, B, U, L, K: (N, P), F32),
    ("prj_owner", lambda N, E, P, B, U, L, K: (N, P), I64),
    ("prj_kind", lambda N, E, P, B, U, L, K: (N, P), I64),
    # constants.ProjClass: how this projectile MOVES and DAMAGES, independent of which weapon
    # fired it (that is prj_kind). PROJECTILE is 0, so zero_'s blanket reset leaves a freed slot
    # in the ordinary class.
    ("prj_class", lambda N, E, P, B, U, L, K: (N, P), I64),
    # PIERCE (the super's bolt): passes through walls and units instead of dying on the first
    # thing it touches. A property of the SHOT, not of its class.
    ("prj_pierce", lambda N, E, P, B, U, L, K: (N, P), BOOL),
    # Which entities a piercing projectile has ALREADY damaged, so it hits each victim once rather
    # than on every tick it overlaps them -- what ent_dash_hits does for the dash capsule.
    ("prj_hits", lambda N, E, P, B, U, L, K: (N, P, E), BOOL),
    ("prj_alive", lambda N, E, P, B, U, L, K: (N, P), BOOL),
)

_BOX_PICKUP_FIELDS = (
    ("box_pos", lambda N, E, P, B, U, L, K: (N, B, 2), F32),
    ("box_hp", lambda N, E, P, B, U, L, K: (N, B), F32),
    ("box_max_hp", lambda N, E, P, B, U, L, K: (N, B), F32),
    ("box_alive", lambda N, E, P, B, U, L, K: (N, B), BOOL),
    ("pku_pos", lambda N, E, P, B, U, L, K: (N, U, 2), F32),
    ("pku_cubes", lambda N, E, P, B, U, L, K: (N, U), I64),
    ("pku_alive", lambda N, E, P, B, U, L, K: (N, U), BOOL),
    ("pku_age", lambda N, E, P, B, U, L, K: (N, U), F32),
)

_ZONE_EPISODE_FIELDS = (
    ("zone_lo", lambda N, E, P, B, U, L, K: (N, 2), F32),
    ("zone_hi", lambda N, E, P, B, U, L, K: (N, 2), F32),
    ("zone_next_t", lambda N, E, P, B, U, L, K: (N,), F32),
    ("zone_step", lambda N, E, P, B, U, L, K: (N,), I32),
    # Latch behind obs zone.active: gas has been on screen at least once this episode
    # (core/zone.mark_seen). Reset with the rest of the episode fields.
    ("zone_seen", lambda N, E, P, B, U, L, K: (N,), BOOL),
    ("map_id", lambda N, E, P, B, U, L, K: (N,), I64),
    ("time", lambda N, E, P, B, U, L, K: (N,), F32),
    ("step_count", lambda N, E, P, B, U, L, K: (N,), I32),
    ("n_alive", lambda N, E, P, B, U, L, K: (N,), I32),
    ("boxes_broken", lambda N, E, P, B, U, L, K: (N,), I32),
)

_LATENCY_FIELDS = (
    ("act_buf", lambda N, E, P, B, U, L, K: (N, L, 2), I64),
    ("act_head", lambda N, E, P, B, U, L, K: (N,), I64),
)

# --- observation history rings. K = cfg.history_frames DECISIONS deep, newest at slot 0, written
# only by core/history.push (at the top of env.step, from the pre-step state) and by zero_ --
# which is what makes a reset row "no history": hist_valid all False and everything else 0. Only
# the observation reads them; nothing in the tick does. `hist_enemy_*` keep EVERY entity (slot 0 =
# the hero, its `seen` column forced False) so the enemy-history grid planes are a gather.
_HISTORY_FIELDS = (
    ("hist_valid", lambda N, E, P, B, U, L, K: (N, K), BOOL),
    ("hist_action", lambda N, E, P, B, U, L, K: (N, K, 2), I64),
    ("hist_hp", lambda N, E, P, B, U, L, K: (N, K), F32),
    ("hist_ammo", lambda N, E, P, B, U, L, K: (N, K), F32),
    ("hist_pos", lambda N, E, P, B, U, L, K: (N, K, 2), F32),
    ("hist_enemy_pos", lambda N, E, P, B, U, L, K: (N, K, E, 2), F32),
    ("hist_enemy_seen", lambda N, E, P, B, U, L, K: (N, K, E), BOOL),
)

# --- tracker-style enemy slots (core/slots.py). Written only by core/slots.update, once per
# DECISION from env._build_observation, and by zero_. Slot k of a `slots: tracked` group is
# entity `slot_ent[k] - 1`: stored +1 so a zeroed (reset) row means "no slots". The slot count is
# E - 1 -- the hero holds none -- and the lambda's K is the history depth, hence `E - 1` here.
_SLOT_FIELDS = (
    ("slot_ent", lambda N, E, P, B, U, L, K: (N, E - 1), I64),   # entity + 1 in slot k; 0 = empty
    ("ent_slot", lambda N, E, P, B, U, L, K: (N, E), I64),       # slot + 1 of entity e; 0 = none
    ("ent_hits", lambda N, E, P, B, U, L, K: (N, E), I32),       # consecutive sightings, unslotted
    ("ent_misses", lambda N, E, P, B, U, L, K: (N, E), I32),     # consecutive misses while slotted
)

# Cached, not part of "state" in the resettable sense -- excluded from zero_ (see below).
_CACHED_FIELDS = (
    ("env_idx", lambda N, E, P, B, U, L, K: (N,), I64),
)

_ALL_FIELD_SPECS = (
    _ENTITY_FIELDS + _PROJECTILE_FIELDS + _BOX_PICKUP_FIELDS
    + _ZONE_EPISODE_FIELDS + _LATENCY_FIELDS + _HISTORY_FIELDS + _SLOT_FIELDS + _CACHED_FIELDS
)
_RESETTABLE_FIELD_NAMES = tuple(name for name, _, _ in _ALL_FIELD_SPECS if name != "env_idx")


class SimState:
    __slots__ = tuple(name for name, _, _ in _ALL_FIELD_SPECS)


def _print_memory_report(state: SimState, N, E, P, B, U, L, K) -> None:
    rows = []
    total = 0
    for name, _, _ in _ALL_FIELD_SPECS:
        t = getattr(state, name)
        nbytes = t.element_size() * t.nelement()
        total += nbytes
        rows.append((name, tuple(t.shape), str(t.dtype), nbytes))
    rows.sort(key=lambda r: -r[3])
    print(f"SimState memory report (N={N}, E={E}, P={P}, B={B}, U={U}, L={L}, K={K}):")
    for name, shape, dtype, nbytes in rows:
        print(f"  {name:18s} {str(shape):16s} {dtype:14s} {nbytes / 1e6:9.4f} MB")
    print(f"  {'TOTAL':18s} {'':16s} {'':14s} {total / 1e6:9.4f} MB")


def allocate(cfg, n_envs: int, device, verbose: bool = True) -> SimState:
    N, E = n_envs, cfg.n_entities
    P, B, U, L = cfg.max_projectiles, cfg.max_boxes, cfg.max_pickups, cfg.latency_buf_len
    K = cfg.history_frames

    state = SimState()
    for name, shape_fn, dtype in _ALL_FIELD_SPECS:
        setattr(state, name, torch.zeros(shape_fn(N, E, P, B, U, L, K), dtype=dtype, device=device))
    state.env_idx = torch.arange(N, dtype=torch.int64, device=device)

    if verbose:
        _print_memory_report(state, N, E, P, B, U, L, K)

    return state


def zero_(state: SimState, reset_mask: torch.Tensor) -> None:
    """MUTATES: every field except env_idx. torch.where over the full batch -- masked rows
    become zero/False, unmasked rows are untouched. env_idx is a cached identity, never
    reset."""
    for name in _RESETTABLE_FIELD_NAMES:
        tensor = getattr(state, name)
        extra_dims = tensor.dim() - reset_mask.dim()
        mask = reset_mask.reshape(reset_mask.shape + (1,) * extra_dims)
        zero_val = torch.zeros((), dtype=tensor.dtype, device=tensor.device)
        tensor.copy_(torch.where(mask, zero_val, tensor))


def check_invariants(state: SimState, cfg, params) -> None:
    """Only meaningful (and only ever called) when cfg.debug_checks is True -- gated here too
    as a safety net. Not sync-free and not meant to be: this replaces the guarantees
    functional purity would have given, so it deliberately checks everything it can.

    Takes `params` because the cube, dash_t and gadget_cd bounds are per-env/per-kind SimParams
    fields, not EnvConfig ones.
    """
    if not cfg.debug_checks:
        return

    for name, _, _ in _ALL_FIELD_SPECS:
        t = getattr(state, name)
        if t.is_floating_point() and not torch.isfinite(t).all():
            raise ValueError(f"check_invariants: {name} has NaN/Inf")

    for pos_name in ("ent_pos", "prj_pos", "box_pos", "pku_pos"):
        pos = getattr(state, pos_name)
        if not torch.all((pos[..., 0] >= 0) & (pos[..., 0] <= cfg.map_w)):
            raise ValueError(f"check_invariants: {pos_name} x out of [0, map_w]")
        if not torch.all((pos[..., 1] >= 0) & (pos[..., 1] <= cfg.map_h)):
            raise ValueError(f"check_invariants: {pos_name} y out of [0, map_h]")

    if not torch.all((state.ent_hp >= 0) & (state.ent_hp <= state.ent_max_hp + 1e-3)):
        raise ValueError("check_invariants: ent_hp out of [0, max_hp]")
    if not torch.all((state.box_hp >= 0) & (state.box_hp <= state.box_max_hp + 1e-3)):
        raise ValueError("check_invariants: box_hp out of [0, max_hp]")

    if not torch.all(state.ent_alive | (state.ent_hp == 0)):
        raise ValueError("check_invariants: a dead entity has nonzero hp")

    # projectiles._box_grid keeps ONE alive box per tile, so two sharing a tile would make the
    # projectile phase miss one of them. boxes.spawn_boxes places them on distinct tiles.
    box_tile = torch.floor(state.box_pos[..., 1]) * cfg.map_w + torch.floor(state.box_pos[..., 0])
    both_alive = state.box_alive.unsqueeze(-1) & state.box_alive.unsqueeze(-2)
    not_self = ~torch.eye(state.box_alive.shape[-1], dtype=torch.bool, device=box_tile.device)
    if torch.any((box_tile.unsqueeze(-1) == box_tile.unsqueeze(-2)) & both_alive & not_self):
        raise ValueError("check_invariants: two alive boxes share a tile")

    if not torch.all(state.ent_cubes <= params.max_cubes.unsqueeze(-1)):
        raise ValueError("check_invariants: ent_cubes exceeds max_cubes")

    dash_duration = torch.gather(params.dash_duration, 1, state.ent_kind)
    if not torch.all(state.ent_dash_t <= dash_duration + 1e-3):
        raise ValueError("check_invariants: ent_dash_t exceeds dash_duration")

    # A countdown that clamps at 0 (hero.tick_timers) and is only ever SET to the kind's
    # gadget_cooldown, so either bound failing means a write path other than those two.
    gadget_cooldown = torch.gather(params.gadget_cooldown, 1, state.ent_kind)
    if not torch.all((state.ent_gadget_cd >= 0) & (state.ent_gadget_cd <= gadget_cooldown + 1e-3)):
        raise ValueError("check_invariants: ent_gadget_cd out of [0, gadget_cooldown]")

    if not torch.all(state.prj_alive.sum(dim=-1) <= state.prj_alive.shape[-1]):
        raise ValueError("check_invariants: live projectile count exceeds P slots")

    L = state.act_buf.shape[1]
    if not torch.all((state.act_head >= 0) & (state.act_head < L)):
        raise ValueError("check_invariants: act_head out of [0, L)")

    n = state.env_idx.shape[0]
    expected = torch.arange(n, dtype=torch.int64, device=state.env_idx.device)
    if not torch.equal(state.env_idx, expected):
        raise ValueError("check_invariants: env_idx no longer the identity permutation")


def snapshot(state: SimState, env_index: int) -> dict:
    """CPU, rendering and tests ONLY -- never call this from the sim's hot path.

    Every array is a detached COPY, not a view. The `.copy()` is not redundant: on a CPU-device
    `state`, `.cpu()` is a no-op and `.numpy()` shares memory with the live tensor, so without it
    a snapshot would change underneath its caller on the next step()."""
    return {name: getattr(state, name)[env_index].detach().cpu().numpy().copy() for name, _, _ in _ALL_FIELD_SPECS}
