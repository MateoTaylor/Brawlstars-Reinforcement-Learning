"""The simulation's per-env, per-entity/projectile/box/pickup tensor buffers. See
BRAWL_SIM_BUILD_PLAN.md Step 9. Everything is preallocated once in allocate() and mutated in
place thereafter -- nothing here should ever be reallocated per step.
"""
import torch

F32, I64, I32, BOOL = torch.float32, torch.int64, torch.int32, torch.bool

# (field name, shape(N,E,P,B,U,L) -> tuple, dtype). Declarative so allocate() and the memory
# report share one source of truth for shapes/dtypes.
_ENTITY_FIELDS = (
    ("ent_pos", lambda N, E, P, B, U, L: (N, E, 2), F32),
    ("ent_vel", lambda N, E, P, B, U, L: (N, E, 2), F32),
    ("ent_facing", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_hp", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_max_hp", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_alive", lambda N, E, P, B, U, L: (N, E), BOOL),
    ("ent_kind", lambda N, E, P, B, U, L: (N, E), I64),
    ("ent_cubes", lambda N, E, P, B, U, L: (N, E), I64),
    ("ent_ammo", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_attack_cd", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_dash_t", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_dash_dir", lambda N, E, P, B, U, L: (N, E, 2), F32),
    ("ent_dash_speed", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_dash_hits", lambda N, E, P, B, U, L: (N, E, E), BOOL),
    ("ent_invuln_t", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_reveal_t", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_react_t", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_out_of_combat_t", lambda N, E, P, B, U, L: (N, E), F32),
    # Seconds since this entity last ATTACKED -- a stopwatch counting UP, like
    # ent_out_of_combat_t and unlike every other timer here. Drives Mortis's long dash (Step D1).
    #
    # Deliberately NOT ent_out_of_combat_t, which looks like the same quantity but is also reset by
    # TAKING DAMAGE. Charging off that would mean a Mortis under fire never builds his long dash --
    # backwards, since being shot at while repositioning is exactly when it should be charging.
    ("ent_attack_idle_t", lambda N, E, P, B, U, L: (N, E), F32),
    # Hits landed on living PLAYERS since this entity's super was last fired (Step D2). Per-ENTITY,
    # not hero-only: only `hero_mortis` configures a super today, but bots are expected to get them,
    # and a charge counter that only tracked slot 0 would have to be rebuilt to allow that.
    ("ent_super_charge", lambda N, E, P, B, U, L: (N, E), I32),
    ("ent_target", lambda N, E, P, B, U, L: (N, E), I64),
    ("ent_move_smooth", lambda N, E, P, B, U, L: (N, E, 2), F32),
    # --- bot personality (Step 41). All six are per-entity and episode-scoped; core/spawn.py
    # initializes them and bots/personality.py is the only thing that reads or advances them.
    # The hero's slot 0 carries them too (allocate is uniform over E) and they are simply never
    # read for it, exactly like ent_target/ent_move_smooth already are.
    ("ent_person", lambda N, E, P, B, U, L: (N, E), I64),
    ("ent_wander_dir", lambda N, E, P, B, U, L: (N, E, 2), F32),
    ("ent_wander_t", lambda N, E, P, B, U, L: (N, E), F32),
    # BITMASK, not a count: bit w set means "I have already searched bush waypoint w" (see
    # maps/loader.bush_waypoints, capped at 63 so one int64 covers every waypoint). 0 is the
    # correct fresh value, so core/state.zero_ initializes this one for free.
    ("ent_hunt_seen", lambda N, E, P, B, U, L: (N, E), I64),
    ("ent_hunt_t", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_death_step", lambda N, E, P, B, U, L: (N, E), I32),
    ("ent_death_cause", lambda N, E, P, B, U, L: (N, E), I32),
    ("ent_last_hit_by", lambda N, E, P, B, U, L: (N, E), I64),
    ("ent_damage_dealt", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_damage_taken", lambda N, E, P, B, U, L: (N, E), F32),
    ("ent_kills", lambda N, E, P, B, U, L: (N, E), I32),
    ("ent_shots_fired", lambda N, E, P, B, U, L: (N, E), I32),
)

_PROJECTILE_FIELDS = (
    ("prj_pos", lambda N, E, P, B, U, L: (N, P, 2), F32),
    ("prj_vel", lambda N, E, P, B, U, L: (N, P, 2), F32),
    ("prj_target", lambda N, E, P, B, U, L: (N, P, 2), F32),
    ("prj_dist_left", lambda N, E, P, B, U, L: (N, P), F32),
    ("prj_damage", lambda N, E, P, B, U, L: (N, P), F32),
    ("prj_radius", lambda N, E, P, B, U, L: (N, P), F32),
    ("prj_aoe", lambda N, E, P, B, U, L: (N, P), F32),
    ("prj_age", lambda N, E, P, B, U, L: (N, P), F32),
    ("prj_owner", lambda N, E, P, B, U, L: (N, P), I64),
    ("prj_kind", lambda N, E, P, B, U, L: (N, P), I64),
    # constants.ProjClass: how this projectile MOVES and DAMAGES, independent of which weapon
    # fired it (that is prj_kind). Replaced the `prj_lobbed` bool in Step C3, which was already a
    # two-value version of the same idea. PROJECTILE is 0, so state.zero_'s blanket reset leaves a
    # freed slot in the ordinary class rather than an exotic one.
    ("prj_class", lambda N, E, P, B, U, L: (N, P), I64),
    # PIERCE (Step D2): this projectile passes through walls and units instead of dying on the
    # first thing it touches. A property of the SHOT, not of its class -- a future piercing
    # artillery shell or piercing hazard is expressible without a fourth ProjClass.
    ("prj_pierce", lambda N, E, P, B, U, L: (N, P), BOOL),
    # Which entities a piercing projectile has ALREADY damaged. Without it a bolt would re-damage
    # the same victim on every tick it overlaps them. Exactly the shape and purpose of
    # ent_dash_hits, which solves the identical problem for the dash capsule.
    #
    # (N,P,E) bool is the largest new buffer in the whole overhaul: 7.9 MB at n_envs=4096, P=192,
    # E=10 -- against a measured 2.2 GB total, and only allocated once.
    ("prj_hits", lambda N, E, P, B, U, L: (N, P, E), BOOL),
    ("prj_alive", lambda N, E, P, B, U, L: (N, P), BOOL),
)

_BOX_PICKUP_FIELDS = (
    ("box_pos", lambda N, E, P, B, U, L: (N, B, 2), F32),
    ("box_hp", lambda N, E, P, B, U, L: (N, B), F32),
    ("box_max_hp", lambda N, E, P, B, U, L: (N, B), F32),
    ("box_alive", lambda N, E, P, B, U, L: (N, B), BOOL),
    ("pku_pos", lambda N, E, P, B, U, L: (N, U, 2), F32),
    ("pku_cubes", lambda N, E, P, B, U, L: (N, U), I64),
    ("pku_alive", lambda N, E, P, B, U, L: (N, U), BOOL),
    ("pku_age", lambda N, E, P, B, U, L: (N, U), F32),
)

_ZONE_EPISODE_FIELDS = (
    ("zone_lo", lambda N, E, P, B, U, L: (N, 2), F32),
    ("zone_hi", lambda N, E, P, B, U, L: (N, 2), F32),
    ("zone_next_t", lambda N, E, P, B, U, L: (N,), F32),
    ("zone_step", lambda N, E, P, B, U, L: (N,), I32),
    ("map_id", lambda N, E, P, B, U, L: (N,), I64),
    ("time", lambda N, E, P, B, U, L: (N,), F32),
    ("step_count", lambda N, E, P, B, U, L: (N,), I32),
    ("n_alive", lambda N, E, P, B, U, L: (N,), I32),
    ("boxes_broken", lambda N, E, P, B, U, L: (N,), I32),
)

_LATENCY_FIELDS = (
    ("act_buf", lambda N, E, P, B, U, L: (N, L, 2), I64),
    ("act_head", lambda N, E, P, B, U, L: (N,), I64),
)

# Cached, not part of "state" in the resettable sense -- excluded from zero_ (see below).
_CACHED_FIELDS = (
    ("env_idx", lambda N, E, P, B, U, L: (N,), I64),
)

_ALL_FIELD_SPECS = (
    _ENTITY_FIELDS + _PROJECTILE_FIELDS + _BOX_PICKUP_FIELDS
    + _ZONE_EPISODE_FIELDS + _LATENCY_FIELDS + _CACHED_FIELDS
)
_RESETTABLE_FIELD_NAMES = tuple(name for name, _, _ in _ALL_FIELD_SPECS if name != "env_idx")


class SimState:
    __slots__ = tuple(name for name, _, _ in _ALL_FIELD_SPECS)


def _print_memory_report(state: SimState, N, E, P, B, U, L) -> None:
    rows = []
    total = 0
    for name, _, _ in _ALL_FIELD_SPECS:
        t = getattr(state, name)
        nbytes = t.element_size() * t.nelement()
        total += nbytes
        rows.append((name, tuple(t.shape), str(t.dtype), nbytes))
    rows.sort(key=lambda r: -r[3])
    print(f"SimState memory report (N={N}, E={E}, P={P}, B={B}, U={U}, L={L}):")
    for name, shape, dtype, nbytes in rows:
        print(f"  {name:18s} {str(shape):16s} {dtype:14s} {nbytes / 1e6:9.4f} MB")
    print(f"  {'TOTAL':18s} {'':16s} {'':14s} {total / 1e6:9.4f} MB")


def allocate(cfg, n_envs: int, device, verbose: bool = True) -> SimState:
    N, E = n_envs, cfg.n_entities
    P, B, U, L = cfg.max_projectiles, cfg.max_boxes, cfg.max_pickups, cfg.latency_buf_len

    state = SimState()
    for name, shape_fn, dtype in _ALL_FIELD_SPECS:
        setattr(state, name, torch.zeros(shape_fn(N, E, P, B, U, L), dtype=dtype, device=device))
    state.env_idx = torch.arange(N, dtype=torch.int64, device=device)

    if verbose:
        _print_memory_report(state, N, E, P, B, U, L)

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

    Deviates from the Step 9 pseudocode's two-argument signature by taking `params` too --
    two of the listed checks (cubes <= max_cubes, dash_t <= dash_duration) are against
    per-env/per-kind SimParams fields that don't exist on EnvConfig, so there's no way to
    perform them without it.
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

    if not torch.all(state.ent_cubes <= params.max_cubes.unsqueeze(-1)):
        raise ValueError("check_invariants: ent_cubes exceeds max_cubes")

    dash_duration = torch.gather(params.dash_duration, 1, state.ent_kind)
    if not torch.all(state.ent_dash_t <= dash_duration + 1e-3):
        raise ValueError("check_invariants: ent_dash_t exceeds dash_duration")

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

    Every array is a genuine detached COPY, not a view -- `.copy()` after `.numpy()` is not
    redundant here despite `.cpu()` already copying on a CUDA source: on a CPU-device `state`,
    `.cpu()` is a no-op (already CPU) and `.numpy()` shares memory with the live tensor, so
    without the explicit `.copy()` a CPU-device snapshot is a VIEW that changes underneath the
    caller on the next mutating call -- exactly the "zero-copy view valid only until the next
    call" footgun this codebase documents everywhere else (Step 25/27/29), except a caller of
    `snapshot()` has no way to know that from the name alone the way `build_obs`'s docs make
    explicit. Found by `scripts/record_rollout.py` (Step 37), the first caller to hold more than
    one snapshot alive across intervening `step()` calls on CPU -- every earlier call site only
    ever used one snapshot at a time, so this never surfaced before."""
    return {name: getattr(state, name)[env_index].detach().cpu().numpy().copy() for name, _, _ in _ALL_FIELD_SPECS}
