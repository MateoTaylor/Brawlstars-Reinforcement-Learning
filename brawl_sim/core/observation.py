"""The full, unredacted observation: everything the sim knows, before `obs_select.py` trims it
to the agent's fair view. Deployment (`brawl_deployment/perception/assemble.py`) builds a
minimal copy of this dict live and hands it to the same `obs_select.build_agent_obs`, so what a
field means here is also a contract with deployment.

This module never imports `bots/`: the caller (`env._build_observation`) precomputes `vis`
(`bots/perception.visibility`) and `raw_los` (`bots/perception.raw_los`) and hands them in, and
`_in_bush` duplicates `perception.in_bush`. `entities.team` is `ent_kind` cloned, an inert
placeholder (Solo Showdown is free-for-all). `rank` is not stored: `compute_rank` derives it
from `ent_alive`/`ent_death_step` each call.

`_build_grid` rasterizes both `obs["view"]` (the hero-centred `view_h x view_w` crop) and
`obs["world"]` (origin (0, 0), the whole map). Channels, named in `obs_select._CHANNEL_INDEX`:
0-3 terrain, 4 the zone (`zone.zone_grid`), 5-11 occupancy counts, then one enemy_hist plane per
history slot. Terrain reads the PADDED bank tensors with a `+ pad_h`/`+ pad_w` index shift,
which for the world grid lands exactly on the unpadded map, so one indexing path serves both.
Counts accumulate in int32 (uint8 has no atomic-add guarantee) and are clamped to [0, 255] on
the cast to uint8. brawl_deployment/perception/grid.py restates `_view_origin`, these channel
rules and `_history_drawn`: change them together.

History: `obs["hist"]` and the enemy_hist planes read core/history.py's rings, newest first.
Slot k is the observation k+1 decisions back plus the action that answered it; its plane is
`enemy_hist{k+1}`. Every `hist` field is 0 where `hist.valid` is False: an empty slot's action
(0, 0) is a real "idle, no attack", so the mask alone tells "no history" from "stood still". The
enemy_hist planes draw past sightings at their world position in the CURRENT window, with no
re-centring, and only within `cfg.history_radius_tiles` of the hero's current tile.

Many fields are views of `state`'s tensors, so an observation is valid only until the next
`step()`/`reset()`; `clone_obs` keeps one.
"""
import torch

from ..constants import N_KINDS, N_PROJ_CLASSES, N_PROJ_KINDS, ProjClass
from . import camera
from . import geometry as geo
from . import hero as hero_mod
from . import stats
from . import terrain
from . import zone as zone_mod

_HERO = 0  # entity slot 0 is always the hero
_EPS = 1e-6
_ARTILLERY = int(ProjClass.ARTILLERY)
# `_build_grid`'s fixed channels (terrain and zone 0-4, occupancy 5-11); one enemy_hist plane
# per history slot follows. obs_schema's "C" dim and obs_select._CHANNEL_INDEX restate the 12.
_N_BASE_CHANNELS = 12


def _in_bush(state, bank) -> torch.Tensor:
    """(N,E) bool -- see module docstring on why this duplicates perception.in_bush."""
    ix, iy = terrain.to_tile(state.ent_pos)
    map_id = terrain._broadcast_map_id(state.map_id, ix.shape)
    h, w = bank.is_bush.shape[1:]
    ix = torch.clamp(ix, 0, w - 1)
    iy = torch.clamp(iy, 0, h - 1)
    return bank.is_bush[map_id, iy, ix]


def compute_rank(alive: torch.Tensor, death_step: torch.Tensor) -> torch.Tensor:
    """(N,E) i64, 1-INDEXED placement (1 = best). Every alive entity ties at `rank == n_alive`
    (its finishing order is not decided yet); a dead one's rank is 1 + the entities that
    outlived it: everyone alive now, plus anyone with a later death_step.

    core/events.py reuses this for info["hero_rank"], which is 0-INDEXED (0 = winner), so that
    call site subtracts 1."""
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


def _onehot(index: torch.Tensor, width: int, valid: torch.Tensor) -> torch.Tensor:
    """(..., width) uint8 one-hot of `index`, zeroed where `valid` is False. By comparison with
    an arange, so the mask folds into the same expression and nothing can raise on the index.
    brawl_deployment/perception/assemble.py imports it."""
    hot = index.unsqueeze(-1) == torch.arange(width, device=index.device)
    return (hot & valid.unsqueeze(-1)).to(torch.uint8)


def _history_drawn(state, cfg) -> torch.Tensor:
    """(N,K,E) bool: the past sightings the enemy_hist planes draw. The slot must be valid, the
    entity seen back then (`hist_enemy_seen` already leaves out the dead and the hero's own
    column), and its tile back then within `cfg.history_radius_tiles` of the hero's tile NOW by
    Chebyshev distance on tile indices. On tiles the radius is exactly a (2r+1) x (2r+1) block
    of cells, the same block brawl_deployment can scatter from map-frame tracks; a distance on
    raw positions would take a boundary cell or not depending on where inside its tile each of
    the two stood."""
    hero_ix, hero_iy = terrain.to_tile(state.ent_pos[:, _HERO])
    ix, iy = terrain.to_tile(state.hist_enemy_pos)
    cheb = torch.maximum((ix - hero_ix.view(-1, 1, 1)).abs(), (iy - hero_iy.view(-1, 1, 1)).abs())
    return state.hist_enemy_seen & state.hist_valid.unsqueeze(-1) & (cheb <= cfg.history_radius_tiles)


def _build_grid(state, bank, hero_view: torch.Tensor, cfg, origin: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """(N, 12 + K, out_h, out_w) uint8, K = cfg.history_frames. See module docstring for the
    channel layout and the view/world unification. `hero_view`: (N,E) bool, what the hero's
    observation may show (`core/camera.hero_view`: concealment AND the camera window), the set
    the `enemy_revealed` plane draws. `origin`: (N,2) i64 real tile-space top-left corner."""
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

    n_hist = cfg.history_frames
    grid = torch.zeros((N, _N_BASE_CHANNELS + n_hist, out_h, out_w), dtype=torch.uint8, device=device)
    grid[:, 0] = bank.pad_blocks_unit[map_id_b, iy_pad, ix_pad].to(torch.uint8)
    grid[:, 1] = bank.pad_blocks_proj[map_id_b, iy_pad, ix_pad].to(torch.uint8)
    grid[:, 2] = bank.pad_is_bush[map_id_b, iy_pad, ix_pad].to(torch.uint8)
    grid[:, 3] = bank.pad_is_water[map_id_b, iy_pad, ix_pad].to(torch.uint8)
    grid[:, 4] = zone_mod.zone_grid(state, cfg, out_h, out_w, origin.to(state.ent_pos.dtype)).to(torch.uint8)

    # local 0..6 -> channels 5..11, local 7.. -> the enemy_hist planes from 12
    occ = torch.zeros((N, 7 + n_hist, out_h, out_w), dtype=torch.int32, device=device)
    not_hero = (torch.arange(E, device=device).view(1, E) != _HERO)
    enemy_alive = state.ent_alive & not_hero
    revealed_to_hero = hero_view

    _scatter_count(occ, 0, state.ent_pos, enemy_alive, origin, out_h, out_w)
    _scatter_count(occ, 1, state.ent_pos, enemy_alive & revealed_to_hero, origin, out_h, out_w)
    _scatter_count(occ, 2, state.ent_pos, enemy_alive & ~revealed_to_hero, origin, out_h, out_w)
    _scatter_count(occ, 3, state.ent_pos[:, _HERO:_HERO + 1], state.ent_alive[:, _HERO:_HERO + 1], origin, out_h, out_w)
    _scatter_count(occ, 4, state.box_pos, state.box_alive, origin, out_h, out_w)
    _scatter_count(occ, 5, state.pku_pos, state.pku_alive, origin, out_h, out_w)
    _scatter_count(occ, 6, state.prj_pos, state.prj_alive, origin, out_h, out_w)

    drawn = _history_drawn(state, cfg)
    for k in range(n_hist):  # K is a config constant: a fixed loop, no host read
        _scatter_count(occ, 7 + k, state.hist_enemy_pos[:, k], drawn[:, k], origin, out_h, out_w)

    grid[:, 5:] = torch.clamp(occ, max=255).to(torch.uint8)
    return grid


def clone_obs(node):
    """Recursively `.clone()`s every tensor leaf, keeping the dict structure (`compute_info`'s
    flat dict too). `build_obs`'s fields are views of `state`, so anything that must outlive a
    later in-place mutation needs a clone: env.py's `info["final_observation"]`/
    `info["final_info"]`, which autoreset's `spawn.reset_envs` would otherwise overwrite with
    post-reset values."""
    if isinstance(node, dict):
        return {key: clone_obs(value) for key, value in node.items()}
    return node.clone()


def build_obs(state, bank, vis: torch.Tensor, raw_los: torch.Tensor | None, params, cfg,
              hero_view: torch.Tensor | None = None) -> dict:
    """The full nested-dict observation: every entity/projectile/box/pickup, always present,
    index-stable, never masked. `vis`/`raw_los`: (N,E,E) bool, precomputed by the caller;
    `raw_los` is read only under `cfg.obs_include_raw_los` (it feeds `entities.los_from_hero`
    and `visibility.los`) and may be None otherwise. `hero_view`: (N,E) bool, what the hero's
    observation may show -- `vis`'s hero row AND the camera window (`core/camera.hero_view`);
    computed here when the caller passes None (the env passes the one it stashed for
    `history.push`). Every hero-side reveal field reads it; `hero_revealed_to` stays on `vis`,
    being what the bots see."""
    N, E = state.ent_pos.shape[:2]
    if hero_view is None:
        hero_view = camera.hero_view(state, vis, cfg)
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
    # The mask's own predicate, so `hero.gadget_ready` IS `action_mask.attack[:, 3]`.
    gadget_ready_all = hero_mod.gadget_ready(state, params)
    gadget_frac_all = hero_mod.gadget_charge_frac(state, params)
    rank_all = compute_rank(state.ent_alive, state.ent_death_step)

    revealed_to_hero = hero_view  # hero sees j: concealment AND on screen
    hero_revealed_to = vis[:, :, _HERO]  # j sees hero
    hidden_by_bush = in_bush_all & ~revealed_to_hero

    view_origin = _view_origin(state, cfg)             # the grid crop, hero-centred
    cam = camera.camera_centre(hero_pos, cfg)          # the screen's centre, clamped near edges

    def on_screen(pos: torch.Tensor) -> torch.Tensor:
        """(N,K,2) world positions -> (N,K) inside the ground window the screen shows."""
        return camera.in_camera(pos - cam.unsqueeze(1), cfg)

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
        "gadget_ready": gadget_ready_all[:, _HERO],
        "gadget_charge_frac": gadget_frac_all[:, _HERO],
        "attack_idle_t": state.ent_attack_idle_t[:, _HERO],
        "invuln": state.ent_invuln_t[:, _HERO] > 0,
        "in_bush": in_bush_all[:, _HERO],
        "in_zone": in_zone_all[:, _HERO],
        # The camera has stopped following the hero (near a map edge) by more than
        # `camera.edge_flag_tiles` on either axis. Live, `assemble._near_edge` thresholds the
        # player box's offset from its nominal screen anchor at the same cfg number.
        "near_edge": (hero_pos - cam).abs().amax(-1) > cfg.camera_edge_flag_tiles,
        "tile": torch.stack(terrain.to_tile(hero_pos), dim=-1),
        "damage_dealt": state.ent_damage_dealt[:, _HERO],
        "damage_taken": state.ent_damage_taken[:, _HERO],
        "kills": state.ent_kills[:, _HERO],
        "shots_fired": state.ent_shots_fired[:, _HERO],
        "rank": rank_all[:, _HERO],
    }

    # ---- hist: the last K decisions, newest first (see module docstring) ----
    # Masked field by field: an empty slot's action (0, 0) would one-hot as "idle, no attack"
    # and its zero position would read as a displacement of -hero.pos. hp and ammo are already
    # 0 there (state.zero_), and are masked anyway so the contract does not rest on that.
    # `assemble._put_history` repeats these expressions: change both together.
    valid = state.hist_valid
    n_move, n_attack = cfg.action_nvec
    obs["hist"] = {
        "valid": valid,
        "move_onehot": _onehot(state.hist_action[..., 0], n_move, valid),
        "attack_onehot": _onehot(state.hist_action[..., 1], n_attack, valid),
        "hp": torch.where(valid, state.hist_hp, 0.0),
        "ammo_frac": torch.where(
            valid, state.hist_ammo / torch.clamp(max_ammo_all[:, _HERO:_HERO + 1], min=_EPS), 0.0
        ),
        "displacement": torch.where(valid.unsqueeze(-1), state.hist_pos - hero_pos.unsqueeze(1), 0.0),
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
        "in_view": on_screen(state.ent_pos),
        "dist_rank": dist_rank,
        "revealed_to_hero": revealed_to_hero,
        "hero_revealed_to": hero_revealed_to,
        **({"los_from_hero": raw_los[:, _HERO, :]} if cfg.obs_include_raw_los else {}),
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
        # `class` tells a bullet, a lobbed shell and a stationary hazard puddle apart; `lobbed`
        # is its ARTILLERY-only view.
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
        "in_view": on_screen(state.prj_pos),
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
        "in_view": on_screen(state.box_pos),
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
        "in_view": on_screen(state.pku_pos),
    }

    # ---- zone ----
    map_area = float(cfg.map_w * cfg.map_h)
    zone_size = state.zone_hi - state.zone_lo
    hero_margin = torch.stack([
        hero_pos[:, 0] - state.zone_lo[:, 0],
        state.zone_hi[:, 0] - hero_pos[:, 0],
        hero_pos[:, 1] - state.zone_lo[:, 1],
        state.zone_hi[:, 1] - hero_pos[:, 1],
    ], dim=-1)
    obs["zone"] = {
        "lo": state.zone_lo,
        "hi": state.zone_hi,
        "lo_norm": state.zone_lo / map_wh,
        "hi_norm": state.zone_hi / map_wh,
        "active": state.zone_seen,  # latched by zone.mark_seen once gas has been on screen
        "step": state.zone_step,
        "dps": zone_mod.current_dps(state, params),
        "next_shrink_in": torch.clamp(state.zone_next_t - state.time, min=0.0),
        "hero_margin": hero_margin,
        # The same four margins, clamped to +/- `zone_margin_horizon_tiles`: deployment
        # reconstructs them from the gas it has SEEN, so its answer saturates past that horizon,
        # and it clamps at the same cfg number (BRAWL_DEPLOYMENT_DESIGN.md 9.14/9.15). Symmetric
        # because, deep in gas, the nearest clear ground is as unobservable as distant gas.
        "hero_margin_local": torch.clamp(
            hero_margin, -cfg.zone_margin_horizon_tiles, cfg.zone_margin_horizon_tiles
        ),
        "safe_area_frac": (zone_size[:, 0] * zone_size[:, 1]) / map_area,
    }

    # ---- tracker-style slots (core/slots.py) ----
    # Bookkeeping for obs_select's `slots: tracked`, not an observation: `load_agent_spec`
    # refuses `slots.*` in a spec's fields. Slot k holds entity `entity[k] - 1`; 0 is empty.
    obs["slots"] = {"entity": state.slot_ent, "valid": state.slot_ent > 0}

    # ---- visibility ----
    obs["visibility"] = {
        "vis": vis, **({"los": raw_los} if cfg.obs_include_raw_los else {}), "dist_matrix": dist_matrix,
    }

    # ---- grids ----
    obs["view"] = _build_grid(state, bank, hero_view, cfg, view_origin, cfg.view_h, cfg.view_w)
    if cfg.obs_include_world_grid:
        world_origin = torch.zeros((N, 2), dtype=torch.int64, device=device)
        obs["world"] = _build_grid(state, bank, hero_view, cfg, world_origin, cfg.map_h, cfg.map_w)

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
