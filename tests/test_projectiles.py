import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import load_config, build_params
from brawl_sim.constants import Kind, Proj, ProjClass, Tile, TILE_BLOCKS_PROJ
from brawl_sim.core import projectiles as proj
from brawl_sim.core import stats
from brawl_sim.core import terrain
from brawl_sim.core.state import allocate

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


class _FakeBank:
    def __init__(self, blocks_proj):
        self.blocks_proj = blocks_proj


def _grid(h, w, fill=Tile.FLOOR):
    t = torch.full((h, w), int(fill), dtype=torch.int64)
    t[0, :] = Tile.WALL
    t[-1, :] = Tile.WALL
    t[:, 0] = Tile.WALL
    t[:, -1] = Tile.WALL
    return t


def _bank_from_grid(tiles):
    return _FakeBank(blocks_proj=TILE_BLOCKS_PROJ[tiles].unsqueeze(0))


def _cfg_and_params(n_envs=1, map_h=20, map_w=20, max_projectiles=16, max_boxes=8):
    cfg = load_config(CONFIGS / "default.yaml", overrides={
        "world": {"map_h": map_h, "map_w": map_w},
        "limits": {"max_projectiles": max_projectiles, "max_boxes": max_boxes, "max_pickups": 8},
    })
    spec = {
        **yaml.safe_load((CONFIGS / "default.yaml").read_text()),
        **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    return cfg, params


def _grom_split_count(params) -> int:
    """Grom's arm count, read from his own params rather than a module constant.

    It used to be `projectiles.N_SPLITS`, a module-level 4 that WAS the arm count for every kind
    because there was only one ring. `split_count` is now per kind (Spike's star is 6), so a
    module constant can only mean the ceiling -- asserting against it here would silently stop
    testing Grom the moment a wider brawler was added."""
    return int(params.split_count[0, int(Kind.BOT_ARTILLERY)])


def _fresh_state(cfg, n_envs=1):
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[:, 0] = int(Kind.HERO_MORTIS)
    for e in range(1, cfg.n_entities):
        state.ent_kind[:, e] = int(Kind.BOT_SNIPER)
    state.map_id.fill_(0)
    return state


# ---- alloc_slots ------------------------------------------------------------

def test_alloc_slots_basic_sequential_assignment():
    prj_alive = torch.zeros(1, 10, dtype=torch.bool)
    demand = torch.zeros(1, 3, dtype=torch.int64)
    demand[0, 0] = 2  # entity 0 wants 2 slots
    idx, ok = proj.alloc_slots(prj_alive, demand, max_per_entity=3)

    assert ok[0, 0, 0] and ok[0, 0, 1] and not ok[0, 0, 2]
    claimed = sorted(idx[0, 0, ok[0, 0]].tolist())
    assert claimed == [0, 1]


def test_alloc_slots_collision_free_across_entities_same_tick():
    prj_alive = torch.zeros(1, 20, dtype=torch.bool)
    demand = torch.tensor([[3, 1, 2, 0]], dtype=torch.int64)  # E=4, total demand=6
    idx, ok = proj.alloc_slots(prj_alive, demand, max_per_entity=3)

    claimed = idx[ok].tolist()
    assert len(claimed) == 6
    assert len(set(claimed)) == 6  # every claimed slot is distinct


def test_alloc_slots_never_reuses_a_live_slot():
    prj_alive = torch.zeros(1, 6, dtype=torch.bool)
    prj_alive[0, [1, 3]] = True  # slots 1 and 3 already occupied
    demand = torch.tensor([[4]], dtype=torch.int64)  # 4 free slots available: 0,2,4,5
    idx, ok = proj.alloc_slots(prj_alive, demand, max_per_entity=4)

    claimed = idx[0, 0][ok[0, 0]].tolist()
    assert sorted(claimed) == [0, 2, 4, 5]


def test_alloc_slots_full_buffer_drops_shot_silently():
    prj_alive = torch.ones(1, 4, dtype=torch.bool)  # buffer completely full
    demand = torch.tensor([[2]], dtype=torch.int64)
    idx, ok = proj.alloc_slots(prj_alive, demand, max_per_entity=3)
    assert not torch.any(ok)


def test_alloc_slots_partial_overflow_only_drops_the_excess():
    prj_alive = torch.zeros(1, 5, dtype=torch.bool)
    prj_alive[0, :3] = True  # only 2 free slots (indices 3, 4)
    demand = torch.tensor([[4]], dtype=torch.int64)
    idx, ok = proj.alloc_slots(prj_alive, demand, max_per_entity=4)
    assert ok[0, 0].tolist() == [True, True, False, False]
    assert sorted(idx[0, 0][ok[0, 0]].tolist()) == [3, 4]


def test_alloc_slots_batched_across_envs_independent():
    prj_alive = torch.zeros(2, 5, dtype=torch.bool)
    prj_alive[1, :] = True  # env 1's buffer is full, env 0's is empty
    demand = torch.tensor([[2], [2]], dtype=torch.int64)
    idx, ok = proj.alloc_slots(prj_alive, demand, max_per_entity=2)
    assert torch.all(ok[0])
    assert not torch.any(ok[1])


def _live(state, cls=None):
    """Live projectile slots, optionally of one ProjClass.

    "The shot is over" does not imply "no slots are alive": Brock's rocket
    leaves a HAZARD-class sphere behind wherever it dies. Tests about the ROCKET must therefore
    say which class they mean rather than asserting on the whole buffer."""
    alive = state.prj_alive[0]
    if cls is not None:
        alive = alive & (state.prj_class[0] == int(cls))
    return int(alive.sum())


# ---- spawn_volley ------------------------------------------------------------

def test_spawn_volley_single_shot_fields():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, 1] = True  # bot_sniper (proj_count=1)
    origin = state.ent_pos.clone()
    origin[0, 1] = torch.tensor([5.0, 5.0])
    aim_dir = torch.zeros(1, cfg.n_entities, 2)
    aim_dir[0, 1] = torch.tensor([1.0, 0.0])
    aim_point = origin.clone()
    aim_point[0, 1] = torch.tensor([16.0, 5.0])
    kind = state.ent_kind
    damage = stats.effective_damage(kind, state.ent_cubes, params)

    proj.spawn_volley(state, fire_mask, origin, aim_dir, aim_point, kind, damage, params, cfg)

    assert int(state.prj_alive.sum()) == 1
    (p_idx,) = torch.nonzero(state.prj_alive[0], as_tuple=True)
    p = p_idx.item()
    assert torch.allclose(state.prj_pos[0, p], torch.tensor([5.0, 5.0]))
    assert state.prj_owner[0, p].item() == 1
    assert state.prj_kind[0, p].item() == int(Proj.SNIPER_BOLT)
    assert int(state.prj_class[0, p]) == int(ProjClass.PROJECTILE)
    attack_range = params.attack_range[0, int(Kind.BOT_SNIPER)].item()
    assert abs(state.prj_dist_left[0, p].item() - attack_range) < 1e-4


def test_rifle_volley_occupies_one_slot_per_pellet():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_kind[0, 1] = int(Kind.BOT_RIFLE)
    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, 1] = True
    origin = state.ent_pos.clone()
    aim_dir = torch.zeros(1, cfg.n_entities, 2)
    aim_dir[0, 1] = torch.tensor([1.0, 0.0])
    aim_point = origin.clone()
    kind = state.ent_kind
    damage = stats.effective_damage(kind, state.ent_cubes, params)

    proj.spawn_volley(state, fire_mask, origin, aim_dir, aim_point, kind, damage, params, cfg)
    # One slot per pellet, whatever proj_count currently is (5 for Shelly).
    assert int(state.prj_alive.sum()) == int(params.proj_count[0, int(Kind.BOT_RIFLE)])

    alive_idx = torch.nonzero(state.prj_alive[0], as_tuple=True)[0]
    angles = torch.atan2(state.prj_vel[0, alive_idx, 1], state.prj_vel[0, alive_idx, 0])
    spread = params.proj_spread_rad[0, int(Kind.BOT_RIFLE)].item()
    assert abs((angles.max() - angles.min()).item() - spread) < 1e-3


def test_symmetric_fan_matches_expected_offsets():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_kind[0, 1] = int(Kind.BOT_RIFLE)
    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, 1] = True
    origin = state.ent_pos.clone()
    aim_dir = torch.zeros(1, cfg.n_entities, 2)
    aim_dir[0, 1] = torch.tensor([1.0, 0.0])  # angle 0
    aim_point = origin.clone()
    kind = state.ent_kind
    damage = stats.effective_damage(kind, state.ent_cubes, params)

    proj.spawn_volley(state, fire_mask, origin, aim_dir, aim_point, kind, damage, params, cfg)
    alive_idx = torch.nonzero(state.prj_alive[0], as_tuple=True)[0]
    angles = torch.sort(torch.atan2(state.prj_vel[0, alive_idx, 1], state.prj_vel[0, alive_idx, 0])).values
    spread = params.proj_spread_rad[0, int(Kind.BOT_RIFLE)].item()
    n = int(params.proj_count[0, int(Kind.BOT_RIFLE)])
    # Evenly spaced across the FULL span, symmetric about aim_dir. linspace is the same statement
    # for any odd or even count, so a new pellet count is a config change, not a test rewrite.
    expected = torch.linspace(-spread / 2, spread / 2, n)
    assert torch.allclose(angles, expected, atol=1e-4)


def test_artillery_shell_gets_the_ARTILLERY_class():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_kind[0, 1] = int(Kind.BOT_ARTILLERY)
    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, 1] = True
    origin = state.ent_pos.clone()
    aim_dir = torch.zeros(1, cfg.n_entities, 2)
    aim_dir[0, 1] = torch.tensor([1.0, 0.0])
    aim_point = origin.clone()
    kind = state.ent_kind
    damage = stats.effective_damage(kind, state.ent_cubes, params)

    proj.spawn_volley(state, fire_mask, origin, aim_dir, aim_point, kind, damage, params, cfg)
    (p_idx,) = torch.nonzero(state.prj_alive[0], as_tuple=True)
    assert int(state.prj_class[0, p_idx.item()]) == int(ProjClass.ARTILLERY)


def test_simultaneous_volleys_never_collide_in_allocation():
    cfg, params = _cfg_and_params(max_projectiles=32)
    state = _fresh_state(cfg)
    state.ent_kind[0, 1] = int(Kind.BOT_RIFLE)
    state.ent_kind[0, 2] = int(Kind.BOT_RIFLE)
    # Read the volley width off params rather than pinning it: this test is about ALLOCATION (two
    # shooters must never claim the same slot), not about how wide the fan happens to be.
    n = int(params.proj_count[0, int(Kind.BOT_RIFLE)])
    assert n > 1, "this test needs a multi-projectile kind to be meaningful"

    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, 1] = True
    fire_mask[0, 2] = True
    origin = state.ent_pos.clone()
    aim_dir = torch.zeros(1, cfg.n_entities, 2)
    aim_dir[0, 1] = torch.tensor([1.0, 0.0])
    aim_dir[0, 2] = torch.tensor([0.0, 1.0])
    aim_point = origin.clone()
    kind = state.ent_kind
    damage = stats.effective_damage(kind, state.ent_cubes, params)

    proj.spawn_volley(state, fire_mask, origin, aim_dir, aim_point, kind, damage, params, cfg)
    assert int(state.prj_alive.sum()) == 2 * n
    owners = state.prj_owner[0][state.prj_alive[0]].tolist()
    assert sorted(owners) == [1] * n + [2] * n


def test_volley_width_constant_covers_every_kinds_proj_count():
    """`spawn_volley` allocates a (N, E, MAX_PROJ_PER_ENTITY) grid, so a kind whose `proj_count`
    exceeds that constant silently fires only part of its volley -- no error, no warning, just a
    thinner shotgun.

    `config.validate`'s `peak_projectile_demand` does NOT catch this: it sizes the buffer against
    concurrent demand, which is a different question from how wide a single volley may be. So the
    relationship is asserted here, against the shipped roster."""
    cfg, params = _cfg_and_params()
    widest = int(params.proj_count.max())
    assert widest <= proj.MAX_PROJ_PER_ENTITY, (
        f"a kind fires {widest} projectiles but MAX_PROJ_PER_ENTITY is "
        f"{proj.MAX_PROJ_PER_ENTITY}; volleys would be silently truncated"
    )


def test_firing_into_full_buffer_changes_nothing():
    cfg, params = _cfg_and_params(max_projectiles=2)
    state = _fresh_state(cfg)
    state.prj_alive[0, :2] = True
    before_pos = state.prj_pos.clone()
    before_alive = state.prj_alive.clone()

    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, 1] = True
    origin = state.ent_pos.clone()
    aim_dir = torch.zeros(1, cfg.n_entities, 2)
    aim_dir[0, 1] = torch.tensor([1.0, 0.0])
    aim_point = origin.clone()
    kind = state.ent_kind
    damage = stats.effective_damage(kind, state.ent_cubes, params)

    proj.spawn_volley(state, fire_mask, origin, aim_dir, aim_point, kind, damage, params, cfg)

    assert torch.equal(state.prj_alive, before_alive)
    assert torch.equal(state.prj_pos, before_pos)


# ---- step_projectiles ---------------------------------------------------------

def _spawn_single_bolt(state, cfg, params, origin, direction, shooter_kind=Kind.BOT_SNIPER, owner=1):
    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, owner] = True
    state.ent_kind[0, owner] = int(shooter_kind)
    origins = state.ent_pos.clone()
    origins[0, owner] = torch.tensor(origin)
    aim_dir = torch.zeros(1, cfg.n_entities, 2)
    aim_dir[0, owner] = torch.tensor(direction)
    aim_point = origins.clone()
    kind = state.ent_kind
    damage = stats.effective_damage(kind, state.ent_cubes, params)
    proj.spawn_volley(state, fire_mask, origins, aim_dir, aim_point, kind, damage, params, cfg)
    return owner


def test_bolt_dies_at_wall():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    tiles[:, 12] = Tile.WALL
    bank = _bank_from_grid(tiles)

    _spawn_single_bolt(state, cfg, params, [5.0, 5.0], [1.0, 0.0])

    for _ in range(50):
        if not torch.any(state.prj_alive):
            break
        proj.step_projectiles(state, bank, params, cfg)

    # The ROCKET is gone. A hazard sphere may remain in its place (Brock leaves one
    # wherever his rocket dies) -- that is a different class and a different
    # mechanic, so this assertion is about the PROJECTILE-class slot only.
    assert _live(state, ProjClass.PROJECTILE) == 0


def test_bolt_over_water_reaches_far_side():
    """Water blocks units but not projectiles.

    Start position, finish line and tick budget are all DERIVED from the shooter's own stats. They
    were hardcoded (spawn x=5.0, finish x>13.0, 100 ticks), which silently assumed
    `attack_range = 8.67`: a bolt fired from 5.0 expires at exactly 5.0 + range, so when Brock's
    range moved to 8.0 the bolt died at exactly 13.0 and the strict `> 13.0` failed by
    0.0 -- a range change reading as a water-passability bug.
    """
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    water_col = 12
    tiles = _grid(20, 20)
    tiles[:, water_col] = Tile.WATER
    bank = _bank_from_grid(tiles)

    attack_range = params.attack_range[0, int(Kind.BOT_SNIPER)].item()
    proj_speed = params.proj_speed[0, int(Kind.BOT_SNIPER)].item()
    far_side_x = water_col + 1.0                 # past the tile spanning [12, 13)
    start_x = far_side_x - attack_range + 1.5    # 1.5 tiles of range to spare beyond the far side
    assert 1.0 < start_x < water_col, (
        f"derived start x={start_x:.2f} is not on the near side of the water column; "
        f"attack_range={attack_range} no longer suits this 20x20 fixture"
    )
    # Enough ticks to fly the full range, plus slack. proj_speed once dropped 14.0 -> 5.33, which
    # nearly tripled the flight time -- a fixed budget would have been the next thing to rot.
    max_ticks = int(attack_range / (proj_speed * cfg.dt)) + 20

    _spawn_single_bolt(state, cfg, params, [start_x, 5.0], [1.0, 0.0])
    reached_far_side = False
    for _ in range(max_ticks):
        if not torch.any(state.prj_alive):
            break
        proj.step_projectiles(state, bank, params, cfg)
        if state.prj_alive.any() and state.prj_pos[state.prj_alive][0, 0].item() > far_side_x:
            reached_far_side = True

    assert reached_far_side


def test_unit_collision_deals_damage_and_kills_projectile():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 0] = torch.tensor([10.0, 5.0])  # hero sits in the bolt's path
    state.ent_hp[0, 0] = 999999.0
    state.ent_max_hp[0, 0] = 999999.0

    _spawn_single_bolt(state, cfg, params, [5.0, 5.0], [1.0, 0.0], owner=1)

    total_dmg = torch.zeros(1, cfg.n_entities)
    for _ in range(50):
        if not torch.any(state.prj_alive):
            break
        dmg_ent, dmg_by, dmg_box, _heal, _charge = proj.step_projectiles(state, bank, params, cfg)
        total_dmg += dmg_ent

    assert total_dmg[0, 0].item() > 0.0
    # The ROCKET is gone. A hazard sphere may remain in its place (Brock leaves one
    # wherever his rocket dies) -- that is a different class and a different
    # mechanic, so this assertion is about the PROJECTILE-class slot only.
    assert _live(state, ProjClass.PROJECTILE) == 0


def test_projectile_never_hits_its_own_owner():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)

    owner = _spawn_single_bolt(state, cfg, params, [10.0, 5.0], [1.0, 0.0], owner=1)
    state.ent_pos[0, owner] = torch.tensor([10.0, 5.0])  # owner sitting right at spawn point

    total_dmg = torch.zeros(1, cfg.n_entities)
    for _ in range(50):
        if not torch.any(state.prj_alive):
            break
        dmg_ent, dmg_by, dmg_box, _heal, _charge = proj.step_projectiles(state, bank, params, cfg)
        total_dmg += dmg_ent
    assert total_dmg[0, owner].item() == 0.0


def test_earliest_t_wins_hits_closer_target_not_farther():
    cfg, params = _cfg_and_params(map_h=30, map_w=30)
    state = _fresh_state(cfg)
    tiles = _grid(30, 30)
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 0] = torch.tensor([8.0, 5.0])   # closer
    state.ent_pos[0, 2] = torch.tensor([15.0, 5.0])  # farther, same line
    state.ent_kind[0, 2] = int(Kind.BOT_SNIPER)
    for e in (0, 2):
        state.ent_hp[0, e] = 999999.0
        state.ent_max_hp[0, e] = 999999.0

    _spawn_single_bolt(state, cfg, params, [5.0, 5.0], [1.0, 0.0], owner=1)

    total_dmg = torch.zeros(1, cfg.n_entities)
    for _ in range(50):
        if not torch.any(state.prj_alive):
            break
        dmg_ent, dmg_by, dmg_box, _heal, _charge = proj.step_projectiles(state, bank, params, cfg)
        total_dmg += dmg_ent

    assert total_dmg[0, 0].item() > 0.0
    assert total_dmg[0, 2].item() == 0.0


def test_box_collision_deals_damage_and_kills_projectile():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)

    state.box_pos[0, 0] = torch.tensor([10.0, 5.0])
    state.box_alive[0, 0] = True
    state.box_hp[0, 0] = 999999.0
    state.box_max_hp[0, 0] = 999999.0

    _spawn_single_bolt(state, cfg, params, [5.0, 5.0], [1.0, 0.0])

    total_box_dmg = 0.0
    for _ in range(50):
        if not torch.any(state.prj_alive):
            break
        _, _, dmg_box, _, _ = proj.step_projectiles(state, bank, params, cfg)
        total_box_dmg += dmg_box[0, 0].item()

    assert total_box_dmg > 0.0
    # The ROCKET is gone. A hazard sphere may remain in its place (Brock leaves one
    # wherever his rocket dies) -- that is a different class and a different
    # mechanic, so this assertion is about the PROJECTILE-class slot only.
    assert _live(state, ProjClass.PROJECTILE) == 0


def test_expiry_at_max_range_with_nothing_hit():
    cfg, params = _cfg_and_params(map_h=60, map_w=60)
    state = _fresh_state(cfg)
    tiles = _grid(60, 60)
    bank = _bank_from_grid(tiles)

    _spawn_single_bolt(state, cfg, params, [5.0, 30.0], [1.0, 0.0])
    attack_range = params.attack_range[0, int(Kind.BOT_SNIPER)].item()
    proj_speed = params.proj_speed[0, int(Kind.BOT_SNIPER)].item()
    n_ticks = int(attack_range / proj_speed / cfg.dt) + 5

    for _ in range(n_ticks):
        if not torch.any(state.prj_alive):
            break
        proj.step_projectiles(state, bank, params, cfg)

    # The ROCKET is gone. A hazard sphere may remain in its place (Brock leaves one
    # wherever his rocket dies) -- that is a different class and a different
    # mechanic, so this assertion is about the PROJECTILE-class slot only.
    assert _live(state, ProjClass.PROJECTILE) == 0


def test_artillery_shell_lands_past_wall_and_damages_unit_behind_it():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    tiles[:, 8] = Tile.WALL  # a wall between shooter and target
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 0] = torch.tensor([12.0, 5.0])  # hero, behind the wall
    state.ent_hp[0, 0] = 999999.0
    state.ent_max_hp[0, 0] = 999999.0

    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, 1] = True
    state.ent_kind[0, 1] = int(Kind.BOT_ARTILLERY)
    origins = state.ent_pos.clone()
    origins[0, 1] = torch.tensor([5.0, 5.0])
    target = torch.tensor([12.0, 5.0])  # land right on the hero
    aim_dir = torch.zeros(1, cfg.n_entities, 2)
    aim_dir[0, 1] = torch.tensor([1.0, 0.0])
    aim_point = origins.clone()
    aim_point[0, 1] = target
    kind = state.ent_kind
    damage = stats.effective_damage(kind, state.ent_cubes, params)
    proj.spawn_volley(state, fire_mask, origins, aim_dir, aim_point, kind, damage, params, cfg)

    total_dmg = torch.zeros(1, cfg.n_entities)
    for _ in range(200):
        if not torch.any(state.prj_alive):
            break
        dmg_ent, _, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
        total_dmg += dmg_ent

    assert total_dmg[0, 0].item() > 0.0  # damaged despite the wall in between


# ---- Grom-style lob: fixed flight time + split cross -----------------------------

def _spawn_shell(state, cfg, params, origin, target, owner=1):
    """One artillery shell from `origin` at `target`, straight through spawn_volley."""
    state.ent_kind[0, owner] = int(Kind.BOT_ARTILLERY)
    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, owner] = True
    origins = state.ent_pos.clone()
    origins[0, owner] = torch.tensor(origin)
    aim_point = origins.clone()
    aim_point[0, owner] = torch.tensor(target)
    aim_dir = torch.zeros(1, cfg.n_entities, 2)
    aim_dir[0, owner] = torch.tensor([1.0, 0.0])
    damage = stats.effective_damage(state.ent_kind, state.ent_cubes, params)
    proj.spawn_volley(state, fire_mask, origins, aim_dir, aim_point, state.ent_kind, damage, params, cfg)
    return damage[0, owner].item()


def _ticks_until_detonation(state, bank, params, cfg, limit=200):
    for tick in range(1, limit + 1):
        proj.step_projectiles(state, bank, params, cfg)
        if not bool((state.prj_alive[0] & (state.prj_class[0] == int(ProjClass.ARTILLERY))).any()):
            return tick
    return None


def test_lob_takes_the_same_time_at_any_distance():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    bank = _bank_from_grid(_grid(20, 20))
    flight = params.proj_flight_seconds[0, int(Kind.BOT_ARTILLERY)].item()

    ticks = []
    for target in ([6.0, 10.0], [12.0, 10.0]):  # 1 tile away, then 7
        state = _fresh_state(cfg)
        _spawn_shell(state, cfg, params, [5.0, 10.0], target)
        ticks.append(_ticks_until_detonation(state, bank, params, cfg))

    assert ticks[0] == ticks[1]  # distance-independent, which is the whole point
    assert abs(ticks[0] * cfg.dt - flight) <= cfg.dt


def test_lob_detonates_on_its_landing_point_not_past_it():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))

    target = [12.3, 10.0]
    _spawn_shell(state, cfg, params, [5.0, 10.0], target)
    _ticks_until_detonation(state, bank, params, cfg)

    # The shards mark where the shell went off: all four sit on the rim of the blast, so their
    # midpoint is the detonation point itself.
    shards = state.prj_pos[0][state.prj_alive[0]]
    assert shards.shape[0] == _grom_split_count(params)
    assert torch.allclose(shards.mean(dim=0), torch.tensor(target), atol=1e-4)


def test_detonation_splits_into_four_axis_aligned_half_damage_shards():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))

    full_damage = _spawn_shell(state, cfg, params, [5.0, 10.0], [12.0, 10.0])
    _ticks_until_detonation(state, bank, params, cfg)

    alive = state.prj_alive[0]
    assert int(alive.sum()) == _grom_split_count(params)
    # shards are ordinary projectiles, not lobbed ones
    assert bool((state.prj_class[0][alive] == int(ProjClass.PROJECTILE)).all())
    assert bool((state.prj_aoe[0][alive] == 0).all())      # ... with no blast of their own
    assert bool((state.prj_owner[0][alive] == 1).all())    # ... still owned by the shooter

    fraction = params.split_damage_fraction[0, int(Kind.BOT_ARTILLERY)].item()
    assert torch.allclose(state.prj_damage[0][alive], torch.full((4,), full_damage * fraction))

    # One shard per world axis, each carrying split_distance tiles of travel.
    #
    # Shard speed is DERIVED from split_distance / split_seconds, not read from proj_speed
    # (which only describes a kind's ordinary shots).
    split_distance = params.split_distance[0, int(Kind.BOT_ARTILLERY)].item()
    split_seconds = params.split_seconds[0, int(Kind.BOT_ARTILLERY)].item()
    speed = (split_distance / split_seconds if split_seconds > 0
             else params.proj_speed[0, int(Kind.BOT_ARTILLERY)].item())
    vel = state.prj_vel[0][alive]
    assert sorted(vel.tolist()) == sorted([[speed, 0.0], [-speed, 0.0], [0.0, speed], [0.0, -speed]])
    assert torch.allclose(state.prj_dist_left[0][alive], torch.full((4,), split_distance))


def test_shards_finish_their_path_in_split_seconds():
    """CHARACTER_DETAILS specifies the arms by DURATION ("~0.5 s to finish their paths"),
    so that duration -- not the derived speed -- is what this pins."""
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))

    _spawn_shell(state, cfg, params, [5.0, 10.0], [12.0, 10.0])
    _ticks_until_detonation(state, bank, params, cfg)
    assert int(state.prj_alive.sum()) == _grom_split_count(params)

    ticks = 0
    while bool(state.prj_alive.any()) and ticks < 200:
        proj.step_projectiles(state, bank, params, cfg)
        ticks += 1

    split_seconds = params.split_seconds[0, int(Kind.BOT_ARTILLERY)].item()
    assert abs(ticks * cfg.dt - split_seconds) <= 2 * cfg.dt, (
        f"shards took {ticks * cfg.dt:.2f}s to finish, expected ~{split_seconds}s"
    )


def test_shard_deals_half_damage_two_tiles_off_the_landing_point():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))

    landing = [12.0, 10.0]
    state.ent_pos[0, 0] = torch.tensor([12.0, 12.0])  # 2 tiles south: on an arm, off the blast
    full_damage = _spawn_shell(state, cfg, params, [5.0, 10.0], landing)

    total = 0.0
    for _ in range(100):
        if not torch.any(state.prj_alive):
            break
        dmg_ent, _, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
        total += dmg_ent[0, 0].item()

    fraction = params.split_damage_fraction[0, int(Kind.BOT_ARTILLERY)].item()
    assert abs(total - full_damage * fraction) < 1e-3


def test_direct_hit_is_not_also_hit_by_its_own_shards():
    """The blast disc and the four arms partition the ground -- standing on the landing tile
    costs you the shell's damage exactly once, not that plus four shards at point blank."""
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))

    state.ent_pos[0, 0] = torch.tensor([12.0, 10.0])
    full_damage = _spawn_shell(state, cfg, params, [5.0, 10.0], [12.0, 10.0])

    total = 0.0
    for _ in range(100):
        if not torch.any(state.prj_alive):
            break
        dmg_ent, _, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
        total += dmg_ent[0, 0].item()

    assert abs(total - full_damage) < 1e-3


def test_shards_are_stopped_by_walls():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    tiles[:, 14] = Tile.WALL  # 2 tiles east of the landing point
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 0] = torch.tensor([15.0, 10.0])  # behind that wall, inside the arm's reach
    _spawn_shell(state, cfg, params, [5.0, 10.0], [12.0, 10.0])

    total = 0.0
    for _ in range(100):
        if not torch.any(state.prj_alive):
            break
        dmg_ent, _, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
        total += dmg_ent[0, 0].item()

    assert total == 0.0  # the shell arcs over walls; its shards do not


def test_non_lobbed_expiry_never_splits():
    cfg, params = _cfg_and_params(map_h=60, map_w=60)
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(60, 60))

    _spawn_single_bolt(state, cfg, params, [5.0, 30.0], [1.0, 0.0])
    for _ in range(300):
        if not torch.any(state.prj_alive):
            break
        proj.step_projectiles(state, bank, params, cfg)

    # The ROCKET is gone. A hazard sphere may remain in its place (Brock leaves one
    # wherever his rocket dies) -- that is a different class and a different
    # mechanic, so this assertion is about the PROJECTILE-class slot only.
    assert _live(state, ProjClass.PROJECTILE) == 0


def test_lobbed_shell_ignores_walls_in_flight():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    tiles[:, 8] = Tile.WALL
    bank = _bank_from_grid(tiles)

    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, 1] = True
    state.ent_kind[0, 1] = int(Kind.BOT_ARTILLERY)
    origins = state.ent_pos.clone()
    origins[0, 1] = torch.tensor([5.0, 5.0])
    aim_dir = torch.zeros(1, cfg.n_entities, 2)
    aim_dir[0, 1] = torch.tensor([1.0, 0.0])
    aim_point = origins.clone()
    aim_point[0, 1] = torch.tensor([12.0, 5.0])
    kind = state.ent_kind
    damage = stats.effective_damage(kind, state.ent_cubes, params)
    proj.spawn_volley(state, fire_mask, origins, aim_dir, aim_point, kind, damage, params, cfg)

    # one tick: should have crossed x=8 (the wall column) without dying
    proj.step_projectiles(state, bank, params, cfg)
    proj.step_projectiles(state, bank, params, cfg)
    assert torch.any(state.prj_alive)  # still flying, unaffected by the wall


# ---- HAZARD class: Brock's lingering sphere ---------------------------------------

def _hazard_fixture(max_projectiles=16):
    cfg, params = _cfg_and_params(map_h=30, map_w=30, max_projectiles=max_projectiles)
    tiles = _grid(30, 30)
    return cfg, params, tiles


def _fire_rocket_and_settle(cfg, params, tiles, hero_at, shooter_at=(5.0, 15.0), ticks=120):
    """Fires one Brock rocket east and steps until the buffer empties. Returns
    (state, total_damage_per_entity, [(t, per_entity_damage), ...])."""
    state = _fresh_state(cfg)
    bank = _bank_from_grid(tiles)
    state.ent_max_hp.fill_(1e9)
    state.ent_hp.fill_(1e9)
    state.ent_pos[0, 1] = torch.tensor(list(shooter_at))
    state.ent_pos[0, 0] = torch.tensor(list(hero_at))

    _spawn_single_bolt(state, cfg, params, list(shooter_at), [1.0, 0.0])

    total = torch.zeros(1, cfg.n_entities)
    events = []
    for i in range(ticks):
        dmg_ent, _, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
        if float(dmg_ent[0].abs().sum()) > 0:
            events.append(((i + 1) * cfg.dt, dmg_ent[0].clone()))
        total += dmg_ent
        if not torch.any(state.prj_alive):
            break
    return state, total, events


def test_rocket_leaves_a_hazard_that_ticks_twice_then_vanishes():
    """CHARACTER_DETAILS: "after a hit, Brock's projectiles turn into a 0.75 radius sphere, which
    ticks twice over 4 seconds before disappearing, dealing 696 damage"."""
    cfg, params, tiles = _hazard_fixture()
    _state, total, events = _fire_rocket_and_settle(cfg, params, tiles, hero_at=(9.0, 15.0))

    impact = params.base_damage[0, int(Kind.BOT_SNIPER)].item()
    tick_dmg = params.on_hit_area_damage_fraction[0, int(Kind.BOT_SNIPER)].item() * params.base_damage[0, int(Kind.BOT_SNIPER)].item()
    n_ticks = int(params.on_hit_area_ticks[0, int(Kind.BOT_SNIPER)])
    interval = params.on_hit_area_interval[0, int(Kind.BOT_SNIPER)].item()

    assert len(events) == 1 + n_ticks, f"expected impact + {n_ticks} sphere ticks, got {events}"
    assert abs(float(events[0][1][0]) - impact) < 1e-3
    for k in range(1, len(events)):
        assert abs(float(events[k][1][0]) - tick_dmg) < 1e-3
        # Spacing is `interval` apart. Tolerance of 2 ticks: prj_age accumulates dt in float, so
        # by age 4s it trails the exact value by a hair and the floor() crossing lands one tick
        # late -- 2.05s / 4.05s rather than 2.00s / 4.00s.
        gap = events[k][0] - events[k - 1][0]
        expected = interval if k > 1 else events[1][0] - events[0][0]
        assert abs(gap - expected) < 2 * cfg.dt or k == 1

    assert abs(float(total[0, 0]) - (impact + n_ticks * tick_dmg)) < 1e-3


def test_hazard_never_damages_its_own_owner():
    """The shooter standing in his own sphere takes nothing from it -- unlike an artillery
    detonation, which can still catch its thrower."""
    cfg, params, tiles = _hazard_fixture()
    # Hero far away; the rocket expires at max range and drops a sphere the SHOOTER walks into.
    state, _total, _events = _fire_rocket_and_settle(cfg, params, tiles, hero_at=(28.0, 28.0), ticks=1)
    haz = state.prj_alive[0] & (state.prj_class[0] == int(ProjClass.HAZARD))

    # Put the owner inside whatever sphere exists (or make one by running to expiry first).
    state2 = _fresh_state(cfg)
    bank = _bank_from_grid(tiles)
    state2.ent_max_hp.fill_(1e9)
    state2.ent_hp.fill_(1e9)
    state2.ent_pos[0, 1] = torch.tensor([5.0, 15.0])
    state2.ent_pos[0, 0] = torch.tensor([28.0, 28.0])
    _spawn_single_bolt(state2, cfg, params, [5.0, 15.0], [1.0, 0.0])
    owner_damage = 0.0
    for _ in range(120):
        live = state2.prj_alive[0] & (state2.prj_class[0] == int(ProjClass.HAZARD))
        if bool(live.any()):
            j = int(torch.nonzero(live)[0])
            state2.ent_pos[0, 1] = state2.prj_pos[0, j].clone()   # owner stands in his own sphere
        dmg_ent, _, _, _, _ = proj.step_projectiles(state2, bank, params, cfg)
        owner_damage += float(dmg_ent[0, 1])
        if not torch.any(state2.prj_alive):
            break
    assert owner_damage == 0.0, "the owner took damage from his own lingering sphere"


def test_hazard_ignores_walls():
    """A sphere is a patch of ground, not a shot, so line of sight is irrelevant."""
    cfg, params, tiles = _hazard_fixture()
    tiles[15, 10] = Tile.WALL  # between shooter and where the hero stands
    _state, total, _events = _fire_rocket_and_settle(cfg, params, tiles, hero_at=(9.6, 15.4))
    assert float(total[0, 0]) > 0.0


def test_two_hazards_stack():
    """Two overlapping spheres deal both their ticks, 1392 per tick."""
    cfg, params, tiles = _hazard_fixture()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(tiles)
    state.ent_max_hp.fill_(1e9)
    state.ent_hp.fill_(1e9)
    state.ent_pos[0, 0] = torch.tensor([9.0, 15.0])

    # Two rockets converging on the hero from OPPOSITE sides. Both shooters must be well clear of
    # each other: `not_owner` only exempts a projectile from its own owner, so two shooters on the
    # same tile would simply shoot each other and the rockets would never reach the hero.
    state.ent_pos[0, 1] = torch.tensor([5.0, 15.0])
    state.ent_pos[0, 2] = torch.tensor([13.0, 15.0])
    _spawn_single_bolt(state, cfg, params, [5.0, 15.0], [1.0, 0.0], owner=1)
    _spawn_single_bolt(state, cfg, params, [13.0, 15.0], [-1.0, 0.0], owner=2)

    per_tick = []
    for _ in range(120):
        dmg_ent, _, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
        if float(dmg_ent[0, 0]) > 0:
            per_tick.append(float(dmg_ent[0, 0]))
        if not torch.any(state.prj_alive):
            break

    tick_dmg = params.on_hit_area_damage_fraction[0, int(Kind.BOT_SNIPER)].item() * params.base_damage[0, int(Kind.BOT_SNIPER)].item()
    assert any(abs(d - 2 * tick_dmg) < 1e-3 for d in per_tick), (
        f"no tick dealt 2x{tick_dmg}; per-tick damage was {per_tick}"
    )


def test_hazard_blocks_nothing_and_moves_nowhere():
    """A hazard must not behave like a projectile: it does not collide with units, walls or boxes,
    and it does not travel."""
    cfg, params, tiles = _hazard_fixture()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(tiles)
    state.ent_max_hp.fill_(1e9)
    state.ent_hp.fill_(1e9)
    state.ent_pos[0, 0] = torch.tensor([25.0, 25.0])
    _spawn_single_bolt(state, cfg, params, [5.0, 15.0], [1.0, 0.0])

    positions = []
    for _ in range(60):
        proj.step_projectiles(state, bank, params, cfg)
        live = state.prj_alive[0] & (state.prj_class[0] == int(ProjClass.HAZARD))
        if bool(live.any()):
            j = int(torch.nonzero(live)[0])
            positions.append(state.prj_pos[0, j].clone())
            assert float(state.prj_vel[0, j].abs().sum()) == 0.0, "a hazard has velocity"
    assert len(positions) > 1
    for pos in positions[1:]:
        assert torch.allclose(pos, positions[0]), "a hazard moved"


def test_a_kind_without_on_hit_area_leaves_nothing():
    """Every non-Brock kind resolves on_hit_area_radius to 0 and must behave exactly as before."""
    cfg, params, tiles = _hazard_fixture()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(tiles)
    state.ent_kind[0, 1] = int(Kind.BOT_RIFLE)
    state.ent_pos[0, 0] = torch.tensor([25.0, 25.0])
    _spawn_single_bolt(state, cfg, params, [5.0, 15.0], [1.0, 0.0], shooter_kind=Kind.BOT_RIFLE)

    for _ in range(120):
        proj.step_projectiles(state, bank, params, cfg)
        assert not bool((state.prj_class[0] == int(ProjClass.HAZARD))[state.prj_alive[0]].any()), (
            "a kind with on_hit_area_radius 0 left a hazard behind"
        )
        if not torch.any(state.prj_alive):
            break


# ---- gadget spinner ------------------------------------------------------------------------

def _spawn_spinner(state, cfg, params, origin, direction, travel, owner=0):
    """One gadget spinner from `owner` at `origin`, flying `travel` tiles along `direction`."""
    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, owner] = True
    origins = state.ent_pos.clone()
    origins[0, owner] = torch.tensor(origin)
    dirs = torch.zeros(1, cfg.n_entities, 2)
    dirs[0, owner] = torch.tensor(direction)
    travels = torch.zeros(1, cfg.n_entities)
    travels[0, owner] = travel
    damage = stats.effective_gadget_damage(state.ent_kind, state.ent_cubes, params)
    proj.spawn_gadget(state, fire_mask, origins, dirs, travels, damage, params, cfg)
    return damage[0, owner].item()


def test_spawn_gadget_writes_one_fully_defined_artillery_slot():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    state.ent_pos[0, 0] = torch.tensor([5.0, 5.0])

    _spawn_spinner(state, cfg, params, [5.0, 5.0], [0.0, 1.0], 1.5)

    alive = state.prj_alive[0]
    assert int(alive.sum()) == 1
    s = int(alive.nonzero()[0, 0])
    assert state.prj_pos[0, s].tolist() == [5.0, 5.0]
    assert state.prj_target[0, s].tolist() == pytest.approx([5.0, 6.5], abs=1e-6)
    # travel / gadget_flight_seconds = 1.5 / 0.2 = 7.5, times the 1.0001 landing overshoot
    # (projectiles._LANDING_OVERSHOOT) that makes the landing tick deterministic. Pinned tight so
    # the overshoot is a tested property, not something a loose tolerance happens to admit.
    assert state.prj_vel[0, s].tolist() == pytest.approx([0.0, 7.50075], abs=1e-5)
    assert state.prj_dist_left[0, s].item() == pytest.approx(1.5, abs=1e-5)
    assert state.prj_damage[0, s].item() == 2000.0
    assert state.prj_radius[0, s].item() == 0.0
    assert state.prj_aoe[0, s].item() == 1.0
    assert state.prj_age[0, s].item() == 0.0
    assert int(state.prj_owner[0, s]) == 0
    assert int(state.prj_kind[0, s]) == int(Proj.GADGET_SPINNER) == 7
    assert int(state.prj_class[0, s]) == int(ProjClass.ARTILLERY)
    assert not bool(state.prj_pierce[0, s])


def test_spawn_gadget_with_zero_travel_still_writes_a_slot_that_detonates_on_tick_one():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_max_hp.fill_(1e9)
    state.ent_hp.fill_(1e9)
    state.ent_pos[0, 0] = torch.tensor([5.0, 5.0])
    state.ent_pos[0, 1] = torch.tensor([5.3, 5.0])  # standing on the hero: 0.3 < the 1.0 radius

    _spawn_spinner(state, cfg, params, [5.0, 5.0], [1.0, 0.0], 0.0)
    assert int(state.prj_alive.sum()) == 1
    assert state.prj_vel[0].abs().max().item() == 0.0

    dmg_ent, _, _, _, _ = proj.step_projectiles(state, bank, params, cfg)

    assert int(state.prj_alive.sum()) == 0  # detonated in place on its first tick
    assert dmg_ent[0, 1].item() == 2000.0


def test_spinner_blast_at_the_owners_feet_deals_0_to_the_owner():
    """No self-damage, by KIND -- compare the Grom test right below."""
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_max_hp.fill_(1e9)
    state.ent_hp.fill_(1e9)
    state.ent_pos[0, 0] = torch.tensor([5.0, 5.0])
    state.ent_pos[0, 1] = torch.tensor([5.5, 5.0])

    _spawn_spinner(state, cfg, params, [5.0, 5.0], [1.0, 0.0], 0.0)
    dmg_ent, dmg_by, _, _, _ = proj.step_projectiles(state, bank, params, cfg)

    assert dmg_ent[0, 0].item() == 0.0     # the owner, standing on the landing point
    assert dmg_ent[0, 1].item() == 2000.0  # the enemy 0.5 tiles away
    assert dmg_by[0, 0, 0].item() == 0.0


def test_grom_shell_at_the_owners_feet_still_hits_the_owner():
    """Artillery self-detonation is decided per kind, and Grom is not
    the spinner. 2080 is BOT_ARTILLERY base_damage at 0 cubes and enemy_damage_mult 1.0."""
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_max_hp.fill_(1e9)
    state.ent_hp.fill_(1e9)
    state.ent_pos[0, 1] = torch.tensor([5.0, 5.0])
    state.ent_pos[0, 0] = torch.tensor([15.0, 15.0])  # hero well clear of it

    _spawn_shell(state, cfg, params, [5.0, 5.0], [5.0, 5.0], owner=1)
    dmg_ent, dmg_by, _, _, charge_hit = proj.step_projectiles(state, bank, params, cfg)

    assert dmg_ent[0, 1].item() == 2080.0
    assert dmg_by[0, 1, 1].item() == 2080.0
    assert bool(charge_hit[0, 1, 1])  # and it even charges


def test_charge_hit_excludes_the_spinner_but_keeps_artillery():
    """`charge_hit` is `dmg_by > 0` minus the gadget. Two blasts in the air: the
    spinner on bot 1 (damage yes, charge no) and a Grom shell on the hero (both yes). Bot 1 stands
    2.0 tiles from the hero on the diagonal: at the spinner's max range, outside Grom's 0.6-tile
    blast, and off the axes his 1.2-tile shards fly along, so his shell charges exactly once."""
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_max_hp.fill_(1e9)
    state.ent_hp.fill_(1e9)
    state.ent_pos[0, 0] = torch.tensor([5.0, 5.0])
    state.ent_pos[0, 1] = torch.tensor([5.0 + 1.41421356, 5.0 + 1.41421356])
    state.ent_pos[0, 2] = torch.tensor([12.0, 12.0])  # Grom, far from both blasts

    _spawn_spinner(state, cfg, params, [5.0, 5.0], [0.70710678, 0.70710678], 2.0)  # on bot 1
    _spawn_shell(state, cfg, params, [12.0, 12.0], [5.0, 5.0], owner=2)          # on the hero
    seen = torch.zeros(cfg.n_entities, cfg.n_entities)
    charged = torch.zeros(cfg.n_entities, cfg.n_entities, dtype=torch.bool)
    for _ in range(80):  # spinner lands on tick 4, Grom's on tick 25, his shards expire after
        _dmg_ent, dmg_by, _, _, charge_hit = proj.step_projectiles(state, bank, params, cfg)
        seen += dmg_by[0]
        charged |= charge_hit[0]
        if not bool(state.prj_alive.any()):
            break

    assert seen[0, 1].item() == 2000.0
    assert not bool(charged[0, 1])   # the spinner charged nothing
    assert seen[2, 0].item() == 2080.0
    assert bool(charged[2, 0])       # the Grom shell charged as always
    assert int(charged.sum()) == 1


def test_charge_hit_keeps_an_ordinary_volley():
    """A Brock rocket (BOT_SNIPER, base_damage 2320) into the hero: damage and charge agree."""
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_max_hp.fill_(1e9)
    state.ent_hp.fill_(1e9)
    state.ent_pos[0, 0] = torch.tensor([8.0, 5.0])
    state.ent_pos[0, 1] = torch.tensor([5.0, 5.0])

    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, 1] = True
    aim_dir = torch.zeros(1, cfg.n_entities, 2)
    aim_dir[0, 1] = torch.tensor([1.0, 0.0])
    damage = stats.effective_damage(state.ent_kind, state.ent_cubes, params)
    proj.spawn_volley(state, fire_mask, state.ent_pos.clone(), aim_dir, state.ent_pos.clone(),
                      state.ent_kind, damage, params, cfg)

    hit_dmg, hit_charge = 0.0, False
    for _ in range(40):
        _, dmg_by, _, _, charge_hit = proj.step_projectiles(state, bank, params, cfg)
        if dmg_by[0, 1, 0].item() > 0:
            hit_dmg, hit_charge = dmg_by[0, 1, 0].item(), bool(charge_hit[0, 1, 0])
            break

    assert hit_dmg == 2320.0
    assert hit_charge


# ---- the fast paths against their dense references ---------------------------------------------

_STATE_FIELDS = ("prj_pos", "prj_vel", "prj_target", "prj_dist_left", "prj_damage", "prj_radius",
                 "prj_aoe", "prj_age", "prj_owner", "prj_kind", "prj_class", "prj_pierce",
                 "prj_alive", "prj_hits")
_STEP_OUTPUTS = ("dmg_ent", "dmg_by", "dmg_box", "heal_ent", "charge_hit")


def _busy_state(cfg, params, n_envs, gen):
    """A crowded mid-match projectile buffer on a wall-heavy map: every class, piercing and not,
    shells landing and spheres firing this tick, many slots near boxes and some on the border
    tiles, boxes on distinct tile centres (as spawn_boxes places them). Every stat stays inside
    the spec's own bounds, which is the range the budgets are derived for. Damage is whole
    numbers, so sums agree exactly in any order and a comparison can use torch.equal."""
    H, W = cfg.map_h, cfg.map_w
    N, P, B, E = n_envs, cfg.max_projectiles, cfg.max_boxes, cfg.n_entities
    rand = lambda *shape: torch.rand(*shape, generator=gen)  # noqa: E731

    tiles = _grid(H, W)
    walls = rand(H, W) < 0.18
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = False
    tiles[walls] = int(Tile.WALL)

    wh = torch.tensor([float(W), float(H)])
    state = _fresh_state(cfg, n_envs=N)
    state.ent_kind[:, 1:] = torch.randint(1, len(Kind), (N, E - 1), generator=gen)
    state.ent_pos.copy_(1.5 + rand(N, E, 2) * (wh - 3.0))
    state.ent_alive.copy_(rand(N, E) < 0.85)
    state.ent_max_hp.fill_(1e9)
    state.ent_hp.fill_(1e9)

    interior = torch.stack([torch.randperm((H - 2) * (W - 2), generator=gen)[:B] for _ in range(N)])
    box_tile = torch.stack([interior % (W - 2) + 1, interior // (W - 2) + 1], dim=-1)
    state.box_pos.copy_(box_tile.to(torch.float32) + 0.5)
    state.box_alive.copy_(rand(N, B) < 0.8)
    state.box_max_hp.fill_(1e9)
    state.box_hp.fill_(1e9)

    cls = torch.full((N, P), int(ProjClass.PROJECTILE), dtype=torch.int64)
    roll = rand(N, P)
    cls[roll > 0.55] = int(ProjClass.ARTILLERY)
    cls[roll > 0.85] = int(ProjClass.HAZARD)
    projectile = cls == int(ProjClass.PROJECTILE)
    pierce = projectile & (rand(N, P) < 0.15)

    near_box = state.box_pos.gather(1, torch.randint(0, B, (N, P, 1), generator=gen).expand(N, P, 2))
    pos = torch.where((rand(N, P) < 0.6).unsqueeze(-1), near_box + (rand(N, P, 2) - 0.5) * 3.0,
                      rand(N, P, 2) * wh)
    pos = torch.minimum(pos, wh - 0.01).clamp(min=0.01)

    # The fastest a wall or a box can stop is the budget's own bound; a super or a shell may go
    # faster (neither is stopped, so neither is covered by it).
    stoppable = projectile & ~pierce
    speed = torch.where(stoppable, rand(N, P) * params.shot_step_tiles / cfg.dt, rand(N, P) * 20.0)
    angle = rand(N, P) * 6.2831853
    vel = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1) * speed.unsqueeze(-1)
    hazard = cls == int(ProjClass.HAZARD)
    vel[hazard] = 0.0
    # Half the shells land this tick: their target sits within one tick's travel ahead.
    ahead = torch.where((rand(N, P) < 0.5).unsqueeze(-1), vel * cfg.dt * rand(N, P, 1), vel * 3.0)
    target = torch.where((cls == int(ProjClass.ARTILLERY)).unsqueeze(-1), pos + ahead, rand(N, P, 2) * wh)

    widest_shot = float(params.proj_radius.max())
    widest_blast = max(float(params.aoe_radius.max()), float(params.gadget_radius.max()),
                       float(params.on_hit_area_radius.max()))
    state.prj_pos.copy_(pos)
    state.prj_vel.copy_(vel)
    state.prj_target.copy_(torch.minimum(target, wh - 0.5).clamp(min=0.5))
    state.prj_dist_left.copy_(rand(N, P) * 6.0)
    state.prj_damage.copy_(torch.randint(1, 3000, (N, P), generator=gen).to(torch.float32))
    state.prj_radius.copy_(rand(N, P) * widest_shot)
    state.prj_aoe.copy_(torch.where(cls == int(ProjClass.PROJECTILE), torch.zeros(N, P), rand(N, P) * widest_blast))
    # Spheres sit a hair under an interval multiple, so plenty fire this tick.
    state.prj_age.copy_(torch.where(hazard, 2.0 - rand(N, P) * cfg.dt, rand(N, P) * 5.0))
    state.prj_owner.copy_(torch.randint(0, E, (N, P), generator=gen))
    state.prj_kind.copy_(torch.randint(1, len(Proj), (N, P), generator=gen))
    state.prj_class.copy_(cls)
    state.prj_pierce.copy_(pierce)
    state.prj_alive.copy_(rand(N, P) < 0.7)
    state.prj_hits.copy_(rand(N, P, E) < 0.1)
    return state, _bank_from_grid(tiles)


def _step_both(state, bank, params, cfg, **reference):
    """step_projectiles on two copies of `state`: as configured, and with `reference`'s params
    overrides. Returns ((outputs, state), (outputs, state))."""
    got_state, want_state = copy.deepcopy(state), copy.deepcopy(state)
    got = proj.step_projectiles(got_state, bank, params, cfg)
    saved = {name: getattr(params, name) for name in reference}
    for name, value in reference.items():
        setattr(params, name, value)
    try:
        want = proj.step_projectiles(want_state, bank, params, cfg)
    finally:
        for name, value in saved.items():
            setattr(params, name, value)
    return (got, got_state), (want, want_state)


def _assert_same_step(got, want, label):
    (got_out, got_state), (want_out, want_state) = got, want
    for name, a, b in zip(_STEP_OUTPUTS, got_out, want_out):
        assert torch.equal(a, b), f"{label}: {name} differs"
    for name in _STATE_FIELDS:
        assert torch.equal(getattr(got_state, name), getattr(want_state, name)), f"{label}: {name} differs"


def test_projectile_wall_march_budget_matches_full_ray():
    """step_projectiles marches walls for `params.shot_step_tiles` instead of
    the full `cfg.max_ray_tiles`. That is only sound because no slot whose `wall_hit` is read moves
    further than the budget in one tick, so this pins that the whole tick -- damage, kills, splits,
    spheres, every prj_ field -- matches the full ray on a wall-heavy map."""
    cfg, params = _cfg_and_params(n_envs=4, max_projectiles=64, max_boxes=24)
    assert terrain.ray_steps_for(params.shot_step_tiles, cfg) < cfg.ray_steps, (
        "the budget is not shorter than the full ray -- no saving, and this test passes vacuously")

    gen = torch.Generator().manual_seed(0)
    for trial in range(12):
        state, bank = _busy_state(cfg, params, 4, gen)
        got, want = _step_both(state, bank, params, cfg, shot_step_tiles=cfg.max_ray_tiles)
        _assert_same_step(got, want, f"trial {trial}")


def test_box_candidates_match_testing_every_box():
    """step_projectiles tests each slot against the boxes on the 3x3 tiles
    around it (the `_box_grid` lookup), not all B. Pinned against the every-box fallback, which a
    reach wider than the map forces, over crowded states where plenty of shots hit boxes, shells
    and spheres catch them, and some slots sit on the border tiles (out-of-map neighbours)."""
    cfg, params = _cfg_and_params(n_envs=4, max_projectiles=64, max_boxes=24)
    cells = math.ceil(params.box_reach_tiles)
    assert (2 * cells + 1) ** 2 < cfg.max_boxes, "the grid path never runs -- this test proves nothing"

    gen = torch.Generator().manual_seed(1)
    hits = 0
    for trial in range(12):
        state, bank = _busy_state(cfg, params, 4, gen)
        got, want = _step_both(state, bank, params, cfg, box_reach_tiles=float(cfg.map_w + cfg.map_h))
        _assert_same_step(got, want, f"trial {trial}")
        hits += int((want[0][2] > 0).sum())
    assert hits > 50, f"only {hits} damaged boxes across all trials -- the states are too quiet to pin anything"


def _spawn_splits_reference(state, detonate, det_pos, params, cfg):
    """`_spawn_splits`' assignment as it was before the slot-side rewrite: alloc_slots' (N,P,K)
    claims, each written to its slot (a spare column takes the claims that did not land). The old
    code wrote through `_write_slots`, whose `old + (new - old)` can land one ulp off the value;
    this writes the value itself, as the new code does, so the comparison can be exact."""
    N, P = state.prj_pos.shape[:2]
    E = state.ent_pos.shape[1]
    K = proj.MAX_SPLITS
    owner_kind = torch.gather(state.ent_kind, 1, torch.clamp(state.prj_owner, min=0, max=E - 1))
    split_distance = stats.gather_kind(params.split_distance, owner_kind)
    split_fraction = stats.gather_kind(params.split_damage_fraction, owner_kind)
    split_count = torch.clamp(stats.gather_kind(params.split_count, owner_kind), 0, K)
    split_seconds = stats.gather_kind(params.split_seconds, owner_kind)
    shard_speed = torch.where(split_seconds > 0, split_distance / torch.clamp(split_seconds, min=proj._EPS),
                              stats.gather_kind(params.proj_speed, owner_kind))
    splits = detonate & (split_distance > 0) & (split_fraction > 0) & (split_count > 0)
    demand = torch.where(splits, split_count, torch.zeros_like(split_count))
    idx, ok = proj.alloc_slots(state.prj_alive, demand, K)

    dirs = proj._split_dir_table(det_pos.device)[split_count]  # (N,P,K,2)
    rim = state.prj_aoe + params.unit_radius.view(N, 1) + state.prj_radius + proj._EPS
    start = det_pos.unsqueeze(2) + dirs * rim.view(N, P, 1, 1)
    margin = cfg.los_step_tiles
    start = torch.stack([torch.clamp(start[..., 0], margin, cfg.map_w - margin),
                         torch.clamp(start[..., 1], margin, cfg.map_h - margin)], dim=-1)
    per_shell = lambda t: t.unsqueeze(-1).expand(N, P, K)  # noqa: E731
    new = {
        "prj_pos": start, "prj_vel": dirs * shard_speed.view(N, P, 1, 1), "prj_target": start,
        "prj_dist_left": per_shell(split_distance), "prj_damage": per_shell(state.prj_damage * split_fraction),
        "prj_radius": per_shell(state.prj_radius), "prj_aoe": torch.zeros(N, P, K),
        "prj_age": torch.zeros(N, P, K), "prj_owner": per_shell(state.prj_owner),
        "prj_kind": per_shell(state.prj_kind),
        "prj_class": torch.full((N, P, K), int(ProjClass.PROJECTILE), dtype=torch.int64),
        "prj_pierce": torch.zeros(N, P, K, dtype=torch.bool), "prj_alive": torch.ones(N, P, K, dtype=torch.bool),
    }
    slot = torch.where(ok, idx, P).reshape(N, P * K)
    for name, values in new.items():
        field = getattr(state, name)
        flat = values.reshape(N, P * K, *field.shape[2:]).to(field.dtype)
        index = slot if field.dim() == 2 else slot.unsqueeze(-1).expand(N, P * K, field.shape[-1])
        spare = torch.cat([field, field[:, :1]], dim=1).scatter_(1, index, flat)
        field.copy_(spare[:, :P])


def test_spawn_splits_matches_alloc_slots_assignment():
    """_spawn_splits finds each free slot's shard from the slot side instead
    of scattering alloc_slots' (N,P,K) claims. Pinned against the claim-side reference over busy
    buffers -- Grom and Spike shells detonating side by side, freed slots reused, and buffers too
    full for every shard (the tail must drop exactly as alloc_slots drops it)."""
    cfg, params = _cfg_and_params(n_envs=6, max_projectiles=48)
    N, P, E = 6, cfg.max_projectiles, cfg.n_entities
    gen = torch.Generator().manual_seed(2)
    landed = dropped = 0
    splitters = torch.tensor([int(Kind.BOT_ARTILLERY), int(Kind.BOT_SPIKE)])
    for trial, occupancy in enumerate((0.1, 0.3, 0.5, 0.7, 0.85, 0.95) * 2):
        state, _ = _busy_state(cfg, params, N, gen)
        # Every bot a splitter, so shells from both rings detonate together (the hero's split nothing).
        state.ent_kind[:, 1:] = splitters[torch.randint(0, 2, (N, E - 1), generator=gen)]
        state.prj_alive.copy_(torch.rand(N, P, generator=gen) < occupancy)
        detonate = ~state.prj_alive & (torch.rand(N, P, generator=gen) < 0.5)
        det_pos = torch.rand(N, P, 2, generator=gen) * cfg.map_w

        got, want = copy.deepcopy(state), copy.deepcopy(state)
        proj._spawn_splits(got, detonate, det_pos, params, cfg)
        _spawn_splits_reference(want, detonate, det_pos, params, cfg)
        for name in _STATE_FIELDS:
            assert torch.equal(getattr(got, name), getattr(want, name)), f"trial {trial}: {name} differs"

        new = int(want.prj_alive.sum() - state.prj_alive.sum())
        demand = torch.where(detonate, stats.gather_kind(params.split_count, torch.gather(state.ent_kind, 1, state.prj_owner)), 0)
        landed += new
        dropped += int(demand.sum()) - new
    assert landed > 100 and dropped > 100, (landed, dropped)
