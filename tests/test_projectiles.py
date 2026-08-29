from pathlib import Path

import torch
import yaml

from brawl_sim.config import load_config, build_params
from brawl_sim.constants import Kind, Proj, ProjClass, Tile, TILE_BLOCKS_PROJ
from brawl_sim.core import projectiles as proj
from brawl_sim.core import stats
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


def _cfg_and_params(n_envs=1, map_h=20, map_w=20, max_projectiles=16):
    cfg = load_config(CONFIGS / "default.yaml", overrides={
        "world": {"map_h": map_h, "map_w": map_w},
        "limits": {"max_projectiles": max_projectiles, "max_boxes": 8, "max_pickups": 8},
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

    Step C3b means "the shot is over" no longer implies "no slots are alive": Brock's rocket
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
    # One slot per pellet, whatever proj_count currently is (3 for Bo, 5 for Shelly after B2).
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
    # Evenly spaced across the FULL span, symmetric about aim_dir. Was written out as the literal
    # 3-element [-s/2, 0, +s/2]; linspace is the same statement for any odd or even count, which
    # is what let Step B2's move to 5 pellets be a config change rather than a test rewrite.
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
    # Read the volley width off params rather than pinning it: Step B2 took Shelly from 3 pellets
    # to 5, and this test is about ALLOCATION (two shooters must never claim the same slot), not
    # about how wide the fan happens to be.
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
    # wherever his rocket dies, Step C3b) -- that is a different class and a different
    # mechanic, so this assertion is about the PROJECTILE-class slot only.
    assert _live(state, ProjClass.PROJECTILE) == 0


def test_bolt_over_water_reaches_far_side():
    """Water blocks units but not projectiles (Notice 4).

    Start position, finish line and tick budget are all DERIVED from the shooter's own stats. They
    were hardcoded (spawn x=5.0, finish x>13.0, 100 ticks), which silently assumed
    `attack_range = 8.67`: a bolt fired from 5.0 expires at exactly 5.0 + range, so when Step B1
    moved Brock's range to 8.0 the bolt died at exactly 13.0 and the strict `> 13.0` failed by
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
    # Enough ticks to fly the full range, plus slack. proj_speed dropped 14.0 -> 5.33 in B1, which
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
        dmg_ent, dmg_by, dmg_box, _heal = proj.step_projectiles(state, bank, params, cfg)
        total_dmg += dmg_ent

    assert total_dmg[0, 0].item() > 0.0
    # The ROCKET is gone. A hazard sphere may remain in its place (Brock leaves one
    # wherever his rocket dies, Step C3b) -- that is a different class and a different
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
        dmg_ent, dmg_by, dmg_box, _heal = proj.step_projectiles(state, bank, params, cfg)
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
        dmg_ent, dmg_by, dmg_box, _heal = proj.step_projectiles(state, bank, params, cfg)
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
        _, _, dmg_box, _ = proj.step_projectiles(state, bank, params, cfg)
        total_box_dmg += dmg_box[0, 0].item()

    assert total_box_dmg > 0.0
    # The ROCKET is gone. A hazard sphere may remain in its place (Brock leaves one
    # wherever his rocket dies, Step C3b) -- that is a different class and a different
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
    # wherever his rocket dies, Step C3b) -- that is a different class and a different
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
        dmg_ent, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
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
    # Shard speed is DERIVED from split_distance / split_seconds as of Step C4, not read from
    # proj_speed (which now only describes a kind's ordinary shots). This read `proj_speed`
    # directly and so silently asserted the pre-C4 rule.
    split_distance = params.split_distance[0, int(Kind.BOT_ARTILLERY)].item()
    split_seconds = params.split_seconds[0, int(Kind.BOT_ARTILLERY)].item()
    speed = (split_distance / split_seconds if split_seconds > 0
             else params.proj_speed[0, int(Kind.BOT_ARTILLERY)].item())
    vel = state.prj_vel[0][alive]
    assert sorted(vel.tolist()) == sorted([[speed, 0.0], [-speed, 0.0], [0.0, speed], [0.0, -speed]])
    assert torch.allclose(state.prj_dist_left[0][alive], torch.full((4,), split_distance))


def test_shards_finish_their_path_in_split_seconds():
    """Step C4: CHARACTER_DETAILS specifies the arms by DURATION ("~0.5 s to finish their paths"),
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
        dmg_ent, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
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
        dmg_ent, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
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
        dmg_ent, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
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
    # wherever his rocket dies, Step C3b) -- that is a different class and a different
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


# ---- HAZARD class: Brock's lingering sphere (Step C3b) ----------------------------

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
        dmg_ent, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
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
    """D4. The shooter standing in his own sphere takes nothing from it -- unlike an artillery
    detonation, which can still catch its thrower (pre-existing behavior, deliberately unchanged)."""
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
        dmg_ent, _, _, _ = proj.step_projectiles(state2, bank, params, cfg)
        owner_damage += float(dmg_ent[0, 1])
        if not torch.any(state2.prj_alive):
            break
    assert owner_damage == 0.0, "the owner took damage from his own lingering sphere"


def test_hazard_ignores_walls():
    """D5: it is a patch of ground, not a shot, so line of sight is irrelevant."""
    cfg, params, tiles = _hazard_fixture()
    tiles[15, 10] = Tile.WALL  # between shooter and where the hero stands
    _state, total, _events = _fire_rocket_and_settle(cfg, params, tiles, hero_at=(9.6, 15.4))
    assert float(total[0, 0]) > 0.0


def test_two_hazards_stack():
    """D4: two overlapping spheres deal both their ticks, 1392 per tick."""
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
        dmg_ent, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
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
