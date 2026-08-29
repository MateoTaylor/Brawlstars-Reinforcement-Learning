from pathlib import Path
import math

import torch
import yaml

from brawl_sim.config import load_config, build_params
from brawl_sim.constants import Kind, Tile, TILE_BLOCKS_UNIT
from brawl_sim.core import movement, terrain
from brawl_sim.core.state import allocate

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


class _FakeBank:
    def __init__(self, blocks_unit):
        self.blocks_unit = blocks_unit


def _grid(h, w, fill=Tile.FLOOR):
    t = torch.full((h, w), int(fill), dtype=torch.int64)
    t[0, :] = Tile.WALL
    t[-1, :] = Tile.WALL
    t[:, 0] = Tile.WALL
    t[:, -1] = Tile.WALL
    return t


def _bank_from_grid(tiles):
    return _FakeBank(blocks_unit=TILE_BLOCKS_UNIT[tiles].unsqueeze(0))


def _cfg_and_params(n_envs=1, map_h=30, map_w=30, extra_overrides=None):
    overrides = {"world": {"map_h": map_h, "map_w": map_w}}
    if extra_overrides:
        overrides = {**overrides, **extra_overrides}
    cfg = load_config(CONFIGS / "default.yaml", overrides=overrides)
    spec = {
        **yaml.safe_load((CONFIGS / "default.yaml").read_text()),
        **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    return cfg, params


def _fresh_state(cfg, n_envs=1):
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[:, 0] = int(Kind.HERO_MORTIS)
    for e in range(1, cfg.n_entities):
        state.ent_kind[:, e] = int(Kind.BOT_SNIPER)
    state.map_id.fill_(0)
    return state


# ---- walking into a wall ------------------------------------------------------

def test_walking_into_wall_stops_and_slides():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    tiles[:, 6] = Tile.WALL  # a wall column at x in [6, 7)
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 0] = torch.tensor([5.5, 5.5])
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 1.0])  # diagonal into the wall

    for _ in range(20):
        movement.apply_movement(state, move_dir, bank, params, cfg)

    # x is stopped near the wall face (x=6.0), y keeps advancing -- the "slide"
    assert state.ent_pos[0, 0, 0].item() < 6.0
    assert state.ent_pos[0, 0, 0].item() > 5.4
    assert state.ent_pos[0, 0, 1].item() > 6.0  # y moved well past its start


def test_walking_straight_into_wall_stops_dead():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    tiles[:, 6] = Tile.WALL
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 0] = torch.tensor([5.5, 5.5])
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])

    for _ in range(20):
        movement.apply_movement(state, move_dir, bank, params, cfg)

    assert state.ent_pos[0, 0, 1].item() == 5.5  # no y drift
    assert state.ent_pos[0, 0, 0].item() < 6.0


# ---- units do NOT collide with each other ---------------------------------------
#
# These replace an earlier "soft separation" section. Entities are deliberately non-colliding
# (see core/movement's module docstring): terrain stops a walk, bodies never do. The tests below
# pin that as a REQUIREMENT rather than leaving it as the absence of a feature, because the
# natural instinct on reading `unit_radius` in a movement file is to add separation back.


def _pair_setup(cfg, walker_x=10.0, other_x=13.0):
    """Entities 0 and 1 on the y=15 line; everyone else parked far away in a column so they
    cannot interact with the pair under test."""
    state = _fresh_state(cfg)
    state.ent_pos[0, 0] = torch.tensor([walker_x, 15.0])
    state.ent_pos[0, 1] = torch.tensor([other_x, 15.0])
    for e in range(2, cfg.n_entities):
        state.ent_pos[0, e] = torch.tensor([2.0, 2.0 + e])
    return state


def test_a_unit_walks_clean_through_another_one():
    """The headline requirement: bodies are not obstacles. The walker must end up PAST the unit
    it walked into, and must pass through the far side rather than stopping at contact."""
    cfg, params = _cfg_and_params(map_h=30, map_w=30)
    state = _pair_setup(cfg)
    bank = _bank_from_grid(_grid(30, 30))

    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])  # straight at entity 1
    for _ in range(120):
        movement.apply_movement(state, move_dir, bank, params, cfg)

    assert state.ent_pos[0, 0, 0].item() > 16.0, \
        f"the walker was stopped by another unit's body (ended at x={state.ent_pos[0, 0, 0].item():.2f})"


def test_walking_through_a_unit_does_not_shove_it():
    """The other half of pass-through, and the half the old separation spring got wrong: a unit
    being walked through is not pushed, nudged, or displaced at all. It has no move input, so it
    must not move by so much as a float."""
    cfg, params = _cfg_and_params(map_h=30, map_w=30)
    state = _pair_setup(cfg)
    bank = _bank_from_grid(_grid(30, 30))

    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])
    other_start = state.ent_pos[0, 1].clone()
    for _ in range(120):
        movement.apply_movement(state, move_dir, bank, params, cfg)

    assert torch.equal(state.ent_pos[0, 1], other_start), \
        f"a unit with no move input was displaced from {other_start.tolist()} to {state.ent_pos[0, 1].tolist()}"


def test_units_may_occupy_the_same_point():
    """Coincident units stay coincident. This is the exact case the removed spring existed to
    break up, so it is the sharpest regression test against the spring coming back."""
    cfg, params = _cfg_and_params(map_h=30, map_w=30)
    state = _pair_setup(cfg, walker_x=15.0, other_x=15.0)  # exactly coincident
    bank = _bank_from_grid(_grid(30, 30))

    move_dir = torch.zeros(1, cfg.n_entities, 2)  # nobody is walking
    for _ in range(50):
        movement.apply_movement(state, move_dir, bank, params, cfg)

    d = (state.ent_pos[0, 0] - state.ent_pos[0, 1]).norm().item()
    assert d == 0.0, f"coincident units drifted apart by {d:.6f} -- something is separating them"


def test_the_whole_roster_can_stack_on_one_point():
    """Every entity in the game piled onto a single tile, walking into each other from all
    directions, must simply stack. The old spring's worst case; now a non-event."""
    cfg, params = _cfg_and_params(map_h=30, map_w=30, extra_overrides={"entities": {"n_enemies": 9}})
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(30, 30))

    # Ring of entities all walking inward at the same point.
    n = cfg.n_entities
    move_dir = torch.zeros(1, n, 2)
    for e in range(n):
        angle = 2 * math.pi * e / n
        state.ent_pos[0, e] = torch.tensor([15.0 + 3.0 * math.cos(angle), 15.0 + 3.0 * math.sin(angle)])
        move_dir[0, e] = torch.tensor([-math.cos(angle), -math.sin(angle)])

    # Tracked across the whole run, not just at the end: with move_dir held constant they
    # converge on the centre, stack, and then keep right on walking out the far side -- which is
    # pass-through working, so the END state is spread out again. The moment that matters is the
    # tightest one.
    tightest = float("inf")
    for _ in range(120):
        movement.apply_movement(state, move_dir, bank, params, cfg)
        p = state.ent_pos[0]
        tightest = min(tightest, (p - p.mean(dim=0)).norm(dim=-1).max().item())

    # Threshold is `unit_radius`, derived rather than magic: every entity within one radius of
    # the centroid puts every PAIR within 2*unit_radius of each other, i.e. every body in the
    # game mutually overlapping -- precisely "they can occupy the same space".
    #
    # Not asserting exact coincidence, because per-kind walk speeds differ (core/stats
    # .effective_speed), so a ring launched simultaneously does not arrive simultaneously and
    # the fastest entities are already leaving as the slowest arrive. That staggering is a
    # property of the roster, not of collision.
    unit_radius = params.unit_radius[0].item()
    assert tightest < unit_radius,         f"units converging on one point refused to stack (tightest spread {tightest:.4f} >= {unit_radius})"


def test_walls_still_stop_a_walk():
    """Guard against over-correcting: removing BODY collision must not have touched TERRAIN
    collision. Kept next to the pass-through tests so the distinction stays visible."""
    cfg, params = _cfg_and_params(map_h=30, map_w=30)
    tiles = _grid(30, 30)
    tiles[:, 14] = Tile.WALL
    state = _pair_setup(cfg, walker_x=10.0, other_x=13.0)
    bank = _bank_from_grid(tiles)

    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])
    for _ in range(120):
        movement.apply_movement(state, move_dir, bank, params, cfg)

    # Passed through entity 1 at x=13, then stopped dead at the wall face at x=14.
    assert state.ent_pos[0, 0, 0].item() > 13.0, "the walker was blocked by a body, not the wall"
    assert state.ent_pos[0, 0, 0].item() < 14.0, "the walker went through a WALL"


def test_dashers_pass_through_bodies_too():
    """Mortis's dash through enemies is handled by hero.advance_dash, which never consulted
    bodies either. Asserted here so "dash passes through" and "walk passes through" are pinned
    by the same file: they are now the same rule, not two separate exemptions."""
    cfg, params = _cfg_and_params(map_h=30, map_w=30)
    state = _pair_setup(cfg, walker_x=13.0, other_x=13.2)
    state.ent_dash_t[0, 1] = 0.2  # entity 1 is mid-dash, sitting on entity 0
    bank = _bank_from_grid(_grid(30, 30))

    dasher_start = state.ent_pos[0, 1].clone()
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])
    for _ in range(20):
        movement.apply_movement(state, move_dir, bank, params, cfg)

    assert state.ent_pos[0, 0, 0].item() > dasher_start[0].item() + 0.5, "a mid-dash entity blocked a walker"
    assert torch.equal(state.ent_pos[0, 1], dasher_start), "apply_movement moved a dashing entity"


# ---- stress: never inside a wall -------------------------------------------------

def test_10k_random_ticks_never_inside_a_wall():
    cfg, params = _cfg_and_params(n_envs=4, map_h=25, map_w=25)
    state = _fresh_state(cfg, n_envs=4)
    tiles = _grid(25, 25)
    tiles[10, :] = Tile.WALL  # an internal wall to make bouncing off something likely
    tiles[10, 12] = Tile.FLOOR  # one gap so the map stays connected
    bank = _bank_from_grid(tiles)

    torch.manual_seed(0)
    state.ent_pos[:, :, 0] = torch.rand(4, cfg.n_entities) * 20 + 2
    state.ent_pos[:, :, 1] = torch.rand(4, cfg.n_entities) * 20 + 2

    radius = params.unit_radius.unsqueeze(-1)  # (N,1), broadcasts against (N,E)
    # A uniform-random initial y can land inside the injected wall row [10, 11) purely by luck of
    # the RNG draw -- how likely depends on cfg.n_entities (more entities -> more draws -> more
    # chances), which is exactly why this held at n_entities=7 and broke the moment n_entities
    # rose: NOT a movement bug, a test-setup gap. A real spawn always lands on a SPAWN tile
    # (spawn.sample_spawn_positions); this synthetic setup has no such guarantee, so enforce
    # tick-0 legality once, explicitly, rather than let the test's own pass/fail hinge on whether
    # a given seed happens to avoid the wall band for however many entities are configured.
    blocked0 = terrain.circle_blocked(bank.blocks_unit, state.map_id, state.ent_pos, radius, cfg)
    safe_y = torch.full_like(state.ent_pos[..., 1], 5.0)  # comfortably outside [10, 11)
    state.ent_pos[..., 1] = torch.where(blocked0, safe_y, state.ent_pos[..., 1])
    n_ticks = 10_000 // 4  # scale down per-env count; total move calls stay in the thousands
    for _ in range(n_ticks):
        move_dir = torch.randn(4, cfg.n_entities, 2)
        movement.apply_movement(state, move_dir, bank, params, cfg)
        blocked = terrain.circle_blocked(bank.blocks_unit, state.map_id, state.ent_pos, radius, cfg)
        assert not torch.any(blocked), "an entity ended up inside a blocks_unit tile"


# ---- facing --------------------------------------------------------------------

def test_facing_survives_idle_ticks():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])
    movement.apply_movement(state, move_dir, bank, params, cfg)
    facing_after_move = state.ent_facing[0, 0].clone()
    assert abs(facing_after_move.item()) < 1e-4  # facing east

    idle = torch.zeros(1, cfg.n_entities, 2)
    for _ in range(10):
        movement.apply_movement(state, idle, bank, params, cfg)
    assert torch.allclose(state.ent_facing[0, 0], facing_after_move)


# ---- dashers untouched -----------------------------------------------------------

def test_dashers_are_untouched_by_movement():
    cfg, params = _cfg_and_params(map_h=20, map_w=20)
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_dash_t[0, 0] = 0.2
    state.ent_facing[0, 0] = 0.75  # arbitrary, should also stay fixed per the active gate...
    before_pos = state.ent_pos[0, 0].clone()
    before_vel = state.ent_vel[0, 0].clone()

    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])  # would move if not dashing
    movement.apply_movement(state, move_dir, bank, params, cfg)

    assert torch.equal(state.ent_pos[0, 0], before_pos)
    assert torch.equal(state.ent_vel[0, 0], before_vel)
