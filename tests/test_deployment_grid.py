"""The egocentric view grid. BRAWL_DEPLOYMENT_DESIGN.md 6.2.

The load-bearing test is `test_the_grid_matches_the_sim_channel_for_channel`: it runs a real
`BrawlVecEnv`, hands `GridBuilder` a perfect-perception view of that env's own world, and asserts
the eight deployed planes equal `observation._build_grid`'s corresponding channels, cell for cell,
every decision. Everything the module claims -- the crop origin, the channel order, counts rather
than booleans, the hero constant, the terrain lookup -- is a claim about matching that function.

**"Perfect perception" is the input, not the thing under test.** The occupancy map is seeded from
the sim's own padded tile bank and the gas map from the sim's own zone rectangle, because the
question here is placement: given the right world, does this put it in the right cells. The two
places where real perception is NOT perfect get their own tests instead -- an UNKNOWN cell (which
the real map has 1.5-8.8% of, always below the hero) and a coasted track (which is invisible to a
`fair` observation and must not be deposited).

The gas seeding is cumulative on purpose. `grid.py` latches gassed cells forever on the grounds
that the poison only ever grows; OR-ing each tick's rectangle and then comparing against the sim's
CURRENT rectangle is a live check of that assumption, and would fail if the sim ever un-gassed a
cell.
"""
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np
import pytest
import torch
import yaml

from brawl_sim.bots import perception
from brawl_sim.config import load_config
from brawl_sim.constants import Tile
from brawl_sim.env import BrawlVecEnv
from brawl_deployment.perception.grid import GasMap, GridBuilder, GridSpec
from brawl_deployment.perception.projectiles import Projectile
from brawl_deployment.perception.tracker import Track
from brawl_vision.terrain.labeling import CLASS_INDEX, CLASSES
from brawl_vision.terrain.occupancy import UNKNOWN, OccupancyMap

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

# `_build_grid`'s channel order, restated here so the test compares against the layout rather than
# against grid.py's own idea of it. obs_select keeps the same table privately.
SIM_CHANNEL = {"blocks_unit": 0, "blocks_projectile": 1, "is_bush": 2, "is_water": 3, "in_zone": 4,
               "enemy_any": 5, "enemy_revealed": 6, "enemy_hidden": 7, "hero": 8, "box": 9,
               "pickup": 10, "projectile": 11}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _occupancy(height: int = 128, width: int = 128) -> OccupancyMap:
    return OccupancyMap(height=height, width=width)


def _spec(**kw) -> GridSpec:
    base = GridSpec.load()
    return GridSpec(**{"channels": base.channels, "view_h": base.view_h,
                       "view_w": base.view_w, **kw})


def _builder(occupancy=None, **kw) -> GridBuilder:
    return GridBuilder(occupancy if occupancy is not None else _occupancy(), **kw)


def _put(occ: OccupancyMap, x: int, y: int, tile: Tile, votes: int = 5) -> None:
    """Vote a world tile into the occupancy map hard enough that `best()` returns it."""
    ox, oy = occ.origin
    occ.votes[y - oy, x - ox, CLASS_INDEX[tile]] += votes


def _track(x, y, *, seen_now=True, slot=0) -> Track:
    return Track(id=slot, slot=slot, label="enemy", pos=(x, y), seen_now=seen_now)


def _sim_env(map_name="island_invasion", seed=0):
    """One env on a real map, so water, fences and bush all appear in the terrain planes.

    `island_invasion` is pinned rather than sampled: the parity assertion is cell-exact, and a
    rotation would make a failure depend on which map the seed happened to draw.
    """
    overrides = {"world": {"maps": [map_name], "map_selection": "fixed", "fixed_map": map_name},
                 "sim": {"max_episode_steps": 2000}}
    cfg = load_config(CONFIGS / "default.yaml", overrides=overrides)
    return BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=seed, autoreset=False, verbose=False)


def _frame_offset(env) -> tuple[int, int]:
    """Sim map tile -> deployment world tile.

    The occupancy grid is 128x128 centred on WHEREVER TRACKING STARTED (`OccupancyMap.origin`),
    not on a map with known absolute coordinates -- there is no pre-built map to align to. So the
    sim's tile (0, 0) is not the deployment frame's (0, 0), and a test that pretends it is fits by
    luck. Everything handed to the builder goes through this offset, which puts the map centre at
    the grid centre and incidentally checks that `GridBuilder` is frame-agnostic.
    """
    return (-env.cfg.map_w // 2, -env.cfg.map_h // 2)


def _seed_from_bank(occ: OccupancyMap, env) -> None:
    """Fill the occupancy map from the sim's PADDED tile bank -- perfect perception of a bounded
    map, including the `Tile.WALL` border `maps/loader.py` pads with.

    The padding is `view_h // 2` by `view_w // 2` deep, which is exactly how far outside the map a
    hero-centred crop can reach, so this covers every cell the comparison can touch.
    """
    cfg = env.cfg
    pad_h, pad_w = cfg.view_h // 2, cfg.view_w // 2
    tiles = env.bank.pad_tiles[int(env.state.map_id[0])].numpy()
    ox, oy = occ.origin
    dx, dy = _frame_offset(env)
    for row in range(tiles.shape[0]):
        for col in range(tiles.shape[1]):
            tile = Tile(int(tiles[row, col]))
            # SPAWN and BOX are map-authoring markers the classifier has no class for; the sim's
            # own lookup tables treat both as walkable non-cover, i.e. exactly FLOOR.
            if tile not in CLASSES:
                tile = Tile.FLOOR
            occ.votes[row - pad_h + dy - oy, col - pad_w + dx - ox, CLASS_INDEX[tile]] += 5


def _seed_gas(gas: GasMap, env) -> None:
    """OR this tick's zone rectangle into the gas map, sampling cell CENTRES as `zone_grid` does."""
    dx, dy = _frame_offset(env)
    lo = env.state.zone_lo[0].numpy()
    hi = env.state.zone_hi[0].numpy()
    ox, oy = gas.origin
    xs = np.arange(gas.width) + ox - dx + 0.5
    ys = np.arange(gas.height) + oy - dy + 0.5
    outside_x = (xs < lo[0]) | (xs > hi[0])
    outside_y = (ys < lo[1]) | (ys > hi[1])
    gas.gassed |= outside_y[:, None] | outside_x[None, :]


def _crowd_the_hero(env, n: int = 4) -> None:
    """Move the first `n` bots into the hero's crop and charge its super.

    Left alone, ten brawlers spread over a 60x60 map do not meet inside forty decisions, and the
    parity run agrees on two planes of zeros -- which proves nothing about the planes that
    actually need checking. Writing `ent_pos` is enough: movement resolves out of whatever the
    landing tile turns out to be, and the bots do the rest by shooting at each other, which is
    where the `projectile` plane comes from. Mortis's own attack is a dash and spawns nothing,
    so the super is charged to give the hero a projectile of its own.
    """
    hero = env.state.ent_pos[0, 0].clone()
    for i in range(1, min(n, env.cfg.n_entities - 1) + 1):
        env.state.ent_pos[0, i] = hero + torch.tensor([float(i) - 2.5, float(i % 3) - 1.0])
    env.state.ent_super_charge[0, 0] = 1.0


def _sim_view(env) -> np.ndarray:
    """`(12, view_h, view_w)` uint8 -- the sim's own egocentric grid for env 0."""
    return env._build_observation()["view"][0].numpy()


def _tracks_and_projectiles(env):
    """The sim's entities dressed as this package's tracks. `enemy_revealed` is
    `alive & vis[hero, j]`, so the revealed set is what gets `seen_now`."""
    state = env.state
    dx, dy = _frame_offset(env)
    vis = perception.visibility(state, env.bank, env.params, env.cfg)
    revealed = (vis[0, 0, :] & state.ent_alive[0]).numpy()
    enemies = [_track(float(state.ent_pos[0, j, 0]) + dx, float(state.ent_pos[0, j, 1]) + dy,
                      seen_now=bool(revealed[j]), slot=j - 1)
               for j in range(1, state.ent_pos.shape[1])
               if bool(state.ent_alive[0, j])]
    projectiles = [Projectile(id=k, pos=(float(state.prj_pos[0, k, 0]) + dx,
                                         float(state.prj_pos[0, k, 1]) + dy))
                   for k in range(state.prj_pos.shape[1]) if bool(state.prj_alive[0, k])]
    return enemies, projectiles


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------

def test_the_spec_is_read_from_the_yaml_rather_than_restated():
    spec = GridSpec.load()
    raw = yaml.safe_load((CONFIGS / "agent_obs_deploy.yaml").read_text())
    grid = next(g for g in raw["groups"] if g["name"] == "grid")
    view = yaml.safe_load((CONFIGS / "default.yaml").read_text())["view"]
    assert spec.channels == tuple(grid["view_channels"])
    assert spec.shape == (len(grid["view_channels"]), view["height"], view["width"])
    assert spec.shape == (8, 13, 21)          # the shape the deploy run actually trained on


def test_the_hero_sits_at_the_exact_centre_of_an_odd_view():
    spec = GridSpec.load()
    assert spec.centre == (spec.view_h // 2, spec.view_w // 2) == (6, 10)


@pytest.mark.parametrize("channel,fragment", [
    ("box", "nothing detects a crate"),
    ("pickup", "power-cube pickup"),
    ("enemy_any", "cannot see"),
    ("enemy_hidden", "hidden"),
])
def test_a_channel_with_no_supplier_is_refused_by_name(channel, fragment):
    """Feeding zeros into a channel the policy was trained to read is the silent failure the whole
    agent_obs_deploy line of configs exists to prevent, so asking for one raises."""
    with pytest.raises(ValueError, match=fragment):
        _spec(channels=(channel,)).check()


def test_a_deploy3_checkpoint_is_refused_until_crates_and_cubes_have_a_supplier():
    """`configs/agent_obs_deploy3.yaml` trains on `box` and `pickup` ahead of their detector, on
    purpose. Deploying what it produces before that detector exists must fail at startup, naming
    the channel -- the alternative is a policy trained to read two planes and handed zeros in both.
    When a supplier lands, this test is the one to update, deliberately."""
    with pytest.raises(ValueError, match="no supplier at deploy time"):
        GridSpec.load(CONFIGS / "agent_obs_deploy3.yaml")


def test_a_typo_lists_what_this_builder_can_actually_fill():
    with pytest.raises(ValueError, match="unknown grid channel"):
        _spec(channels=("blocks_units",)).check()


# ---------------------------------------------------------------------------
# the parity test
# ---------------------------------------------------------------------------

def test_the_grid_matches_the_sim_channel_for_channel():
    env = _sim_env()
    env.reset()
    _crowd_the_hero(env)
    occ = _occupancy()
    _seed_from_bank(occ, env)
    builder = _builder(occ)
    spec = builder.spec

    rng = Random(0)
    seen = {ch: 0 for ch in spec.channels}
    for decision in range(40):
        view = _sim_view(env)
        _seed_gas(builder.gas, env)
        enemies, projectiles = _tracks_and_projectiles(env)
        dx, dy = _frame_offset(env)
        hero_pos = (float(env.state.ent_pos[0, 0, 0]) + dx,
                    float(env.state.ent_pos[0, 0, 1]) + dy)
        grid = builder.build(hero_pos, alive=bool(env.state.ent_alive[0, 0]),
                             enemies=enemies, projectiles=projectiles)

        for i, ch in enumerate(spec.channels):
            expected = view[SIM_CHANNEL[ch]]
            assert np.array_equal(grid[i], expected), (
                f"decision {decision}, channel {ch!r}\n"
                f"got\n{grid[i]}\nexpected\n{expected}")
            seen[ch] += int(grid[i].sum() > 0)

        action = torch.tensor([[rng.randrange(0, 17), rng.randrange(0, 3)]], dtype=torch.int64)
        _, _, terminated, truncated, _ = env.step(action)
        assert not bool(terminated[0] or truncated[0]), "episode ended mid-comparison"

    # Anti-vacuity: agreeing on eight planes of zeros would prove nothing, and one lucky cell is
    # barely better, so this counts DECISIONS on which each plane was non-empty. Measured on this
    # seed: terrain and hero 40/40, in_zone 39, projectile 22, enemy_revealed 18.
    thin = {ch: n for ch, n in seen.items() if n < 5}
    assert not thin, f"barely exercised: {thin} of 40 decisions (all: {seen})"


def test_the_parity_run_puts_the_hero_and_enemies_where_the_sim_does_after_it_moves():
    """The crop origin follows the hero, so a builder that ignored `hero_pos` would still pass a
    single-frame comparison. This asserts the hero actually travelled far enough to move it."""
    env = _sim_env()
    env.reset()
    builder = _builder()
    start = builder.origin_tile(env.state.ent_pos[0, 0].tolist())
    rng = Random(1)
    for _ in range(40):
        env.step(torch.tensor([[rng.randrange(1, 17), 0]], dtype=torch.int64))
    end = builder.origin_tile(env.state.ent_pos[0, 0].tolist())
    assert start != end


# ---------------------------------------------------------------------------
# the crop
# ---------------------------------------------------------------------------

def test_the_crop_origin_is_the_hero_tile_minus_half_the_view():
    b = _builder()
    assert b.origin_tile((10.9, 4.2)) == (10 - 10, 4 - 6)
    assert b.origin_tile((-0.1, -0.1)) == (-1 - 10, -1 - 6)      # floor, not truncation


def test_the_crop_moves_in_whole_tiles_and_only_at_a_cell_boundary():
    b = _builder()
    assert b.origin_tile((10.0, 4.0)) == b.origin_tile((10.99, 4.99))
    assert b.origin_tile((11.0, 4.0)) != b.origin_tile((10.99, 4.0))


def test_terrain_lands_in_the_cell_its_world_position_says():
    occ = _occupancy()
    _put(occ, 12, 7, Tile.WATER)
    b = _builder(occ)
    grid = b.build((10.5, 6.5))
    water = grid[b.spec.channels.index("is_water")]
    blocks = grid[b.spec.channels.index("blocks_unit")]
    # hero tile (10, 6) -> origin (0, 0); the water tile is at crop (row 7, col 12)
    assert water[7, 12] == 1 and water.sum() == 1
    assert blocks[7, 12] == 1, "water blocks units in the sim's own table"


def test_the_lookup_table_is_the_simulators_and_not_a_restatement():
    """A fence blocks movement but not shots, and only WALL stops a projectile. If
    `brawl_sim.constants` ever changes its mind, this should follow it, not contradict it."""
    occ = _occupancy()
    _put(occ, 10, 6, Tile.FENCE)
    _put(occ, 11, 6, Tile.WALL)
    _put(occ, 12, 6, Tile.BUSH)
    b = _builder(occ)
    grid = b.build((10.5, 6.5))
    unit = grid[b.spec.channels.index("blocks_unit")]
    proj = grid[b.spec.channels.index("blocks_projectile")]
    bush = grid[b.spec.channels.index("is_bush")]
    assert (unit[6, 10], proj[6, 10]) == (1, 0)          # fence
    assert (unit[6, 11], proj[6, 11]) == (1, 1)          # wall
    assert (unit[6, 12], proj[6, 12], bush[6, 12]) == (0, 0, 1)


# ---------------------------------------------------------------------------
# UNKNOWN
# ---------------------------------------------------------------------------

def test_an_unobserved_cell_reads_as_floor():
    """Measured over two clips, 1.5-8.8% of the crop is UNKNOWN at any moment and every one of
    those cells is BELOW the hero -- the joystick and the attack buttons. Filling them with WALL
    would put a phantom wall in the bottom-left of the observation that follows the hero around;
    see grid.py's docstring for the per-cell numbers."""
    b = _builder()                                        # nothing ever observed
    grid = b.build((10.5, 6.5))
    for ch in ("blocks_unit", "blocks_projectile", "is_bush", "is_water"):
        assert grid[b.spec.channels.index(ch)].sum() == 0, ch


def test_the_unknown_fill_is_a_constructor_argument_so_the_choice_can_be_taken_back():
    b = _builder(unknown_tile=Tile.WALL)
    grid = b.build((10.5, 6.5))
    unit = grid[b.spec.channels.index("blocks_unit")]
    proj = grid[b.spec.channels.index("blocks_projectile")]
    assert unit.sum() == unit.size and proj.sum() == proj.size


def test_falling_off_the_occupancy_grid_reads_unknown_rather_than_wrapping():
    """64 tiles from where tracking started is further than any Showdown map allows, but a
    negative index would wrap silently instead of erroring."""
    occ = _occupancy()
    _put(occ, 0, 0, Tile.WALL)
    b = _builder(occ)
    grid = b.build((-70.5, -70.5))
    assert grid[b.spec.channels.index("blocks_unit")].sum() == 0


def test_an_unknown_cell_is_distinguishable_from_a_floor_one_only_by_the_fill():
    """Documents the cost of the decision plainly: after the fill, nothing downstream can tell the
    two apart. That is why the fill is chosen by measurement rather than by taste."""
    occ = _occupancy()
    for x in range(21):
        _put(occ, x, 6, Tile.FLOOR)
    b = _builder(occ)
    grid = b.build((10.5, 6.5))
    assert np.array_equal(grid[b.spec.channels.index("blocks_unit")][6],
                          grid[b.spec.channels.index("blocks_unit")][0])
    assert (occ.best() == UNKNOWN).any()


# ---------------------------------------------------------------------------
# the dynamic planes
# ---------------------------------------------------------------------------

def test_the_hero_channel_is_the_centre_cell_and_nothing_else():
    b = _builder()
    grid = b.build((10.9, 6.1))
    hero = grid[b.spec.channels.index("hero")]
    assert hero[6, 10] == 1 and hero.sum() == 1


def test_a_dead_hero_empties_its_own_channel():
    b = _builder()
    hero = b.build((10.5, 6.5), alive=False)[b.spec.channels.index("hero")]
    assert hero.sum() == 0


def test_only_a_track_seen_this_tick_reaches_the_grid():
    """Under `fair: true` an enemy the hero cannot see appears in NO channel -- `enemy_any` and
    `enemy_hidden` are refused at spec load. A coasted track is a prediction, not a sighting."""
    b = _builder()
    grid = b.build((10.5, 6.5), enemies=[_track(12.5, 6.5, seen_now=True, slot=0),
                                         _track(8.5, 6.5, seen_now=False, slot=1)])
    plane = grid[b.spec.channels.index("enemy_revealed")]
    assert plane[6, 12] == 1
    assert plane[6, 8] == 0 and plane.sum() == 1


def test_an_empty_slot_is_skipped_rather_than_crashing():
    """`TrackerResult.enemies` is slot-ordered with `None` holes; it is passed straight through."""
    b = _builder()
    grid = b.build((10.5, 6.5), enemies=[None, _track(12.5, 6.5), None])
    assert grid[b.spec.channels.index("enemy_revealed")].sum() == 1


def test_two_enemies_in_one_cell_count_two():
    """`_scatter_count` accumulates; these planes are occupancy counts, not booleans."""
    b = _builder()
    grid = b.build((10.5, 6.5), enemies=[_track(12.2, 6.4, slot=0), _track(12.8, 6.9, slot=1)])
    assert grid[b.spec.channels.index("enemy_revealed")][6, 12] == 2


def test_an_entity_outside_the_crop_is_dropped_not_clamped_to_the_edge():
    b = _builder()
    grid = b.build((10.5, 6.5), enemies=[_track(40.5, 6.5)],
                   projectiles=[Projectile(id=0, pos=(10.5, -40.5))])
    assert grid[b.spec.channels.index("enemy_revealed")].sum() == 0
    assert grid[b.spec.channels.index("projectile")].sum() == 0


def test_projectiles_land_in_their_own_plane():
    b = _builder()
    grid = b.build((10.5, 6.5), projectiles=[Projectile(id=0, pos=(9.5, 3.5))])
    plane = grid[b.spec.channels.index("projectile")]
    assert plane[3, 9] == 1 and plane.sum() == 1
    assert grid[b.spec.channels.index("enemy_revealed")].sum() == 0


def test_the_output_is_uint8_at_the_declared_shape_and_a_reused_buffer_is_cleared():
    b = _builder()
    buf = np.full(b.spec.shape, 9, np.uint8)
    grid = b.build((10.5, 6.5), enemies=[_track(12.5, 6.5)], out=buf)
    assert grid is buf and grid.dtype == np.uint8 and grid.shape == b.spec.shape
    again = b.build((10.5, 6.5), out=buf)
    assert again[b.spec.channels.index("enemy_revealed")].sum() == 0


# ---------------------------------------------------------------------------
# the gas map
# ---------------------------------------------------------------------------

@dataclass
class _Odo:
    position_tiles: tuple
    segment: int = 0
    status: str = "ok"


@dataclass
class _Plan:
    size_tiles: tuple
    origin_tile: tuple = (0, 0)


@dataclass
class _Zone:
    cells: np.ndarray
    observed: np.ndarray

    @classmethod
    def of(cls, rows, cols, gassed=()):
        c = np.zeros((rows, cols), bool)
        for r, col in gassed:
            c[r, col] = True
        return cls(cells=c, observed=np.ones((rows, cols), bool))


def _gas_tick(gas, gassed=(), pos=(0.0, 0.0), segment=0, status="ok", rows=6, cols=6):
    return gas.update(_Zone.of(rows, cols, gassed), _Plan(size_tiles=(cols, rows)),
                      _Odo(position_tiles=pos, segment=segment, status=status))


def test_gas_lands_at_the_world_position_odometry_reports():
    gas = GasMap(32, 32)
    _gas_tick(gas, gassed=[(2, 3)], pos=(5.0, 4.0))
    ox, oy = gas.origin
    assert gas.gassed[4 + 2 - oy, 5 + 3 - ox]
    assert gas.gassed.sum() == 1


def test_gas_is_sticky_because_the_poison_only_ever_grows():
    """A cell that goes under the joystick after being gassed must not read back as clear."""
    gas = GasMap(32, 32)
    assert _gas_tick(gas, gassed=[(2, 3)]) == 1
    assert _gas_tick(gas, gassed=[]) == 0                 # this frame says nothing
    ox, oy = gas.origin
    assert gas.gassed[2 - oy, 3 - ox]


def test_only_newly_gassed_cells_are_counted():
    gas = GasMap(32, 32)
    assert _gas_tick(gas, gassed=[(2, 3)]) == 1
    assert _gas_tick(gas, gassed=[(2, 3), (2, 4)]) == 1


def test_an_untrusted_pose_deposits_nothing():
    """`occupancy.update`'s rule, and for its reason: `uncertain` has a world frame that is not
    evidence, `lost` has none at all."""
    for status in ("uncertain", "lost"):
        gas = GasMap(32, 32)
        assert _gas_tick(gas, gassed=[(2, 3)], status=status) == 0
        assert not gas.gassed.any()


def test_a_new_segment_drops_the_map_rather_than_overlaying_two_world_frames():
    gas = GasMap(32, 32)
    _gas_tick(gas, gassed=[(2, 3)])
    assert _gas_tick(gas, gassed=[(4, 4)], segment=1) == 0
    assert not gas.gassed.any()


def test_the_first_tick_adopts_the_segment_instead_of_resetting_on_it():
    """Seeding `_segment` to a sentinel would silently throw away the opening frame of every match
    that did not happen to start at segment 0."""
    gas = GasMap(32, 32)
    assert _gas_tick(gas, gassed=[(2, 3)], segment=7) == 1
    assert gas.gassed.sum() == 1


def test_a_deposit_falling_off_the_grid_is_clipped_not_wrapped():
    gas = GasMap(32, 32)
    assert _gas_tick(gas, gassed=[(2, 3)], pos=(500.0, 500.0)) == 0


def test_the_builders_gas_map_matches_its_occupancy_maps_frame():
    """One pair of crop indices serves both, so a mismatched origin would offset every gassed
    cell against the terrain under it."""
    occ = _occupancy(64, 96)
    b = _builder(occ)
    assert (b.gas.height, b.gas.width) == (occ.height, occ.width)
    assert b.gas.origin == occ.origin


def test_gas_reaches_the_in_zone_plane_at_the_right_cell():
    b = _builder()
    ox, oy = b.gas.origin
    b.gas.gassed[6 - oy, 12 - ox] = True
    plane = b.build((10.5, 6.5))[b.spec.channels.index("in_zone")]
    assert plane[6, 12] == 1 and plane.sum() == 1
