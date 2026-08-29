import torch

from brawl_sim.constants import (
    BOT_KINDS,
    CHAR_TO_TILE,
    N_KINDS,
    N_PROJ_CLASSES,
    N_PROJ_KINDS,
    N_TILES,
    TILE_BLOCKS_PROJ,
    TILE_BLOCKS_UNIT,
    TILE_IS_BOX_SPAWN,
    TILE_IS_BUSH,
    TILE_IS_SPAWN,
    TILE_IS_WATER,
    TILE_TO_CHAR,
    DeathCause,
    Kind,
    Proj,
    ProjClass,
    Tile,
)

# Ground truth from BRAWL_SIM_BUILD_PLAN.md Step 2 / Notice 4's pass/block table.
EXPECTED_BLOCKS_UNIT = {
    Tile.FLOOR: False, Tile.WALL: True, Tile.BUSH: False, Tile.WATER: True,
    Tile.FENCE: True, Tile.SPAWN: False, Tile.BOX: False,
}
EXPECTED_BLOCKS_PROJ = {
    Tile.FLOOR: False, Tile.WALL: True, Tile.BUSH: False, Tile.WATER: False,
    Tile.FENCE: False, Tile.SPAWN: False, Tile.BOX: False,
}


def test_enum_values():
    assert (Tile.FLOOR, Tile.WALL, Tile.BUSH, Tile.WATER) == (0, 1, 2, 3)
    assert (Tile.FENCE, Tile.SPAWN, Tile.BOX) == (4, 5, 6)
    assert (Kind.HERO_MORTIS, Kind.BOT_SNIPER, Kind.BOT_ARTILLERY) == (0, 1, 2)
    assert (Kind.BOT_MELEE, Kind.BOT_RIFLE) == (3, 4)
    assert (Proj.NONE, Proj.SNIPER_BOLT, Proj.ARTILLERY_SHELL, Proj.RIFLE_ARROW) == (0, 1, 2, 3)
    assert Proj.SUPER_BOLT == 4
    assert (ProjClass.PROJECTILE, ProjClass.ARTILLERY, ProjClass.HAZARD) == (0, 1, 2)
    assert (DeathCause.ALIVE, DeathCause.COMBAT, DeathCause.ZONE) == (0, 1, 2)
    assert (N_TILES, N_KINDS, N_PROJ_KINDS, N_PROJ_CLASSES) == (7, 8, 7, 3)
    assert BOT_KINDS == (1, 2, 3, 4, 5, 6, 7)


def test_enum_widths_match_their_enums():
    """The N_* constants size observation one-hots (`projectiles.kind_onehot` is N_PROJ_KINDS
    wide), so a member added without bumping its count silently truncates the observation rather
    than raising. Derive the check from the enums themselves."""
    assert N_PROJ_KINDS == len(Proj)
    assert N_PROJ_CLASSES == len(ProjClass)
    assert N_KINDS == len(Kind)
    assert N_TILES == len(Tile)


def test_proj_class_default_is_the_ordinary_case():
    """core/state.zero_ blanket-zeroes every resettable field on reset, so whatever a fresh
    `prj_class` slot decodes to is whatever 0 means. That must be the ordinary bullet, not an
    exotic class -- the same reasoning constants.Person gives for RUSH being 0."""
    assert int(ProjClass.PROJECTILE) == 0


def test_blocks_unit_matches_table():
    for tile, expected in EXPECTED_BLOCKS_UNIT.items():
        assert bool(TILE_BLOCKS_UNIT[tile]) == expected, tile


def test_blocks_proj_matches_table():
    for tile, expected in EXPECTED_BLOCKS_PROJ.items():
        assert bool(TILE_BLOCKS_PROJ[tile]) == expected, tile


def test_only_wall_blocks_proj():
    # Notice 4: fence no longer blocks shots, so WALL is the only opaque tile left --
    # this is also the tensor every physical-LOS check (melee, sniper/rifle fire-gate)
    # reuses, since there's no separate vision table anymore.
    assert int(TILE_BLOCKS_PROJ.sum()) == 1
    assert bool(TILE_BLOCKS_PROJ[Tile.WALL]) is True


def test_is_bush_water_spawn_boxspawn_are_singletons():
    for table, tile in (
        (TILE_IS_BUSH, Tile.BUSH),
        (TILE_IS_WATER, Tile.WATER),
        (TILE_IS_SPAWN, Tile.SPAWN),
        (TILE_IS_BOX_SPAWN, Tile.BOX),
    ):
        assert bool(table[tile]) is True
        assert int(table.sum()) == 1


def test_lookup_tensor_shape_and_dtype():
    for table in (
        TILE_BLOCKS_UNIT, TILE_BLOCKS_PROJ,
        TILE_IS_BUSH, TILE_IS_WATER, TILE_IS_SPAWN, TILE_IS_BOX_SPAWN,
    ):
        assert table.shape == (N_TILES,)
        assert table.dtype == torch.bool


def test_char_tile_roundtrip():
    assert len(CHAR_TO_TILE) == N_TILES
    for char, tile in CHAR_TO_TILE.items():
        assert TILE_TO_CHAR[tile] == char
    # every tile used by the Step 5 map CSV legend is covered
    for tile in (Tile.FLOOR, Tile.WALL, Tile.BUSH, Tile.WATER, Tile.FENCE, Tile.SPAWN, Tile.BOX):
        assert tile in TILE_TO_CHAR
