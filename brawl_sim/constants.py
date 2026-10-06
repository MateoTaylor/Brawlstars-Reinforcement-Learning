"""Fixed enums and tile lookup tables shared by every module."""
from enum import IntEnum

import torch


class Tile(IntEnum):
    FLOOR = 0
    WALL = 1
    BUSH = 2
    WATER = 3
    # Not a sim tile: fences are rare enough in Solo Showdown that no map carries one, and the
    # map CSV vocabulary (MAP_CHAR_TO_TILE below) refuses the character. The member stays because
    # brawl_vision.terrain.labeling.CLASSES, its label files and the trained terrain classifier
    # all index it.
    FENCE = 4
    SPAWN = 5
    BOX = 6


class Kind(IntEnum):
    """The playable archetypes. Members keep their integer values forever -- the value indexes
    every per-kind SimParams tensor's K axis, every `kind_onehot`, and the order of
    `EnvConfig.enemy_type_weights` (config.ARCHETYPE_SHORT_NAMES).

    The first five are named for the ROLE, not the brawler; each role is grounded in one real
    brawler, who can change without renaming the member:

        HERO_MORTIS    Mortis        dash assassin
        BOT_SNIPER     Brock         long-range single-shot rocket
        BOT_ARTILLERY  Grom          lobbed shell that splits
        BOT_MELEE      Buzz          point-blank swept hitscan
        BOT_RIFLE      Shelly        mid-range shotgun spread

    BOT_EDGAR, BOT_SPIKE and BOT_BULL are named for the BRAWLER instead (user's rule): a role label
    would invent a distinction the sim does not make (Edgar vs BOT_MELEE, Spike vs BOT_ARTILLERY,
    Bull vs BOT_RIFLE). The name is load-bearing: `render/viewer._ENTITY_SPRITE_PATHS` derives
    each sprite path from it (`assets/entities/bot_edgar.png`).

        BOT_EDGAR      Edgar         two-hit forward combo, heals off the damage it deals
        BOT_SPIKE      Spike         slow shell that splits into a 6-arm star on landing
        BOT_BULL       Bull          short-range 5-pellet shotgun on a 10000 HP frame
    """
    HERO_MORTIS = 0
    BOT_SNIPER = 1
    BOT_ARTILLERY = 2
    BOT_MELEE = 3
    BOT_RIFLE = 4
    BOT_EDGAR = 5
    BOT_SPIKE = 6
    BOT_BULL = 7


class Person(IntEnum):
    """A bot's PERSONALITY: what it does with its movement, independent of its `Kind` (which
    decides what its weapon does). Kind x Person are orthogonal -- a sniper can be a Rush and a
    melee bot can be a Camper. See bots/personality.py for each one's behavior.

    RUSH is deliberately 0. `core/state.zero_` blanket-zeroes every resettable field on reset,
    so an entity whose personality was never assigned (a bug in core/spawn.sample_personalities,
    a hand-built test state) comes out RUSH, the most aggressive option. Failing toward pressure
    is obvious in a rollout; failing toward CAMPER would silently hand the agent a lobby of
    passive bots and reward bush-hiding.
    """
    RUSH = 0
    CAMPER = 1
    HUNTER = 2
    TRAPPER = 3
    KITE = 4


class Proj(IntEnum):
    """WHICH WEAPON fired a projectile -- its identity, for rendering and for the observation's
    `kind_onehot`. Orthogonal to `ProjClass` below, which says how it MOVES and DAMAGES.

    A projectile keeps its firing weapon's `Proj` even when it changes class mid-life: an
    artillery shell's split shards stay `ARTILLERY_SHELL` (core/projectiles._spawn_splits: "they
    are the same weapon"), and Brock's lingering sphere stays `SNIPER_BOLT` (class HAZARD). That
    is why there is no `HAZARD_AREA` member here -- a hazard is a class, not a weapon.
    """
    NONE = 0
    SNIPER_BOLT = 1
    ARTILLERY_SHELL = 2
    RIFLE_ARROW = 3
    # Mortis's Super bolt (core/projectiles.spawn_supers).
    SUPER_BOLT = 4
    # Spike's cactus. Its own member rather than ARTILLERY_SHELL, though both are splitting
    # ARTILLERY shells, because `Proj` is what the observation exposes and the two are different
    # THREATS: Grom's shell breaks into a short 4-arm cross, Spike's into a longer, faster 6-arm
    # star, and an agent that cannot tell them apart cannot learn where it is safe to stand.
    SPIKE_SHELL = 5
    # Bull's shotgun. Mechanically Shelly's RIFLE_ARROW, but its own member because `kind_onehot`
    # is the ONLY categorical weapon signal the agent gets (configs/agent_obs.yaml selects
    # `projectiles.kind_onehot`, not `projectiles.owner_kind`): folded in, Bull's heavier pellets
    # would be told apart only through the continuous `damage` field.
    BULL_SLUG = 6
    # Mortis's gadget: a spinner that flies up to `gadget_range` toward the nearest enemy its
    # thrower can see (along the facing if none), lands after `gadget_flight_seconds`, and deals
    # `gadget_damage` in `gadget_radius`. ARTILLERY-class in PROJ_CLASS_OF (it hurts nothing in
    # flight and resolves where it lands) but its own `Proj` because it is its own THREAT: a
    # heavy burst that lands almost at once is not a Grom shell.
    GADGET_SPINNER = 7


class ProjClass(IntEnum):
    """HOW a projectile moves and damages, independent of which weapon fired it (`Proj` above).
    Stored per projectile slot in `state.prj_class`.

        PROJECTILE  moves, damages in flight, dies on the first unit/wall/box hit
        ARTILLERY   moves, deals NO damage in flight, resolves at its landing point (AoE + splits)
        HAZARD      does not move, does not disappear on contact, damages whatever stands in it
                    (Brock's lingering sphere)

    PROJECTILE is deliberately 0 so that `core/state.zero_`'s blanket reset leaves a slot in the
    most ordinary class rather than an exotic one.
    """
    PROJECTILE = 0
    ARTILLERY = 1
    HAZARD = 2


class AimModel(IntEnum):
    """HOW a kind converts "there is my target" into an aim direction and an aim point. One of the
    five per-kind fire-rule fields bots/combat_rules.py reads, and the only one that is not a
    plain number, because the three models differ in their arithmetic, not in a threshold.

        DIRECT  straight at the target's CURRENT position: no lead, no noise. For a weapon with
                nothing in flight to lead (Buzz and Edgar, `proj_kind: NONE`) -- the aim outputs
                exist only for shape consistency, since combat.melee_hitscan's cone follows
                `ent_facing` rather than anything this produces.
        LEAD    geo.lead_target's intercept, blended by `lead_target_fraction`, then per-entity
                Gaussian ANGULAR noise ~ N(0, aim_noise_std_rad). The ordinary straight-shot
                model: Brock, Shelly, Bull.
        LOB     the leaded LANDING POINT, then per-entity Gaussian POSITIONAL noise in tile-space
                (aim_noise_tiles). Grom, Spike. A different noise model on purpose: the payload is
                an area landing on a point, not a line through one, so scattering the point is
                what a miss physically means. When `proj_flight_seconds > 0` the intercept is
                closed form (the shell lands in that many seconds whatever the distance) and the
                fixed-point solve is skipped.

    DIRECT is deliberately 0, so a kind that never mentions `aim_model` resolves to the model with
    NO lead -- an aim that misses behind every moving target -- rather than silently handing a new
    brawler a perfect intercept.
    """
    DIRECT = 0
    LEAD = 1
    LOB = 2


class DeathCause(IntEnum):
    ALIVE = 0
    COMBAT = 1
    ZONE = 2


# Which ProjClass a weapon's projectiles spawn as, indexed by `Proj`. A property of the WEAPON,
# and deliberately NOT a brawlers.yaml field: "Grom's shell arcs" is part of what an artillery
# shell IS, and a config able to declare a SNIPER_BOLT as ARTILLERY could only ever be a mistake.
# A table, not an inline `proj_kind == ARTILLERY_SHELL` test, so a new lobbing weapon is a row:
# one the test missed would silently spawn as a PROJECTILE and never reach _spawn_splits.
PROJ_CLASS_OF = (
    ProjClass.PROJECTILE,   # NONE -- no projectile is spawned with this kind; the row must exist
    ProjClass.PROJECTILE,   # SNIPER_BOLT
    ProjClass.ARTILLERY,    # ARTILLERY_SHELL
    ProjClass.PROJECTILE,   # RIFLE_ARROW
    ProjClass.PROJECTILE,   # SUPER_BOLT   (spawn_supers writes PROJECTILE directly; kept in sync)
    ProjClass.ARTILLERY,    # SPIKE_SHELL
    ProjClass.PROJECTILE,   # BULL_SLUG
    ProjClass.ARTILLERY,    # GADGET_SPINNER (no damage in flight; a burst where it lands)
)

N_TILES = 7
N_KINDS = 8
N_PROJ_KINDS = 8
N_PROJ_CLASSES = 3

N_PERSONS = 5
# Personalities that reliably pressure the hero: RUSH closes on anything it can see, HUNTER
# sweeps bushes for what it can't. core/spawn.sample_personalities guarantees at least
# cfg.bots_min_aggressive of these per env, so a random draw cannot hand the agent a lobby it
# wins by standing still.
AGGRESSIVE_PERSONS = (int(Person.RUSH), int(Person.HUNTER))

# A hunter's search history is a BITMASK over bush waypoints (state.ent_hunt_seen), so the
# waypoint count is capped by int64's usable bits; maps/loader.MAX_BUSH_WAYPOINTS enforces it.
HUNT_WAYPOINT_BITS = 63


def _bool_table(true_tiles: tuple[Tile, ...]) -> torch.Tensor:
    table = torch.zeros(N_TILES, dtype=torch.bool)
    for tile in true_tiles:
        table[tile] = True
    return table


# FLOOR / SPAWN / BOX pass units and projectiles. WALL blocks both. BUSH passes both
# (occupant hiding is a perception-layer concern in bots/perception.py, not a tile block).
# WATER and FENCE block units only; both pass projectiles. (FENCE appears in no map, but
# brawl_vision's scorer and brawl_deployment's class table read its row.)
#
# There is no tile-level vision table: the camera is a fixed bird's-eye view, so no tile blocks
# sight, and bush-hiding (bots/perception.py) is the only concealment. Physical line-of-sight
# checks (melee hit validation, `fire_needs_los` fire-gating) reuse TILE_BLOCKS_PROJ via
# core/terrain.line_of_sight(): with only WALL opaque, a shot and a sightline block alike.
TILE_BLOCKS_UNIT = _bool_table((Tile.WALL, Tile.WATER, Tile.FENCE))
TILE_BLOCKS_PROJ = _bool_table((Tile.WALL,))
TILE_IS_BUSH = _bool_table((Tile.BUSH,))
TILE_IS_WATER = _bool_table((Tile.WATER,))

# The complete tile alphabet, one character per Tile. This is what brawl_vision's label files
# and terrain classes are written in (they carry `f`), so it keeps every member.
CHAR_TO_TILE = {
    ".": Tile.FLOOR,
    "#": Tile.WALL,
    "b": Tile.BUSH,
    "~": Tile.WATER,
    "f": Tile.FENCE,
    "S": Tile.SPAWN,
    "X": Tile.BOX,
}
TILE_TO_CHAR = {tile: char for char, tile in CHAR_TO_TILE.items()}

# The map CSV vocabulary: what maps/loader.load_map_csv accepts. The alphabet minus FENCE --
# see the Tile.FENCE comment. A CSV carrying `f` fails to load rather than silently becoming
# a wall or a floor.
MAP_CHAR_TO_TILE = {char: tile for char, tile in CHAR_TO_TILE.items() if tile is not Tile.FENCE}
