"""Fixed enums and tile lookup tables shared by every module. See
BRAWL_SIM_BUILD_PLAN.md Step 2 / Notice 4 for the source-of-truth pass/block table.
"""
from enum import IntEnum

import torch


class Tile(IntEnum):
    FLOOR = 0
    WALL = 1
    BUSH = 2
    WATER = 3
    FENCE = 4
    SPAWN = 5
    BOX = 6


class Kind(IntEnum):
    """The playable archetypes. Members keep their integer values forever -- the value indexes
    every per-kind SimParams tensor's K axis, every `kind_onehot`, and `configs/default.yaml`'s
    `enemy_type_weights` ordering.

    The first five are named for the ROLE, not the brawler, and each role is grounded in one real
    brawler whose identity CAN change without renaming the member (see bot_overhaul.md Step B1,
    which re-grounded four of the five):

        HERO_MORTIS    Mortis        dash assassin
        BOT_SNIPER     Brock         long-range single-shot rocket   (was Nani)
        BOT_ARTILLERY  Grom          lobbed shell that splits        (was Dynamike)
        BOT_MELEE      Buzz          point-blank swept hitscan       (was El Primo)
        BOT_RIFLE      Shelly        mid-range shotgun spread        (was Bo)

    BOT_EDGAR and BOT_SPIKE break that naming rule deliberately, at the repo owner's direction:
    they are named for the BRAWLER, not a role. Neither has a role label that would distinguish it
    from an existing member without inventing a distinction the sim does not make (Edgar vs
    BOT_MELEE: both are point-blank multi-hit hitscan; Spike vs BOT_ARTILLERY: both are a shell
    that splits on landing), and the render layer derives the sprite path from this member's name
    (`render/viewer._ENTITY_SPRITE_PATHS` -> `assets/entities/bot_edgar.png`), so the name is
    load-bearing rather than cosmetic.

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
    so an entity whose personality was never explicitly assigned (a bug in
    core/spawn.sample_personalities, a hand-built test state) comes out RUSH -- the most
    aggressive option. Failing toward pressure is recoverable and obvious in a rollout; failing
    toward CAMPER would silently hand the agent a lobby of passive bots and reward exactly the
    bush-hiding degenerate policy this whole system exists to stop.
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
    are the same weapon"), and Brock's lingering sphere will stay `SNIPER_BOLT` when Step C3 lands.
    That is why there is no `HAZARD_AREA` member here -- a hazard is a class, not a weapon.
    """
    NONE = 0
    SNIPER_BOLT = 1
    ARTILLERY_SHELL = 2
    RIFLE_ARROW = 3
    # Declared in Step B1 but not spawned until Step D2 (Mortis's Super). Widening the enum once,
    # up front, keeps the observation's kind_onehot width churn to a single step -- D11 makes the
    # resulting checkpoint break a non-issue.
    SUPER_BOLT = 4
    # Spike's cactus. A SEPARATE member rather than reusing ARTILLERY_SHELL even though both are
    # ProjClass.ARTILLERY shells that split, because `Proj` is what the observation exposes and the
    # two are different THREATS: Grom's shell breaks into a 4-arm cross reaching ~3.2 tiles, Spike's
    # into a 6-arm star reaching ~5.4, at nearly triple the shard speed. An agent that cannot tell
    # them apart cannot learn where it is safe to stand when one lands.
    SPIKE_SHELL = 5
    # Bull's shotgun. Shares every mechanic with Shelly's RIFLE_ARROW -- a constant-speed fan of
    # PROJECTILE-class pellets -- and is still its own member, because `kind_onehot` is the ONLY
    # categorical weapon signal the agent gets: configs/agent_obs.yaml selects
    # `projectiles.kind_onehot` but NOT `projectiles.owner_kind`. Folding Bull into RIFLE_ARROW
    # would leave the policy unable to tell an 880-per-pellet volley from a 600-per-pellet one
    # except by regressing on the continuous `damage` field.
    BULL_SLUG = 6


class ProjClass(IntEnum):
    """HOW a projectile moves and damages, independent of which weapon fired it (`Proj` above).

    Replaces the `prj_lobbed` bool, which was already a two-value version of exactly this:
    `lobbed=False` is PROJECTILE, `lobbed=True` is ARTILLERY. HAZARD is the third case Brock's
    lingering sphere needs.

        PROJECTILE  moves, damages in flight, dies on the first unit/wall/box hit
        ARTILLERY   moves, deals NO damage in flight, resolves at its landing point (AoE + splits)
        HAZARD      does not move, does not disappear on contact, damages whatever stands in it

    PROJECTILE is deliberately 0 so that `core/state.zero_`'s blanket reset leaves a slot in the
    most ordinary class rather than an exotic one.

    Declared in Step B1; `prj_class` is not wired up until Step C3. See bot_overhaul.md D8.
    """
    PROJECTILE = 0
    ARTILLERY = 1
    HAZARD = 2


class AimModel(IntEnum):
    """HOW a kind converts "there is my target" into an aim direction and an aim point. One of the
    five data fields that replaced the four hand-written archetype modules in Step E1 (see
    bots/combat_rules.py) -- it is the only one of the five that is not a plain number, because
    the three models differ in their arithmetic, not in a threshold.

        DIRECT  straight at the target's CURRENT position: no lead, no noise. For a weapon with
                nothing in flight to lead (Buzz's hitscan cone, `proj_kind: NONE`) -- the aim
                outputs exist only for shape consistency, since combat.melee_hitscan reads
                `ent_facing` rather than anything this produces.
        LEAD    geo.lead_target's intercept, blended by `lead_target_fraction`, then per-entity
                Gaussian ANGULAR noise ~ N(0, aim_noise_std_rad). The ordinary straight-shot
                model: Brock, Shelly.
        LOB     the leaded LANDING POINT, then per-entity Gaussian POSITIONAL noise in tile-space
                (aim_noise_tiles). Grom. A different noise model on purpose: the payload is an
                area landing on a point, not a line through one, so scattering the point is what
                a miss physically means. When `proj_flight_seconds > 0` the intercept is closed
                form (the shell lands in that many seconds whatever the distance) and the
                fixed-point solve is skipped.

    DIRECT is deliberately 0, so a kind that never mentions `aim_model` resolves to the model with
    NO lead -- an aim that misses behind every moving target. The alternative default (LEAD) would
    silently hand a new brawler a perfect intercept solution the day someone forgets the field.
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
#
# It was `proj_kind == ARTILLERY_SHELL`, written inline in core/projectiles.spawn_volley, until a
# SECOND lobbing weapon existed. That form was correct exactly while there was one, and its failure
# mode when Spike arrived was silent: SPIKE_SHELL simply is not ARTILLERY_SHELL, so his cactus
# would have spawned as an ordinary PROJECTILE -- damaging the first body it touched (his shell is
# supposed to deal nothing) and never reaching _spawn_splits at all, which only ever sees
# detonations. A table makes the third one a row rather than a bug.
PROJ_CLASS_OF = (
    ProjClass.PROJECTILE,   # NONE -- no projectile is spawned with this kind; the row must exist
    ProjClass.PROJECTILE,   # SNIPER_BOLT
    ProjClass.ARTILLERY,    # ARTILLERY_SHELL
    ProjClass.PROJECTILE,   # RIFLE_ARROW
    ProjClass.PROJECTILE,   # SUPER_BOLT   (spawn_supers writes PROJECTILE directly; kept in sync)
    ProjClass.ARTILLERY,    # SPIKE_SHELL
    ProjClass.PROJECTILE,   # BULL_SLUG
)

N_TILES = 7
N_KINDS = 8
N_PROJ_KINDS = 7
N_PROJ_CLASSES = 3
N_AIM_MODELS = 3
BOT_KINDS = (1, 2, 3, 4, 5, 6, 7)

N_PERSONS = 5
# Personalities that reliably generate pressure on the hero: RUSH closes on anything it can see,
# HUNTER sweeps bushes looking for what it can't. core/spawn.sample_personalities guarantees at
# least cfg.bots_min_aggressive of these per env, so a random draw can never produce a lobby the
# agent wins by standing still -- see that function's docstring.
AGGRESSIVE_PERSONS = (int(Person.RUSH), int(Person.HUNTER))
# Personalities whose movement wants a bush, and which therefore fall back to RUSH steering when
# no bush outside the zone is reachable (the user-specified rule).
BUSH_PERSONS = (int(Person.CAMPER), int(Person.HUNTER), int(Person.TRAPPER))

# A hunter's search history is a BITMASK over bush waypoints (state.ent_hunt_seen), so the
# waypoint count is capped by int64's usable bits. See maps/loader.MAX_BUSH_WAYPOINTS, which is
# where the cap is actually enforced when the bank is built.
HUNT_WAYPOINT_BITS = 63


def _bool_table(true_tiles: tuple[Tile, ...]) -> torch.Tensor:
    table = torch.zeros(N_TILES, dtype=torch.bool)
    for tile in true_tiles:
        table[tile] = True
    return table


# FLOOR / SPAWN / BOX pass units and projectiles. WALL blocks both. BUSH passes both
# (occupant hiding is a perception-layer concern in bots/perception.py, not a tile block).
# WATER and FENCE block units only; both pass projectiles.
#
# There is no tile-level vision table. The camera is fixed bird's-eye, so no tile ever
# blocks sight, not even WALL. Bush-hiding is the only thing that hides an entity, and
# that's computed directly in bots/perception.py, not here. Anywhere a genuine physical
# line-of-sight check is needed (melee hit validation, sniper/rifle fire-gating), it reuses
# TILE_BLOCKS_PROJ via core/terrain.line_of_sight() -- "blocks a shot" and "blocks a
# sightline" are the same question once only WALL is opaque.
TILE_BLOCKS_UNIT = _bool_table((Tile.WALL, Tile.WATER, Tile.FENCE))
TILE_BLOCKS_PROJ = _bool_table((Tile.WALL,))
TILE_IS_BUSH = _bool_table((Tile.BUSH,))
TILE_IS_WATER = _bool_table((Tile.WATER,))
TILE_IS_SPAWN = _bool_table((Tile.SPAWN,))
TILE_IS_BOX_SPAWN = _bool_table((Tile.BOX,))

# Matches the Step 5 map CSV legend exactly.
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
