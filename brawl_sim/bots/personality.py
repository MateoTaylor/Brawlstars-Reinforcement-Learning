"""Bot PERSONALITIES: what a bot does with its movement, drawn per entity per reset and
orthogonal to its archetype. See BRAWL_SIM_BUILD_PLAN.md Step 41.

**Why this exists.** Before Step 41, movement flavor was baked into each of the four archetype
policies, and all four had the same failure: with no visible target they converged. Melee steered
at the zone rect's center, artillery hid behind walls, sniper/rifle held range on a target they
didn't have. Nothing in the game ever went looking for a player it couldn't see. An agent that
sat in a bush at the edge of the map therefore beat a 9-bot lobby by declining to participate:
the bots gathered in the middle, ground each other down, and the agent walked out at the end.
This module replaces "movement follows from your weapon" with "movement follows from your
personality", and two of the five personalities exist specifically to punish that policy.

**The five personalities** (constants.Person; weights in configs/default.yaml's
`bots.personality_weights`):
  - RUSH    closes on anything it can see, and explores at random when it can't.
  - CAMPER  sits in a bush and fires only once something can actually see it. Does NOT leave, not
            even when idle, until the shrinking zone gets within `bots_camper_zone_flee_tiles`.
  - HUNTER  engages what it can see, and otherwise SWEEPS THE MAP for what it can't -- walking to
            the nearest BUSH WAYPOINT (maps/loader.bush_waypoints) it has not visited yet, tracked
            per entity as a bitmask in `ent_hunt_seen`. This is the direct counter to a hidden
            agent, and it is the one personality whose first implementation measurably did not
            work: see bush_waypoints' docstring for why "nearest unsearched bush TILE" produced a
            hunter that shuffled around its spawn instead of searching anything.
  - TRAPPER CAMPER's cover discipline with RUSH's trigger discipline: holds a bush and shoots
            anything in range from it, but drifts bush-to-bush (via the same waypoint machinery
            HUNTER uses) while nothing is visible, so its ambush spots don't go stale.
  - KITE    works the open map holding its archetype's own ideal range
            (bots/policy.RANGE_FRACTION_BY_KIND) from whatever it can see.

**Rules that hold for every personality**, per the same specification:
  - Nobody walks into the green zone. bots/policy.zone_avoid_contribution pushes inward from
    `bots_zone_avoid_tiles` out, before any damage is taken; zone_contribution drags anyone the
    zone has already swallowed back out. Bush search itself also refuses to target a bush whose
    own zone clearance is below `bots_camper_zone_flee_tiles`, so no bot ever walks toward cover
    that is about to become lethal.
  - A personality that wants a bush and cannot find a safe one behaves as RUSH instead
    (`~scan.found` below). This is not a rare fallback: two of the three default maps (`walled`,
    and tests' `blank`) contain no bush tiles at all, and `open` has 62 out of 3600.
  - With nothing visible, everyone reverts to randomized exploration rather than a shared
    destination, so bots spread out and can wander into an ambush the AGENT has set. CAMPER is
    the one deliberate exception (its stay-put rule is more specific and is what makes it a real
    threat that has to be cleared); TRAPPER relocates between bushes rather than into the open.

**Structure.** Behavior is resolved in two stages, which is what keeps this batched and cheap:
`_select_mode` maps (personality, world state) to one of seven `Mode`s, then a single
(N_MODES, N_TERMS) constant weight table is gathered by mode to produce every steering weight at
once. Adding a personality means adding rows to `_select_mode`, not a new steering pass -- the
expensive shared work (one local bush scan, one waypoint query, one wander advance) is paid once
for all (N,E) entities regardless of how many distinct personalities are in play.
"""
import math
from enum import IntEnum

import torch

from ..constants import Person
from ..core import geometry as geo
from ..core import terrain
from . import perception
from . import policy as shared
from . import steering

_EPS = 1e-6
_WANDER_LOOKAHEAD_TILES = 2.5

# Public: tests assert against these, and they are the two numbers most worth tuning by hand.
# RANGE_DEADBAND replaces the per-archetype deadbands Steps 17/18/20 each carried separately
# (sniper 1.5, artillery 2.0, rifle 2.0) -- after the split, "how sloppy is range-keeping" is a
# property of the KITE behavior, not of the weapon, and only the ideal DISTANCE stays per-archetype
# (bots/policy.RANGE_FRACTION_BY_KIND).
RANGE_DEADBAND = 1.5
RETREAT_HP_FRACTION = 0.35


class Mode(IntEnum):
    """The behavior an entity is actually executing this tick. A personality is a fixed
    disposition; a Mode is what that disposition resolves to given what the entity can currently
    see -- several personalities share Modes (RUSH and HUNTER both CLOSE on a visible enemy;
    HUNTER and TRAPPER both HUNT_BUSH when idle), which is exactly why the steering weights are
    keyed off Mode rather than off Person. Surfaced on BotIntent.mode for rendering/diagnostics
    only; nothing in the simulation reads it back."""
    WANDER = 0      # explore at random; nothing worth reacting to
    CLOSE = 1       # take ground toward the enemy
    HOLD_RANGE = 2  # sit at the archetype's ideal distance and orbit
    HOLD_STILL = 3  # do not move at all (in cover)
    TO_BUSH = 4     # walk to the nearest safe bush
    HUNT_BUSH = 5   # walk to the nearest safe bush not already searched
    RETREAT = 6     # break contact


N_MODES = len(Mode)

# Steering terms, in the column order of _MODE_WEIGHTS below.
_TERMS = ("seek", "range", "strafe", "flee", "bush", "hunt", "wander")
N_TERMS = len(_TERMS)

# (N_MODES, N_TERMS). The small `wander` component in TO_BUSH/HUNT_BUSH is deliberate: it
# decorrelates two bots that picked the same bush so they take different paths to it, and it is
# the "with some variance" the specification asks for. HOLD_STILL is all-zero on purpose --
# steering.combine's normalize maps an all-zero sum to the zero vector (not NaN), and
# movement.apply_movement reads a zero move_dir as "stand here", leaving facing untouched.
_MODE_WEIGHTS = {
    Mode.WANDER:     (0.0, 0.0, 0.00, 0.0, 0.0, 0.0, 1.00),
    Mode.CLOSE:      (1.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.00),
    Mode.HOLD_RANGE: (0.0, 1.0, 0.50, 0.0, 0.0, 0.0, 0.00),
    Mode.HOLD_STILL: (0.0, 0.0, 0.00, 0.0, 0.0, 0.0, 0.00),
    Mode.TO_BUSH:    (0.0, 0.0, 0.00, 0.0, 1.0, 0.0, 0.15),
    Mode.HUNT_BUSH:  (0.0, 0.0, 0.00, 0.0, 0.0, 1.0, 0.15),
    Mode.RETREAT:    (0.0, 0.0, 0.20, 1.0, 0.0, 0.0, 0.00),
}

_WEIGHT_TABLE_CACHE: dict = {}


def _weight_table(device) -> torch.Tensor:
    """(N_MODES, N_TERMS) f32, built once per device. Gathering this by mode gives every steering
    weight for every entity in ONE indexing op, instead of a torch.where chain per (mode, term)
    pair -- 7 kernels rather than ~50."""
    cached = _WEIGHT_TABLE_CACHE.get(device)
    if cached is None:
        cached = torch.stack([
            torch.stack([
                torch.full((), w, device=device, dtype=torch.float32)
                for w in _MODE_WEIGHTS[Mode(m)]
            ])
            for m in range(N_MODES)
        ])
        _WEIGHT_TABLE_CACHE[device] = cached
    return cached


# ---------------------------------------------------------------------------------------------
# persistent per-entity state
# ---------------------------------------------------------------------------------------------

def advance_wander(state, bank, cfg, gen) -> torch.Tensor:
    """MUTATES: ent_wander_dir, ent_wander_t. Returns the current (N,E,2) unit explore heading.

    A heading is held for `bots_wander_seconds` (jittered 0.5x-1.5x per draw, so bots don't all
    turn on the same tick) and re-rolled early whenever continuing along it would walk into a
    wall or into the zone -- probed `_WANDER_LOOKAHEAD_TILES` ahead. Re-rolling to a fresh uniform
    angle rather than reflecting off the obstacle is what gets a bot out of a dead-end corner: it
    keeps drawing until it draws an open direction, which takes a couple of ticks at worst.

    Run for EVERY entity every tick, not just the ones currently wandering, so that a bot which
    loses sight of its target has a live heading to fall back on immediately rather than a stale
    one pointing into a wall."""
    pos = state.ent_pos
    heading = state.ent_wander_dir
    remaining = state.ent_wander_t - cfg.dt

    probe = pos + heading * _WANDER_LOOKAHEAD_TILES
    zone_lo, zone_hi, rect_active = shared.zone_rect(state)
    into_zone = perception.in_zone(probe, zone_lo, zone_hi) & rect_active
    into_wall = terrain.sample(bank.blocks_unit, state.map_id, probe, cfg)
    # Covers a freshly reset entity, whose ent_wander_dir is core/state.zero_'s (0,0).
    degenerate = geo.safe_norm(heading, dim=-1) < _EPS
    reroll = (remaining <= 0) | into_zone | into_wall | degenerate

    shape = pos.shape[:-1]
    theta = (torch.rand(shape, generator=gen, device=pos.device) * 2.0 - 1.0) * math.pi
    fresh_dir = geo.from_angle(theta)
    jitter = 0.5 + torch.rand(shape, generator=gen, device=pos.device)
    fresh_t = cfg.bots_wander_seconds * jitter

    state.ent_wander_dir.copy_(torch.where(reroll.unsqueeze(-1), fresh_dir, heading))
    state.ent_wander_t.copy_(torch.where(reroll, fresh_t, remaining))
    return state.ent_wander_dir


def advance_hunt(state, hunt, mode: torch.Tensor, cfg) -> None:
    """MUTATES: ent_hunt_seen, ent_hunt_t -- the visited-waypoint bitmask that turns "walk toward
    a bush" into "sweep the map".

    A waypoint is marked searched when the hunter gets within `bots_hunt_arrive_tiles` of it, OR
    when `bots_hunt_timeout_seconds` elapse without arriving. The timeout is what makes this
    robust: without it, a waypoint behind a wall stays "nearest and unvisited" forever and the
    hunter grinds against that wall for the rest of the episode. Marking it searched anyway is a
    give-up, and give-up is correct -- there is no pathfinder here, only steering.

    **The mask resets once every waypoint has been visited**, so a hunter that has swept the whole
    map starts over rather than degrading into an aimless wanderer for the rest of the episode.
    Episodes are long (3000 ticks at the defaults) and players move, so a second pass over ground
    already covered is genuinely worth walking.

    There is no separate "current target" field. The target is always "nearest unvisited", which
    is stable while the bot walks toward it precisely BECAUSE arrival is what changes the answer --
    so committing whatever is currently selected is equivalent to committing a stored target, with
    one less piece of state to keep consistent.
    """
    hunting = mode == int(Mode.HUNT_BUSH)
    full_timeout = torch.full_like(state.ent_hunt_t, cfg.bots_hunt_timeout_seconds)
    # Anyone not hunting holds a full timer, so a bot entering HUNT_BUSH always gets the complete
    # allowance rather than whatever was left over from a previous hunt.
    remaining = torch.where(hunting, state.ent_hunt_t - cfg.dt, full_timeout)

    arrived = hunt.dist <= cfg.bots_hunt_arrive_tiles  # False where ~found (+inf)
    commit = hunting & hunt.found & (arrived | (remaining <= 0))

    bit = torch.ones_like(state.ent_hunt_seen) << hunt.idx
    seen = torch.where(commit, state.ent_hunt_seen | bit, state.ent_hunt_seen)
    # Sweep complete: usable waypoints exist, but none are unvisited any more. Computed against the
    # count AFTER this tick's commit, not before it -- checking `~hunt.found` alone would only
    # notice one tick later, which shows up as a stray tick of WANDER between sweeps.
    exhausted = hunt.any_valid & ((hunt.n_unvisited - commit.to(torch.int64)) <= 0)
    state.ent_hunt_seen.copy_(torch.where(exhausted, torch.zeros_like(seen), seen))
    state.ent_hunt_t.copy_(torch.where(commit, full_timeout, remaining))


# ---------------------------------------------------------------------------------------------
# mode selection
# ---------------------------------------------------------------------------------------------

def _select_mode(state, tgt, scan, hunt, in_bush_now, clearance, cfg) -> torch.Tensor:
    """(N,E) i64 Mode. Pure function of this tick's state -- no mutation, no randomness, so a
    test can assert one entity's mode directly.

    Reads `state.ent_person` as the source of truth. `cfg.bots_personalities` is NOT consulted
    here: it controls only what core/spawn.sample_personalities ASSIGNS (all RUSH when off), so a
    test or a tool that writes ent_person by hand always gets the behavior it asked for."""
    person = state.ent_person
    has_enemy = tgt.has_enemy
    hp_frac = state.ent_hp / torch.clamp(state.ent_max_hp, min=_EPS)
    low_hp = hp_frac < RETREAT_HP_FRACTION
    # The specification's "unless the green zone is 2 or less squares away".
    zone_pressed = clearance <= cfg.bots_camper_zone_flee_tiles

    def mode(m):
        return torch.full_like(person, int(m))

    # RUSH, and the fallback every bush personality uses when no safe bush is reachable.
    rushlike = torch.where(has_enemy, mode(Mode.CLOSE), mode(Mode.WANDER))

    # CAMPER: hold the bush through everything short of the zone arriving. Deliberately holds
    # even with no enemy visible -- see the module docstring's note on why this one personality is
    # exempt from the global "explore when idle" rule.
    camper = torch.where(in_bush_now, mode(Mode.HOLD_STILL), mode(Mode.TO_BUSH))
    camper = torch.where(zone_pressed, mode(Mode.TO_BUSH), camper)
    camper = torch.where(scan.found, camper, rushlike)

    # HUNTER: fight what it sees, sweep for what it doesn't. Falls through to WANDER only when the
    # map has no bush waypoints at all (blank.csv, walled.csv) -- a hunter that has swept
    # everywhere gets its mask reset by advance_hunt instead of giving up.
    hunter = torch.where(hunt.found, mode(Mode.HUNT_BUSH), mode(Mode.WANDER))
    hunter = torch.where(
        has_enemy, torch.where(low_hp, mode(Mode.RETREAT), mode(Mode.CLOSE)), hunter,
    )

    # TRAPPER: shoots from cover, and when idle relocates to a bush it hasn't sat in yet rather
    # than into the open (HOLD_STILL if there is nowhere new to go).
    trapper_idle = torch.where(hunt.found, mode(Mode.HUNT_BUSH), mode(Mode.HOLD_STILL))
    trapper = torch.where(
        in_bush_now,
        torch.where(has_enemy, mode(Mode.HOLD_STILL), trapper_idle),
        mode(Mode.TO_BUSH),
    )
    trapper = torch.where(zone_pressed, mode(Mode.TO_BUSH), trapper)
    trapper = torch.where(scan.found, trapper, rushlike)

    # KITE: never in cover, always at range.
    kite = torch.where(
        has_enemy, torch.where(low_hp, mode(Mode.RETREAT), mode(Mode.HOLD_RANGE)), mode(Mode.WANDER),
    )

    out = rushlike
    out = torch.where(person == int(Person.CAMPER), camper, out)
    out = torch.where(person == int(Person.HUNTER), hunter, out)
    out = torch.where(person == int(Person.TRAPPER), trapper, out)
    out = torch.where(person == int(Person.KITE), kite, out)
    return out


# ---------------------------------------------------------------------------------------------
# the movement layer
# ---------------------------------------------------------------------------------------------

def movement(state, tgt, bank, params, cfg, gen):
    """MUTATES: ent_wander_dir, ent_wander_t, ent_hunt_seen, ent_hunt_t.
    Returns (move_dir (N,E,2) unit-or-zero, mode (N,E) i64).

    Computed for ALL (N,E) entities in one pass -- including entity 0, whose result
    bots/policy.all_bot_intents discards (the hero moves via hero.decode_action). Same
    compute-for-everyone-select-later discipline as the archetype combat functions.
    """
    pos = state.ent_pos
    E = pos.shape[1]
    zone_lo, zone_hi, _rect_active = shared.zone_rect(state)

    # --- the two shared spatial queries, each computed ONCE for every (N,E) entity. `scan` is
    # local ("nearest cover"), `hunt` is map-scale ("where should I look next"); see their
    # docstrings for why one cannot serve both roles. ---
    scan = perception.bush_scan(
        pos, state.map_id, bank, cfg,
        zone_lo=zone_lo, zone_hi=zone_hi, zone_margin=cfg.bots_camper_zone_flee_tiles,
    )
    hunt = perception.hunt_waypoint(
        pos, state.map_id, bank, cfg, state.ent_hunt_seen,
        zone_lo=zone_lo, zone_hi=zone_hi, zone_margin=cfg.bots_camper_zone_flee_tiles,
    )
    in_bush_now = perception.in_bush(state, bank)
    clearance = shared.zone_clearance(state, cfg)
    wander_dir = advance_wander(state, bank, cfg, gen)

    mode = _select_mode(state, tgt, scan, hunt, in_bush_now, clearance, cfg)
    advance_hunt(state, hunt, mode, cfg)

    # --- steering directions ---
    # Every enemy-relative term is gated on has_enemy below, and that gate is load-bearing, not
    # defensive: bots/policy.target_info CLAMPS an absent target index to 0, so tgt.enemy_pos for
    # a bot with no target is the HERO's position. Ungated, every idle bot on the map would walk
    # straight at the agent with no way of having seen it.
    seek_dir = steering.seek(pos, tgt.enemy_pos)
    range_dir = steering.maintain_range(pos, tgt.enemy_pos, tgt.desired_range, RANGE_DEADBAND)
    strafe_dir = steering.strafe(pos, tgt.enemy_pos, shared.strafe_sign(E, pos.device))
    flee_dir = steering.flee(pos, tgt.enemy_pos)
    bush_dir = steering.seek(pos, scan.pos)
    hunt_dir = steering.seek(pos, hunt.pos)

    # --- weights: one gather, then mask each term by whether its target actually exists ---
    w = _weight_table(pos.device)[mode]  # (N,E,N_TERMS)
    enemy_f = tgt.has_enemy.to(w.dtype)
    found_f = scan.found.to(w.dtype)
    hunt_f = hunt.found.to(w.dtype)

    # --- universal terms, identical for every personality ---
    zone_dir, zone_w = shared.zone_contribution(state, cfg)
    avoid_dir, avoid_w = shared.zone_avoid_contribution(state, cfg)
    box_dir, box_w = shared.box_contribution(state, cfg)
    cube_dir, cube_w = shared.cube_contribution(state, cfg)

    # HOLD_STILL has to mean STILL. The three optional pulls are suppressed for it -- otherwise a
    # camper in a bush would be dragged out by a loot box 9 tiles away, or nudged off its tile by
    # zone avoidance while still perfectly safe, and its "does not leave the bush until the zone
    # is 2 tiles out" contract would be quietly false. zone_contribution is NOT suppressed: a bot
    # the zone has actually swallowed leaves, personality notwithstanding.
    mobile = (mode != int(Mode.HOLD_STILL)).to(w.dtype)

    move_dir = steering.combine(
        (seek_dir, w[..., 0] * enemy_f),
        (range_dir, w[..., 1] * enemy_f),
        (strafe_dir, w[..., 2] * enemy_f),
        (flee_dir, w[..., 3] * enemy_f),
        (bush_dir, w[..., 4] * found_f),
        (hunt_dir, w[..., 5] * hunt_f),
        (wander_dir, w[..., 6]),
        (zone_dir, zone_w),
        (avoid_dir, avoid_w * mobile),
        (box_dir, box_w * mobile),
        (cube_dir, cube_w * mobile),
    )
    return move_dir, mode


def fire_allowed(state, tgt, cfg) -> torch.Tensor:
    """(N,E) bool, ANDed onto every archetype's fire decision by
    bots/policy.all_bot_intents.

    Only CAMPER restricts anything: "makes no attacks unless the other player can see it". Keyed
    off `tgt.seen_by_other` (does ANY other entity currently see me) rather than off the camper's
    own target specifically, because the target-specific version has a hole -- a camper being shot
    by A while its sticky target is a nearer, non-looking B would sit there and take it. A camper
    that has been spotted fights back, whoever spotted it.

    Note how this composes with `perception.reveal_after_attack`: firing sets the camper's own
    reveal timer, so the shot that breaks its cover also keeps it broken for a second afterward.
    A camper cannot fire from concealment and stay concealed."""
    silent = (state.ent_person == int(Person.CAMPER)) & ~tgt.seen_by_other
    return ~silent
