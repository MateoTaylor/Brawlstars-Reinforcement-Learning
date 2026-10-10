"""Bot personality tests: one section per personality, then the rules that must hold
for ALL of them, then the edge cases where a bot could end up undirected or the agent could game
the system.

Movement assertions read `personality.movement`'s RAW output, not `policy.all_bot_intents`'
output: the dispatcher low-passes movement through ent_move_smooth (an EMA with a per-archetype
reaction delay), so a single-tick assertion through the dispatcher measures the EMA's ramp rather
than the steering decision. Tests that genuinely need the smoothed, integrated behavior run many
ticks through the real env instead (see the acceptance section at the bottom).
"""
import torch

from tests.bot_fixtures import FakeBank, build_targeting, cfg_and_params, fresh_state, grid
from brawl_sim.bots import perception, personality, policy
from brawl_sim.bots.personality import Mode
from brawl_sim.constants import Kind, Person, Tile
from brawl_sim.core import geometry as geo
from brawl_sim.core import stats
from brawl_sim.core.movement import apply_movement
from brawl_sim.maps import nav


def _move(state, bank, params, cfg, gen):
    """(move_dir, mode) for the whole batch."""
    _vis, tgt = build_targeting(state, bank, params, cfg)
    return personality.movement(state, tgt, bank, params, cfg, gen)


def _mode_of(state, bank, params, cfg, gen, env=0, ent=1):
    return Mode(int(_move(state, bank, params, cfg, gen)[1][env, ent]))


BUSH_E = torch.tensor([13.5, 10.5])   # tiles[10, 13]
BUSH_W = torch.tensor([10.5, 10.5])   # tiles[10, 10]


def _one_bush_grid(h=20, w=20):
    """A single bush tile, so "the nearest bush" is unambiguous and "no other cover exists" is
    guaranteed."""
    tiles = grid(h, w)
    tiles[10, 13] = Tile.BUSH
    return tiles


def _two_bush_grid(h=20, w=20):
    """Two bushes 3.0 tiles apart -- close enough that one local `bush_scan` sees both."""
    tiles = grid(h, w)
    tiles[10, 13] = Tile.BUSH
    tiles[10, 10] = Tile.BUSH
    return tiles


def _waypoint_bank(cfg, *cells):
    """A FakeBank whose bush waypoints are exactly `cells` (each an (x, y) tile-center), bypassing
    maps/loader.bush_waypoints' cell subsampling so a hunt test can place its own destinations. The
    bushes are also laid into the tile grid so `in_bush` and `bush_scan` agree with the waypoints."""
    tiles = grid(cfg.map_h, cfg.map_w)
    for x, y in cells:
        tiles[int(y), int(x)] = Tile.BUSH
    bank = FakeBank(tiles)
    pts = torch.tensor([[x + 0.5, y + 0.5] for x, y in cells], dtype=torch.float32)
    bank.bush_wp = pts.unsqueeze(0)                              # (M=1, W, 2)
    bank.n_bush_wp = torch.tensor([len(cells)], dtype=torch.int64)
    return bank


def _no_waypoint_bank(cfg, tiles=None):
    """A FakeBank with bush TILES but zero hunt waypoints -- what a map with no bush at all gives
    (blank.csv, walled.csv), and what an already-fully-swept hunter effectively sees."""
    bank = FakeBank(tiles if tiles is not None else grid(cfg.map_h, cfg.map_w))
    bank.bush_wp = torch.zeros((1, 0, 2))
    bank.n_bush_wp = torch.zeros((1,), dtype=torch.int64)
    return bank


# =============================================================================================
# RUSH
# =============================================================================================

def test_rush_closes_on_a_visible_enemy():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.RUSH)
    bank = FakeBank(grid(20, 20))

    state.ent_pos[0, 0] = torch.tensor([5.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([15.0, 10.0])  # bot east of the hero

    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.CLOSE
    assert move[0, 1, 0].item() < 0  # moves west, toward the hero


def test_rush_never_retreats_even_at_low_hp_and_point_blank():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.RUSH)
    bank = FakeBank(grid(20, 20))

    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([10.4, 10.0])
    state.ent_hp[0, 1] = 0.05 * state.ent_max_hp[0, 1]

    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.CLOSE
    assert move[0, 1, 0].item() <= 0  # any x-motion is toward the hero, never away


# =============================================================================================
# CAMPER
# =============================================================================================

def test_camper_walks_to_the_nearest_bush_then_stops_dead_inside_it():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.CAMPER)
    bank = FakeBank(_one_bush_grid())

    state.ent_alive[0, 0] = False  # nothing visible anywhere
    state.ent_pos[0, 1] = torch.tensor([10.5, 10.5])  # 3 tiles west of the only bush

    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.TO_BUSH
    assert move[0, 1, 0].item() > 0  # heads east, toward the bush

    state.ent_pos[0, 1] = BUSH_E.clone()  # now standing in it
    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.HOLD_STILL
    assert torch.allclose(move[0, 1], torch.zeros(2))  # exactly zero: it does not budge


def test_camper_holds_even_with_an_enemy_visible():
    """The distinguishing property: a camper in cover does not react to being seen by moving --
    only by shooting, and only once it has been spotted (tested separately below)."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.CAMPER)
    bank = FakeBank(_one_bush_grid())

    state.ent_pos[0, 1] = BUSH_E.clone()
    state.ent_pos[0, 0] = torch.tensor([9.0, 10.5])  # hero close enough to be a target

    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert bool(tgt.has_enemy[0, 1])
    move, mode = personality.movement(state, tgt, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.HOLD_STILL
    assert torch.allclose(move[0, 1], torch.zeros(2))


def test_camper_does_not_fire_until_something_can_see_it():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.CAMPER)
    bank = FakeBank(_one_bush_grid())
    bush_reveal = params.bush_reveal_radius[0].item()

    # In the bush, hero outside bush_reveal_radius: the camper can see the hero (the hero is not
    # in a bush) but the hero cannot see the camper.
    state.ent_pos[0, 1] = BUSH_E.clone()
    state.ent_pos[0, 0] = torch.tensor([13.5 - (bush_reveal + 2.0), 10.5])

    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert bool(tgt.has_enemy[0, 1])          # it has a target
    assert not bool(tgt.seen_by_other[0, 1])  # but nobody can see it
    assert not bool(personality.fire_allowed(state, tgt, stats.aggression_of(state.ent_kind, params), cfg)[0, 1])

    # Step inside the reveal radius and the camper is exposed -- and opens fire.
    state.ent_pos[0, 0] = torch.tensor([13.5 - (bush_reveal - 0.5), 10.5])
    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert bool(tgt.seen_by_other[0, 1])
    assert bool(personality.fire_allowed(state, tgt, stats.aggression_of(state.ent_kind, params), cfg)[0, 1])


def test_camper_fires_back_at_a_spotter_that_is_not_its_own_target():
    """`fire_allowed` keys off "can ANYONE see me", not "can my target see me". With two enemies
    -- a near one that cannot see the camper and a far one that can -- the target-specific version
    would leave the camper sitting silently while the far one shot it."""
    cfg, params, gen = cfg_and_params(n_enemies=2)
    state = fresh_state(cfg, params, person=Person.CAMPER)
    tiles = _one_bush_grid()
    tiles[10, 8] = Tile.BUSH  # the near bot hides too, so the camper's sticky target is the hero
    bank = FakeBank(tiles)

    state.ent_pos[0, 1] = BUSH_E.clone()              # camper, in a bush
    state.ent_pos[0, 0] = torch.tensor([12.0, 10.5])  # hero: near, in the open, sees the camper
    state.ent_pos[0, 2] = torch.tensor([8.5, 10.5])   # other bot: far, in a bush

    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert bool(tgt.seen_by_other[0, 1])
    assert bool(personality.fire_allowed(state, tgt, stats.aggression_of(state.ent_kind, params), cfg)[0, 1])


def test_camper_abandons_its_bush_when_the_zone_closes_to_two_tiles():
    cfg, params, gen = cfg_and_params(n_enemies=1, overrides={"zone": {"enabled": True}})
    state = fresh_state(cfg, params, person=Person.CAMPER)
    tiles = grid(20, 20)
    tiles[10, 12] = Tile.BUSH  # the bush it sits in, near the eventual zone edge
    tiles[10, 8] = Tile.BUSH   # a second bush deeper inside the safe rect
    bank = FakeBank(tiles)

    state.ent_pos[0, 1] = torch.tensor([12.5, 10.5])
    state.ent_alive[0, 0] = False

    # Roomy rect: clearance is well over camper_zone_flee_tiles, so it holds.
    state.zone_lo[0] = torch.tensor([2.0, 2.0])
    state.zone_hi[0] = torch.tensor([18.0, 18.0])
    assert _mode_of(state, bank, params, cfg, gen) is Mode.HOLD_STILL

    # Squeeze the rect so the camper's own clearance drops to 1.5 tiles (< the 2.0 default).
    state.zone_hi[0] = torch.tensor([14.0, 18.0])
    clearance = policy.zone_clearance(state, cfg)[0, 1].item()
    assert clearance < cfg.bots_camper_zone_flee_tiles
    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.TO_BUSH
    assert move[0, 1, 0].item() < 0  # relocates WEST, deeper into the safe rect


def test_a_bush_too_close_to_the_zone_edge_is_never_chosen_as_a_destination():
    """Without the zone MARGIN in bush_scan, a camper fleeing the zone could pick the very bush it
    is standing in (still technically safe) and therefore never move at all."""
    cfg, params, gen = cfg_and_params(n_enemies=1, overrides={"zone": {"enabled": True}})
    state = fresh_state(cfg, params, person=Person.CAMPER)
    tiles = grid(20, 20)
    tiles[10, 12] = Tile.BUSH
    bank = FakeBank(tiles)

    state.ent_pos[0, 1] = torch.tensor([12.5, 10.5])
    state.zone_lo[0] = torch.tensor([2.0, 2.0])
    state.zone_hi[0] = torch.tensor([14.0, 18.0])  # bush clearance = 1.5 < margin 2.0

    zone_lo, zone_hi, _ = policy.zone_rect(state)
    scan = perception.bush_scan(
        state.ent_pos, state.map_id, bank, cfg, zone_lo=zone_lo, zone_hi=zone_hi,
        zone_margin=cfg.bots_camper_zone_flee_tiles,
    )
    assert not bool(scan.found[0, 1])  # the only bush on the map is rejected

    # And with no acceptable bush, the camper falls back to RUSH behavior (here: wander, since
    # nothing is visible) rather than standing in a doomed tile.
    state.ent_alive[0, 0] = False
    assert _mode_of(state, bank, params, cfg, gen) is Mode.WANDER


# =============================================================================================
# HUNTER
# =============================================================================================

def test_hunter_walks_to_the_nearest_unvisited_waypoint():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.HUNTER)
    bank = _waypoint_bank(cfg, (16, 10), (3, 10))  # east waypoint nearer than the west one

    state.ent_alive[0, 0] = False
    state.ent_pos[0, 1] = torch.tensor([12.0, 10.5])

    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.HUNT_BUSH
    assert move[0, 1, 0].item() > 0  # east, to the nearer one


def test_hunter_marks_a_waypoint_visited_on_arrival_and_moves_to_the_next_one():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.HUNTER)
    bank = _waypoint_bank(cfg, (16, 10), (3, 10))

    state.ent_alive[0, 0] = False
    state.ent_pos[0, 1] = torch.tensor([16.5, 10.5])  # standing ON the east waypoint

    # First evaluation: inside hunt_arrive_tiles, so advance_hunt sets that waypoint's bit.
    _move(state, bank, params, cfg, gen)
    assert int(state.ent_hunt_seen[0, 1]) != 0

    # Second evaluation: the east waypoint is masked out, so the west one becomes the target.
    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.HUNT_BUSH
    assert move[0, 1, 0].item() < 0  # now heads west, across the map


def test_hunter_visits_every_waypoint_then_starts_a_fresh_sweep():
    """A hunter that has been everywhere must start over, not degrade into a wanderer for the rest
    of a 3000-tick episode."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.HUNTER)
    bank = _waypoint_bank(cfg, (16, 10), (3, 10))

    state.ent_alive[0, 0] = False
    state.ent_pos[0, 1] = torch.tensor([16.5, 10.5])
    _move(state, bank, params, cfg, gen)                 # visits the east one
    state.ent_pos[0, 1] = torch.tensor([3.5, 10.5])
    _move(state, bank, params, cfg, gen)                 # visits the west one -> all visited
    assert int(state.ent_hunt_seen[0, 1]) == 0           # mask cleared: sweep restarts

    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.HUNT_BUSH        # hunting again, not wandering


def test_hunter_gives_up_on_an_unreachable_waypoint_after_the_timeout():
    """A waypoint behind a wall stays "nearest and unvisited" forever; without the timeout the
    hunter would grind against that wall for the rest of the episode."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.HUNTER)
    bank = _waypoint_bank(cfg, (13, 10), (3, 10))
    walled = grid(20, 20)
    walled[10, 13] = Tile.BUSH
    walled[3, 3] = Tile.BUSH
    walled[6:15, 11] = Tile.WALL  # seals the east waypoint off
    bank.blocks_unit = FakeBank(walled).blocks_unit

    state.ent_alive[0, 0] = False
    state.ent_pos[0, 1] = torch.tensor([9.5, 10.5])
    state.ent_hunt_t[0, 1] = cfg.dt / 2.0  # timeout expires on the very next evaluation

    _move(state, bank, params, cfg, gen)
    assert int(state.ent_hunt_seen[0, 1]) != 0                        # gave up, marked it visited
    assert state.ent_hunt_t[0, 1].item() == cfg.bots_hunt_timeout_seconds  # timer rearmed


def test_hunter_wanders_on_a_map_with_no_bush_waypoints_at_all():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.HUNTER)
    bank = _no_waypoint_bank(cfg)
    state.ent_alive[0, 0] = False
    assert _mode_of(state, bank, params, cfg, gen) is Mode.WANDER
    assert int(state.ent_hunt_seen[0, 1]) == 0  # nothing to mark, and no spurious reset churn


def test_hunter_engages_a_visible_enemy_instead_of_hunting():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.HUNTER)
    bank = _waypoint_bank(cfg, (16, 10), (3, 10))

    state.ent_pos[0, 0] = torch.tensor([8.0, 10.5])
    state.ent_pos[0, 1] = torch.tensor([12.0, 10.5])
    assert _mode_of(state, bank, params, cfg, gen) is Mode.CLOSE


def test_hunter_retreats_at_low_hp():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.HUNTER)
    bank = FakeBank(grid(20, 20))

    state.ent_pos[0, 0] = torch.tensor([8.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([12.0, 10.0])
    state.ent_hp[0, 1] = (personality.RETREAT_HP_FRACTION - 0.05) * state.ent_max_hp[0, 1]

    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.RETREAT
    assert move[0, 1, 0].item() > 0  # east, away from the hero


# =============================================================================================
# TRAPPER
# =============================================================================================

def test_trapper_holds_its_bush_while_an_enemy_is_visible():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.TRAPPER)
    bank = FakeBank(_one_bush_grid())

    state.ent_pos[0, 1] = BUSH_E.clone()
    state.ent_pos[0, 0] = torch.tensor([9.0, 10.5])

    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.HOLD_STILL
    assert torch.allclose(move[0, 1], torch.zeros(2))


def test_trapper_fires_freely_unlike_a_camper():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.TRAPPER)
    bank = FakeBank(_one_bush_grid())
    bush_reveal = params.bush_reveal_radius[0].item()

    state.ent_pos[0, 1] = BUSH_E.clone()                                    # in a bush
    state.ent_pos[0, 0] = torch.tensor([13.5 - (bush_reveal + 2.0), 10.5])  # cannot see the trapper

    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert not bool(tgt.seen_by_other[0, 1])
    assert bool(personality.fire_allowed(state, tgt, stats.aggression_of(state.ent_kind, params), cfg)[0, 1])  # shoots anyway


def test_trapper_drifts_to_a_different_bush_when_idle():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.TRAPPER)
    bank = _waypoint_bank(cfg, (16, 10), (3, 10))

    state.ent_alive[0, 0] = False                     # nothing visible
    state.ent_pos[0, 1] = torch.tensor([16.5, 10.5])  # sitting in the east bush

    # First evaluation commits the waypoint it is standing on; the next sends it to the other.
    assert _mode_of(state, bank, params, cfg, gen) is Mode.HUNT_BUSH
    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.HUNT_BUSH
    assert move[0, 1, 0].item() < 0  # west, to the other bush


def test_trapper_with_nowhere_new_to_go_holds_instead_of_wandering_into_the_open():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.TRAPPER)
    # Bush tiles exist (so it has cover) but there is nowhere to relocate TO.
    bank = _no_waypoint_bank(cfg, _one_bush_grid())

    state.ent_alive[0, 0] = False
    state.ent_pos[0, 1] = BUSH_E.clone()

    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.HOLD_STILL
    assert torch.allclose(move[0, 1], torch.zeros(2))


# =============================================================================================
# KITE
# =============================================================================================

def test_kite_closes_when_too_far_and_backs_off_when_too_close():
    cfg, params, gen = cfg_and_params(n_enemies=1, map_h=40, map_w=40)
    state = fresh_state(cfg, params, person=Person.KITE)
    bank = FakeBank(grid(40, 40))

    attack_range = params.attack_range[0, int(Kind.BOT_SNIPER)].item()
    desired = params.desired_range_fraction[0, int(Kind.BOT_SNIPER)].item() * attack_range

    state.ent_pos[0, 0] = torch.tensor([5.0, 20.0])
    state.ent_pos[0, 1] = torch.tensor([5.0 + desired + 4.0, 20.0])  # too far
    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.HOLD_RANGE
    assert move[0, 1, 0].item() < 0  # closes in (west)

    state.ent_pos[0, 1] = torch.tensor([5.0 + desired - 4.0, 20.0])  # too close
    move, _mode = _move(state, bank, params, cfg, gen)
    assert move[0, 1, 0].item() > 0  # backs off (east)


def test_kite_inside_the_deadband_orbits_rather_than_closing_or_backing_off():
    cfg, params, gen = cfg_and_params(n_enemies=1, map_h=40, map_w=40)
    state = fresh_state(cfg, params, person=Person.KITE)
    bank = FakeBank(grid(40, 40))

    attack_range = params.attack_range[0, int(Kind.BOT_SNIPER)].item()
    desired = params.desired_range_fraction[0, int(Kind.BOT_SNIPER)].item() * attack_range
    state.ent_pos[0, 0] = torch.tensor([5.0, 20.0])
    state.ent_pos[0, 1] = torch.tensor([5.0 + desired, 20.0])  # exactly at the ideal range

    move, mode = _move(state, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) is Mode.HOLD_RANGE
    # maintain_range is exactly zero in its deadband, so the only surviving term is strafe:
    # purely lateral, no radial component along the x axis separating the two.
    assert abs(move[0, 1, 0].item()) < 1e-6
    assert abs(move[0, 1, 1].item()) > 0.5


def test_kite_wanders_with_nothing_visible():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.KITE)
    bank = FakeBank(grid(20, 20))
    state.ent_alive[0, 0] = False
    assert _mode_of(state, bank, params, cfg, gen) is Mode.WANDER


# =============================================================================================
# rules that hold for EVERY personality
# =============================================================================================

def test_no_bushes_on_the_map_makes_every_bush_personality_behave_like_rush():
    """The single most-exercised fallback: `walled` and `blank` have no bush tiles at all."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    bank = FakeBank(grid(20, 20))  # no bushes anywhere

    for person in (Person.CAMPER, Person.HUNTER, Person.TRAPPER):
        state = fresh_state(cfg, params, person=person)
        state.ent_pos[0, 0] = torch.tensor([5.0, 10.0])
        state.ent_pos[0, 1] = torch.tensor([15.0, 10.0])
        move, mode = _move(state, bank, params, cfg, gen)
        assert Mode(int(mode[0, 1])) is Mode.CLOSE, person
        assert move[0, 1, 0].item() < 0, person  # closes on the hero, exactly like RUSH

        state.ent_alive[0, 0] = False  # and explores when there is nothing to close on
        assert _mode_of(state, bank, params, cfg, gen) is Mode.WANDER, person


def test_every_personality_avoids_walking_into_the_zone():
    """Universal rule. Each bot sits 1 tile inside the safe rect's east edge with its personality's
    own preferred destination placed OUTSIDE the rect, and must still move inward."""
    cfg, params, gen = cfg_and_params(n_enemies=1, overrides={"zone": {"enabled": True}})
    tiles = grid(20, 20)
    tiles[10, 16] = Tile.BUSH  # a bush out in the doomed area, to bait the bush personalities
    bank = FakeBank(tiles)

    for person in Person:
        state = fresh_state(cfg, params, person=person)
        state.zone_lo[0] = torch.tensor([2.0, 2.0])
        state.zone_hi[0] = torch.tensor([14.0, 18.0])
        state.ent_pos[0, 1] = torch.tensor([13.0, 10.0])   # 1 tile inside the east edge
        state.ent_pos[0, 0] = torch.tensor([17.0, 10.0])   # hero bait, outside the rect
        move, _mode = _move(state, bank, params, cfg, gen)
        assert move[0, 1, 0].item() < 0, f"{person!r} steered toward the zone"


def test_a_bot_already_in_the_zone_escapes_regardless_of_personality():
    cfg, params, gen = cfg_and_params(n_enemies=1, overrides={"zone": {"enabled": True}})
    tiles = grid(20, 20)
    tiles[10, 17] = Tile.BUSH  # a bush right where the camper is standing, deep in the zone
    bank = FakeBank(tiles)

    for person in Person:
        state = fresh_state(cfg, params, person=person)
        state.zone_lo[0] = torch.tensor([2.0, 2.0])
        state.zone_hi[0] = torch.tensor([14.0, 18.0])
        state.ent_alive[0, 0] = False
        state.ent_pos[0, 1] = torch.tensor([17.5, 10.5])  # outside the rect, taking damage
        move, _mode = _move(state, bank, params, cfg, gen)
        assert move[0, 1, 0].item() < 0, f"{person!r} did not flee the zone"


def test_no_living_bot_is_ever_undirected_unless_it_is_deliberately_holding():
    """Fuzz for the failure mode where a steering blend cancels to zero and a bot just stands
    there. The ONLY personality/state combination allowed to produce a zero move_dir is a
    HOLD_STILL mode; anything else must have somewhere to go."""
    cfg, params, gen = cfg_and_params(n_envs=16, n_enemies=5, overrides={"zone": {"enabled": True}})
    tiles = grid(20, 20)
    tiles[6, 6] = Tile.BUSH
    tiles[14, 14] = Tile.BUSH
    bank = FakeBank(tiles)
    state = fresh_state(cfg, params, n_envs=16)

    for _ in range(60):
        state.ent_pos.uniform_(1.5, 18.5)
        state.ent_person.random_(0, len(Person))
        state.ent_hp.uniform_(0.05, 1.0)
        state.ent_hp.mul_(state.ent_max_hp)
        state.ent_alive.copy_(torch.rand(state.ent_alive.shape) > 0.3)
        state.zone_lo.uniform_(1.0, 6.0)
        state.zone_hi.uniform_(14.0, 19.0)

        move, mode = _move(state, bank, params, cfg, gen)
        assert torch.all(torch.isfinite(move))

        magnitude = geo.safe_norm(move, dim=-1)
        stuck = (magnitude < 1e-5) & state.ent_alive
        holding = mode == int(Mode.HOLD_STILL)
        # Entity 0 is the hero, whose movement comes from decode_action and whose personality
        # result the dispatcher discards.
        stuck = stuck[:, 1:]
        holding = holding[:, 1:]
        assert torch.all(~stuck | holding)


def test_idle_bots_do_not_converge_on_a_shared_point():
    """The regression this whole step exists for. Pre-Step-41, every idle melee bot steered at the
    zone rect's center, so a lobby collapsed into one scrum an agent could simply avoid. With
    nothing visible, bots must spread out instead."""
    cfg, params, gen = cfg_and_params(n_envs=1, n_enemies=8)
    bank = FakeBank(grid(20, 20))  # no bushes: forces every personality onto its idle path
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_MELEE)
    state.ent_person[0, 1:] = torch.tensor([int(p) for p in Person] + [0, 0, 0])
    state.ent_alive[0, 0] = False  # nothing visible to anyone

    # All eight bots start on the same spot: if idle movement were a shared destination they would
    # all pick the identical heading.
    state.ent_pos[0, 1:] = torch.tensor([10.5, 10.5])
    move, _mode = _move(state, bank, params, cfg, gen)

    headings = move[0, 1:]
    pairwise = headings @ headings.T  # unit vectors, so this is cos(angle between)
    off_diagonal = pairwise[~torch.eye(headings.shape[0], dtype=torch.bool)]
    assert off_diagonal.min().item() < 0.9  # at least one pair points meaningfully apart
    assert off_diagonal.mean().item() < 0.5  # and they are not merely jittered around one heading


def test_wander_heading_is_rerolled_rather_than_walking_into_a_wall():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    bank = FakeBank(grid(20, 20))
    state = fresh_state(cfg, params, person=Person.RUSH)
    state.ent_alive[0, 0] = False
    state.ent_pos[0, 1] = torch.tensor([1.5, 10.0])       # hugging the west wall
    state.ent_wander_dir[0, 1] = torch.tensor([-1.0, 0.0])  # pointed straight into it
    state.ent_wander_t[0, 1] = 99.0                       # timer would NOT have expired

    personality.advance_wander(state, bank, cfg, gen)
    assert not torch.allclose(state.ent_wander_dir[0, 1], torch.tensor([-1.0, 0.0]))


def test_wander_headings_are_unit_length_and_never_degenerate():
    cfg, params, gen = cfg_and_params(n_envs=8, n_enemies=4)
    bank = FakeBank(grid(20, 20))
    state = fresh_state(cfg, params, n_envs=8)
    assert torch.allclose(state.ent_wander_dir, torch.zeros_like(state.ent_wander_dir))

    for _ in range(20):
        state.ent_pos.uniform_(1.5, 18.5)
        personality.advance_wander(state, bank, cfg, gen)
        norms = geo.safe_norm(state.ent_wander_dir, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# =============================================================================================
# loot boxes: bots contest them (the "rush attacks lootboxes" requirement)
# =============================================================================================

def test_a_bot_with_no_enemy_adopts_a_nearby_box_as_a_fire_target():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.RUSH)
    bank = FakeBank(grid(20, 20))

    state.ent_alive[0, 0] = False
    state.ent_pos[0, 1] = torch.tensor([10.0, 10.0])
    state.box_alive[0, 0] = True
    state.box_pos[0, 0] = torch.tensor([13.0, 10.0])

    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert bool(tgt.is_box[0, 1]) and bool(tgt.has_target[0, 1])
    assert torch.allclose(tgt.pos[0, 1], torch.tensor([13.0, 10.0]))


def test_a_visible_enemy_always_outranks_a_box():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.RUSH)
    bank = FakeBank(grid(20, 20))

    state.ent_pos[0, 0] = torch.tensor([12.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([10.0, 10.0])
    state.box_alive[0, 0] = True
    state.box_pos[0, 0] = torch.tensor([10.5, 10.0])  # closer than the hero, still ignored

    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert not bool(tgt.is_box[0, 1])
    assert torch.allclose(tgt.pos[0, 1], state.ent_pos[0, 0])


def test_a_camper_never_shoots_a_box():
    """Shooting reveals you (env.py wires reveal_after_attack), so a camper breaking cover for a
    loot box would defeat its own purpose -- and the specification contrasts TRAPPER ("attacks any
    boxes and players in range") against CAMPER precisely here."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    bank = FakeBank(_one_bush_grid())

    for person, expected in ((Person.CAMPER, False), (Person.TRAPPER, True)):
        state = fresh_state(cfg, params, person=person)
        state.ent_alive[0, 0] = False
        state.ent_pos[0, 1] = BUSH_E.clone()
        state.box_alive[0, 0] = True
        state.box_pos[0, 0] = torch.tensor([14.5, 10.5])
        _vis, tgt = build_targeting(state, bank, params, cfg)
        assert bool(tgt.is_box[0, 1]) is expected, person


def test_a_crate_is_a_fire_target_only_within_five_tiles():
    """The lead, 2026-09-25: bots shoot the crate beside them, not every crate in range. The
    sniper reaches 8 tiles; a crate at 6 is no target, one at 4 is."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    bank = FakeBank(grid(20, 20))
    assert params.attack_range[0, int(Kind.BOT_SNIPER)].item() > policy._BOX_TARGET_TILES
    for x, expected in ((16.0, False), (14.0, True)):
        state = fresh_state(cfg, params, person=Person.RUSH)
        state.ent_alive[0, 0] = False
        state.ent_pos[0, 1] = torch.tensor([10.0, 10.0])
        state.box_alive[0, 0] = True
        state.box_pos[0, 0] = torch.tensor([x, 10.0])
        _vis, tgt = build_targeting(state, bank, params, cfg)
        assert bool(tgt.is_box[0, 1]) is expected, x


def test_a_camper_is_never_pulled_to_a_crate():
    """`targeting` never lets a camper shoot a crate, so a pull could only park it beside one:
    27 % of camper time within 1.5 tiles of a crate, measured 2026-09-25 with the raw-vector
    pull, 4 % before it. Any mobile personality is pulled at the full weight."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    bank = FakeBank(grid(20, 20))
    for person, expected_w in ((Person.CAMPER, 0.0), (Person.RUSH, policy._BOX_APPROACH_WEIGHT)):
        state = fresh_state(cfg, params, person=person)
        state.ent_alive[0, 0] = False
        state.ent_pos[0, 1] = torch.tensor([10.0, 10.0])
        state.box_alive[0, 0] = True
        state.box_pos[0, 0] = torch.tensor([13.0, 10.0])
        build_targeting(state, bank, params, cfg)
        direction, w = policy.box_contribution(state, bank, params, cfg)
        assert w[0, 1].item() == expected_w, person
        assert torch.allclose(direction[0, 1], torch.tensor([1.0, 0.0]))  # a unit direction


def test_an_engaged_bot_leaves_a_far_cube_but_grabs_one_at_its_feet():
    """The cube pull has the crate pull's enemy gate (no target within 8 tiles) with one
    exception, a cube within 2 tiles: that is where a kill's drop lands, and the next enemy is
    usually in sight. Without any gate (the first strong version) engaged bots walked away from
    their target 40 % of the time, measured 2026-09-25."""
    cfg, params, gen = cfg_and_params(n_enemies=1, map_h=30, map_w=30)
    bank = FakeBank(grid(30, 30))
    for cube_x, expected_w in ((6.0, 0.0), (11.0, policy._CUBE_COLLECT_WEIGHT)):
        state = fresh_state(cfg, params, person=Person.RUSH)
        state.ent_pos[0, 0] = torch.tensor([17.0, 15.0])  # the hero 5 tiles east: inside the gate
        state.ent_pos[0, 1] = torch.tensor([12.0, 15.0])
        state.pku_alive[0, 0] = True
        state.pku_pos[0, 0] = torch.tensor([cube_x, 15.0])
        _vis, tgt = build_targeting(state, bank, params, cfg)
        assert bool(tgt.has_enemy[0, 1])
        _direction, w = policy.cube_contribution(state, bank, params, cfg)
        assert w[0, 1].item() == expected_w, cube_x


def test_a_loot_pull_replaces_the_personality_steering():
    """steering.seek is raw `target - pos` and combine normalises once, so a summed pull weighed
    weight x distance and two pulls rested at a weighted midpoint (measured 2026-09-25: campers
    parked at crates, retreating bots walked back toward the enemy for a cube). A RUSH bot two
    tiles from a crate with an enemy ten tiles the other way (past the 8-tile loot gate, so the
    pull is live) now walks to the crate. Under the old sum the enemy's 1.0 x 10 beat the crate's
    3.0 x 2, and the bot walked away from loot at its feet toward an enemy it could not reach."""
    cfg, params, gen = cfg_and_params(n_enemies=1, map_h=30, map_w=30)
    bank = FakeBank(grid(30, 30))
    state = fresh_state(cfg, params, person=Person.RUSH)
    state.ent_pos[0, 0] = torch.tensor([22.0, 15.0])  # the hero 10 tiles east, in sight
    state.ent_pos[0, 1] = torch.tensor([12.0, 15.0])
    state.box_alive[0, 0] = True
    state.box_pos[0, 0] = torch.tensor([10.0, 15.0])  # a crate 2 tiles west
    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert bool(tgt.has_enemy[0, 1])
    move_dir, mode = personality.movement(state, tgt, bank, params, cfg, gen)
    assert Mode(int(mode[0, 1])) == Mode.CLOSE
    assert torch.allclose(move_dir[0, 1], torch.tensor([-1.0, 0.0]), atol=1e-5)


def test_a_retreating_bot_is_pulled_to_no_loot():
    """RETREAT exists to break contact, so neither pull is live in it. A HUNTER at a fifth of its
    HP with the enemy ten tiles east (past the 8-tile loot gate, so both pulls would otherwise be
    live) and a crate, then a cube, two tiles east between them: it flees west both times. With
    the crate pull live a fleeing bot turned around on 0.60 of those decisions (0.09 of retreat
    decisions at elite, measured 2026-09-25)."""
    cfg, params, gen = cfg_and_params(n_enemies=1, map_h=30, map_w=30)
    bank = FakeBank(grid(30, 30))
    for loot in ("crate", "cube"):
        state = fresh_state(cfg, params, person=Person.HUNTER)
        state.ent_pos[0, 0] = torch.tensor([22.0, 15.0])  # the hero 10 tiles east, in sight
        state.ent_pos[0, 1] = torch.tensor([12.0, 15.0])
        state.ent_hp[0, 1] = 0.2 * state.ent_max_hp[0, 1]
        if loot == "crate":
            state.box_alive[0, 0] = True
            state.box_pos[0, 0] = torch.tensor([14.0, 15.0])
        else:
            state.pku_alive[0, 0] = True
            state.pku_pos[0, 0] = torch.tensor([14.0, 15.0])
        _vis, tgt = build_targeting(state, bank, params, cfg)
        assert bool(tgt.has_enemy[0, 1])
        move_dir, mode = personality.movement(state, tgt, bank, params, cfg, gen)
        assert Mode(int(mode[0, 1])) == Mode.RETREAT, loot
        assert move_dir[0, 1, 0].item() < -0.5, (loot, move_dir[0, 1])  # west, away from both


# =============================================================================================
# aggression: three consumers, every threshold pinned as a literal
# =============================================================================================

def _hunter_at_hp_fraction(hp_fraction, aggression, person=Person.HUNTER):
    """The retreat scene from test_hunter_retreats_at_low_hp with the sniper kind's `aggression`
    column set by hand and the bot's HP at a literal fraction of max."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    params.aggression[:, int(Kind.BOT_SNIPER)] = aggression
    state = fresh_state(cfg, params, person=person)
    bank = FakeBank(grid(20, 20))
    state.ent_pos[0, 0] = torch.tensor([8.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([12.0, 10.0])
    state.ent_hp[0, 1] = hp_fraction * state.ent_max_hp[0, 1]
    return _mode_of(state, bank, params, cfg, gen)


def test_hunter_at_25_percent_hp_closes_at_aggression_1_7_and_retreats_at_1_0():
    """Retreat threshold = clamp(0.35 / a, 0.05, 0.90). At a = 1.7 that is 0.206, so 25% HP is
    above it (CLOSE); at a = 1.0 it is the plain 0.35 (RETREAT)."""
    assert _hunter_at_hp_fraction(0.25, 1.7) is Mode.CLOSE
    assert _hunter_at_hp_fraction(0.25, 1.0) is Mode.RETREAT


def test_aggression_zero_is_read_as_one_for_the_retreat_threshold():
    """A partial spec without the key resolves to 0; the helper reads it as 1.0, so the
    threshold is 0.35: 25% retreats, 40% closes -- identical to a = 1.0."""
    assert _hunter_at_hp_fraction(0.25, 0.0) is Mode.RETREAT
    assert _hunter_at_hp_fraction(0.40, 0.0) is Mode.CLOSE
    assert _hunter_at_hp_fraction(0.40, 1.0) is Mode.CLOSE


def test_timid_hunter_retreats_earlier():
    """a = 0.6 -> 0.35 / 0.6 = 0.583: 50% HP retreats, where a = 1.0 would still close."""
    assert _hunter_at_hp_fraction(0.50, 0.6) is Mode.RETREAT
    assert _hunter_at_hp_fraction(0.50, 1.0) is Mode.CLOSE
    assert _hunter_at_hp_fraction(0.60, 0.6) is Mode.CLOSE


def test_retreat_threshold_is_clamped_to_5_and_90_percent():
    """a = 10 would give 0.035 but the floor is 0.05: 4% retreats, 6% closes. a = 0.1 would give
    3.5 but the ceiling is 0.90: 89% retreats, 91% closes."""
    assert _hunter_at_hp_fraction(0.04, 10.0) is Mode.RETREAT
    assert _hunter_at_hp_fraction(0.06, 10.0) is Mode.CLOSE
    assert _hunter_at_hp_fraction(0.89, 0.1) is Mode.RETREAT
    assert _hunter_at_hp_fraction(0.91, 0.1) is Mode.CLOSE


def test_kite_shares_the_scaled_retreat_threshold():
    """KITE retreats on the same rule as HUNTER: 25% HP holds range at a = 1.7, retreats at 1.0."""
    assert _hunter_at_hp_fraction(0.25, 1.7, person=Person.KITE) is Mode.HOLD_RANGE
    assert _hunter_at_hp_fraction(0.25, 1.0, person=Person.KITE) is Mode.RETREAT


def test_rush_still_never_retreats_whatever_the_aggression():
    """Aggression only moves the threshold for the two personalities that HAVE one."""
    assert _hunter_at_hp_fraction(0.05, 0.1, person=Person.RUSH) is Mode.CLOSE


def _unseen_camper_scene():
    """The first half of test_camper_does_not_fire_until_something_can_see_it: camper in the
    bush, hero outside bush_reveal_radius, so the camper has a target but nobody can see it."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.CAMPER)
    bank = FakeBank(_one_bush_grid())
    bush_reveal = params.bush_reveal_radius[0].item()
    state.ent_pos[0, 1] = BUSH_E.clone()
    state.ent_pos[0, 0] = torch.tensor([13.5 - (bush_reveal + 2.0), 10.5])
    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert bool(tgt.has_enemy[0, 1])
    assert not bool(tgt.seen_by_other[0, 1])
    return state, tgt, params, cfg


def test_unseen_camper_fires_at_aggression_1_5_and_holds_at_1_0():
    """The CAMPER veto is `~seen_by_other & a < 1.25`. Literal (N,E) aggression tensors, so the
    pin does not go through the helper."""
    state, tgt, _params, cfg = _unseen_camper_scene()
    shape = state.ent_person.shape
    assert bool(personality.fire_allowed(state, tgt, torch.full(shape, 1.5), cfg)[0, 1])
    assert not bool(personality.fire_allowed(state, tgt, torch.full(shape, 1.0), cfg)[0, 1])


def test_camper_fire_on_sight_boundary_is_inclusive_at_1_25():
    state, tgt, _params, cfg = _unseen_camper_scene()
    shape = state.ent_person.shape
    assert bool(personality.fire_allowed(state, tgt, torch.full(shape, 1.25), cfg)[0, 1])
    assert not bool(personality.fire_allowed(state, tgt, torch.full(shape, 1.24), cfg)[0, 1])


def test_camper_veto_reads_a_missing_aggression_as_one_through_the_helper():
    """params.aggression = 0 for the kind (a partial spec) must behave as 1.0: still silent."""
    state, tgt, params, cfg = _unseen_camper_scene()
    params.aggression[:, int(Kind.BOT_SNIPER)] = 0.0
    a = stats.aggression_of(state.ent_kind, params)
    assert a[0, 1].item() == 1.0
    assert not bool(personality.fire_allowed(state, tgt, a, cfg)[0, 1])
    params.aggression[:, int(Kind.BOT_SNIPER)] = 1.5
    a = stats.aggression_of(state.ent_kind, params)
    assert a[0, 1].item() == 1.5
    assert bool(personality.fire_allowed(state, tgt, a, cfg)[0, 1])


def test_aggression_does_not_lift_the_camper_veto_for_a_non_camper_or_a_seen_camper():
    """The only thing aggression changes in fire_allowed is the UNSEEN camper; everyone else was
    already allowed to fire and stays that way at any aggression."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.CAMPER)
    bank = FakeBank(_one_bush_grid())
    bush_reveal = params.bush_reveal_radius[0].item()
    state.ent_pos[0, 1] = BUSH_E.clone()
    state.ent_pos[0, 0] = torch.tensor([13.5 - (bush_reveal - 0.5), 10.5])  # inside reveal radius
    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert bool(tgt.seen_by_other[0, 1])
    shape = state.ent_person.shape
    assert bool(personality.fire_allowed(state, tgt, torch.full(shape, 0.5), cfg)[0, 1])
    state.ent_person.fill_(int(Person.TRAPPER))
    assert bool(personality.fire_allowed(state, tgt, torch.full(shape, 0.5), cfg)[0, 1])


def test_kite_holds_closer_at_high_aggression_and_farther_at_low():
    """Through the movement layer, with RANGE_DEADBAND 1.5 and the far edge capped at Brock's
    8.0-tile reach. Brock (attack_range 8.0, desired_range_fraction 0.85) holds 6.8 tiles at
    a = 1.0 (band 5.3 to 8.0). At a = 1.7 the multiplier clamps to 0.6 (4.08, band 2.58 to 5.58),
    so a bot standing at 6.0 is too far out and closes in (west). At a = 1.0 the same bot orbits.
    At a = 0.6 it clamps to 1.4 (9.52, capped to 8.0, band 6.5 to 8.0), so it is too close and
    backs off (east)."""
    for aggression, want_sign in ((1.7, -1.0), (1.0, 0.0), (0.6, 1.0)):
        cfg, params, gen = cfg_and_params(n_enemies=1, map_h=40, map_w=40)
        params.aggression[:, int(Kind.BOT_SNIPER)] = aggression
        state = fresh_state(cfg, params, person=Person.KITE)
        bank = FakeBank(grid(40, 40))
        state.ent_pos[0, 0] = torch.tensor([5.0, 20.0])
        state.ent_pos[0, 1] = torch.tensor([5.0 + 6.0, 20.0])
        move, mode = _move(state, bank, params, cfg, gen)
        assert Mode(int(mode[0, 1])) is Mode.HOLD_RANGE
        if want_sign == 0.0:
            assert abs(move[0, 1, 0].item()) < 1e-6, aggression   # strafe only: no radial part
        else:
            assert move[0, 1, 0].item() * want_sign > 0, aggression


def test_kite_walks_in_to_its_own_fire_reach_rather_than_parking_outside_it():
    """The far edge of the KITE band, not its centre, is where an approaching kiter stops: inside
    the band only strafe is left, and an orbit only drifts outward. Uncapped, that edge was hold +
    1.5, i.e. 8.3 tiles for a hard Brock with an 8.0-tile rocket and 11.02 for an easy one, so a
    kiter walking in parked where it could never fire (operator, 2026-09-21: cap it at the range).
    Capped at the fire reach, a Brock at 8.2 tiles closes at every tier from easy to hard and at
    7.9 orbits. Shelly's reach is her 0.9 fire fraction of 8.0, so at hard she closes from 7.4,
    which sat inside her old 4.5 to 7.5 band, and orbits at 7.1."""
    cases = (
        (Kind.BOT_SNIPER, 0.6, 8.2, -1.0),
        (Kind.BOT_SNIPER, 0.8, 8.2, -1.0),
        (Kind.BOT_SNIPER, 1.0, 8.2, -1.0),
        (Kind.BOT_SNIPER, 0.6, 7.9, 0.0),
        (Kind.BOT_SNIPER, 1.0, 7.9, 0.0),
        (Kind.BOT_RIFLE, 1.0, 7.4, -1.0),
        (Kind.BOT_RIFLE, 1.0, 7.1, 0.0),
    )
    for kind, aggression, dist, want_sign in cases:
        cfg, params, gen = cfg_and_params(n_enemies=1, map_h=40, map_w=40)
        params.aggression[:, int(kind)] = aggression
        state = fresh_state(cfg, params, enemy_kind=kind, person=Person.KITE)
        bank = FakeBank(grid(40, 40))
        state.ent_pos[0, 0] = torch.tensor([5.0, 20.0])
        state.ent_pos[0, 1] = torch.tensor([5.0 + dist, 20.0])
        move, mode = _move(state, bank, params, cfg, gen)
        case = (kind.name, aggression, dist)
        assert Mode(int(mode[0, 1])) is Mode.HOLD_RANGE, case
        if want_sign == 0.0:
            assert abs(move[0, 1, 0].item()) < 1e-6, case
        else:
            assert move[0, 1, 0].item() * want_sign > 0, case


# =============================================================================================
# pathfinding (bots.nav) and the endgame (bots.endgame_players), SIM_ISSUES_PLAN.md §1 and §2
# =============================================================================================

def _walk_bot(state, bank, params, cfg, gen, ticks, until):
    """Up to `ticks` ticks of targeting -> personality.movement -> core/movement.apply_movement,
    the hero standing still, stopping once `until(state)` holds. Returns bot 1's positions, a row
    per tick. Raw steering, not the dispatcher's smoothed intent: the route is what is tested."""
    track = []
    for _ in range(ticks):
        move, _mode = _move(state, bank, params, cfg, gen)
        move[:, 0] = 0.0
        apply_movement(state, move, bank, params, cfg)
        track.append(state.ent_pos[0, 1].clone())
        if until(state):
            break
    return torch.stack(track)


def _pocket_grid():
    """30x30: a U of wall opening west, its back along x=12 (y 8-22) and its arms along y=8 and
    y=22 (x 6-12). A bot inside at (9.5, 15.5) has the hero due east, through the back wall."""
    tiles = grid(30, 30)
    tiles[8:23, 12] = Tile.WALL
    tiles[8, 6:13] = Tile.WALL
    tiles[22, 6:13] = Tile.WALL
    return tiles


def test_a_bot_in_a_walled_pocket_reaches_the_hero_only_under_nav():
    """The watch.py failure: a bot whose enemy stood behind a wall pushed straight at it and
    stayed pinned. Steering alone still does; the nav flow field walks it out of the pocket and
    round. Sight is unlimited so the detour cannot drop the target."""
    hero = torch.tensor([20.5, 15.5])
    for nav_on in (False, True):
        cfg, params, gen = cfg_and_params(n_enemies=1, map_h=30, map_w=30,
                                          overrides={"bots": {"nav": nav_on, "sight_tiles": 0.0}})
        state = fresh_state(cfg, params, person=Person.RUSH)
        bank = FakeBank(_pocket_grid(), nav_cfg=cfg if nav_on else None)
        state.ent_pos[0, 0] = hero
        state.ent_pos[0, 1] = torch.tensor([9.5, 15.5])

        track = _walk_bot(state, bank, params, cfg, gen, 600,
                          until=lambda s: float((s.ent_pos[0, 1] - hero).norm()) < 2.0)
        if nav_on:
            assert float((track[-1] - hero).norm()) < 2.0
        else:
            assert len(track) == 600
            assert float(track[:, 0].max()) < 12.0   # never left the pocket


def test_a_bot_in_the_gas_behind_a_wall_walks_round_it_only_under_nav():
    """The escape aims at the nearest safe point, and a wall across that line held the bot in the
    gas. Under nav it follows the centre field (core/zone.py closes on the map centre) round the
    end of the wall and into the safe rect."""
    zone_lo, zone_hi = torch.tensor([8.0, 8.0]), torch.tensor([22.0, 22.0])  # centred on 30x30

    def inside(s):
        return bool(((s.ent_pos[0, 1] >= zone_lo) & (s.ent_pos[0, 1] <= zone_hi)).all())

    for nav_on in (False, True):
        cfg, params, gen = cfg_and_params(n_enemies=1, map_h=30, map_w=30,
                                          overrides={"bots": {"nav": nav_on}, "zone": {"enabled": True}})
        state = fresh_state(cfg, params, person=Person.RUSH)
        tiles = grid(30, 30)
        tiles[3:27, 6] = Tile.WALL   # between the bot and the rect, open past both ends
        bank = FakeBank(tiles, nav_cfg=cfg if nav_on else None)
        state.ent_alive[0, 0] = False
        state.ent_pos[0, 1] = torch.tensor([3.5, 15.5])
        state.zone_lo[0] = zone_lo
        state.zone_hi[0] = zone_hi

        track = _walk_bot(state, bank, params, cfg, gen, 600, until=inside)
        if nav_on:
            assert inside(state)
        else:
            assert float(track[:, 0].max()) < 6.0   # pinned to the wall, in the gas


def _corner_grid():
    """20x20 with wall tiles at x=9, y 9-10: a bot at x = 10.0 walking due north has a clear
    centre line (column 10), but its body's west edge (x = 9.6) runs into them."""
    tiles = grid(20, 20)
    tiles[9:11, 9] = Tile.WALL
    return tiles


def test_a_bot_closing_past_a_wall_corner_walks_round_it_under_nav():
    """2026-10-07, the last zone death in the probe: a bot closing on an enemy due north, its
    centre line clear past a wall corner its body was not. The straight hand-over pushed it along
    the wall face, which leaves terrain.resolve_move no axis to slide on, and it stood in the gas
    until it died. Under nav the hand-over waits for the body's walk (`_walk_clear`), so the bot
    follows the flow field off the corner first. Without nav steering still pins it."""
    hero = torch.tensor([10.0, 5.5])
    for nav_on in (False, True):
        cfg, params, gen = cfg_and_params(overrides={"bots": {"nav": nav_on, "sight_tiles": 0.0}})
        state = fresh_state(cfg, params, enemy_kind=Kind.BOT_MELEE, person=Person.RUSH)
        bank = FakeBank(_corner_grid(), nav_cfg=cfg if nav_on else None)
        state.ent_pos[0, 0] = hero
        state.ent_pos[0, 1] = torch.tensor([10.0, 12.4])
        goal = hero.expand_as(state.ent_pos)
        # The case under test: the centre line is clear (nav off), the body's walk is not.
        assert bool(policy._walk_clear(state, bank, params, cfg, goal, 8.0)[0, 1]) is not nav_on
        assert _mode_of(state, bank, params, cfg, gen) is Mode.CLOSE

        track = _walk_bot(state, bank, params, cfg, gen, 400,
                          until=lambda s: float((s.ent_pos[0, 1] - hero).norm()) < 2.0)
        if nav_on:
            assert float((track[-1] - hero).norm()) < 2.0
        else:
            assert len(track) == 400
            assert float(track[:, 1].min()) > 11.0   # pinned below the corner


def test_a_crate_pull_waits_for_a_walk_the_body_can_make_under_nav():
    """The 26 s pin of 2026-10-07: a crate pull replaces the bot's steering, and its gate tested
    the centre line, which passed a water corner the body could not. Under nav the gate is the
    body's walk, so the pull stays off and the bot's own steering (on its path) moves it; without
    nav the gate is still the centre line."""
    for nav_on, expected_w in ((False, policy._BOX_APPROACH_WEIGHT), (True, 0.0)):
        cfg, params, gen = cfg_and_params(n_enemies=1, overrides={"bots": {"nav": nav_on}})
        tiles = grid(20, 20)
        tiles[9, 11] = Tile.WATER   # the body's north edge (y = 9.6) crosses it, the centre line not
        bank = FakeBank(tiles, nav_cfg=cfg if nav_on else None)
        state = fresh_state(cfg, params, person=Person.RUSH)
        state.ent_alive[0, 0] = False
        state.ent_pos[0, 1] = torch.tensor([10.0, 10.0])
        state.box_alive[0, 0] = True
        state.box_pos[0, 0] = torch.tensor([13.0, 10.0])
        build_targeting(state, bank, params, cfg)
        _direction, w = policy.box_contribution(state, bank, params, cfg)
        assert w[0, 1].item() == expected_w, nav_on


def test_a_bot_strafing_past_a_water_corner_straightens_up_and_closes_under_nav():
    """The 20.6 s pin of 2026-10-07: a RUSH bot closing on the hero due west, its body overhanging
    the row of a water tile beside its next step, its strafe (south, for entity 1) cancelling the
    small northward part of its aim. Under nav it straightens up to its tile's centre row first
    (nav.step_dir) and closes; without nav it stays pinned against the water's corner."""
    hero = torch.tensor([8.8, 10.6])
    for nav_on in (False, True):
        cfg, params, gen = cfg_and_params(overrides={"bots": {"nav": nav_on}})
        state = fresh_state(cfg, params, enemy_kind=Kind.BOT_MELEE, person=Person.RUSH)
        tiles = grid(20, 20)
        tiles[11, 9] = Tile.WATER
        bank = FakeBank(tiles, nav_cfg=cfg if nav_on else None)
        state.ent_pos[0, 0] = hero
        state.ent_pos[0, 1] = torch.tensor([10.11, 10.67])
        assert _mode_of(state, bank, params, cfg, gen) is Mode.CLOSE

        track = _walk_bot(state, bank, params, cfg, gen, 200,
                          until=lambda s: float((s.ent_pos[0, 1] - hero).norm()) < 1.0)
        if nav_on:
            assert float((track[-1] - hero).norm()) < 1.0
        else:
            assert len(track) == 200
            assert float(track[:, 0].min()) > 9.9   # pinned east of the water's corner


def _edge_pocket_grid():
    """30x30: a U of wall at the west edge of the safe rect (8, 8)-(22, 22), its back along x=11
    (y 12-18) and its arms along y=12 and y=18 (x 8-11), open west into the gas. From inside, the
    centre field's way to the map centre runs out through the opening, at x=7."""
    tiles = grid(30, 30)
    tiles[12:19, 11] = Tile.WALL
    tiles[12, 8:12] = Tile.WALL
    tiles[18, 8:12] = Tile.WALL
    return tiles


def test_zone_avoidance_never_leads_a_bot_out_of_a_pocket_into_the_gas_under_nav(monkeypatch):
    """The first CAMPER zone death of 2026-10-07: in a pocket at the safe rect's edge the centre
    field leads out through the gas, the avoid push followed it there, and in the gas the escape
    (scaled by depth) lost to the bot's pull back in, so it crossed the edge until the gas killed
    it. In a pocket (nav.centre_path_inside false) there is no push now: a bot with no enemy
    wanders inside it, its wander probe keeping it out of the gas. With that test forced true, the
    push follows the field and leads it out."""
    def run():
        cfg, params, gen = cfg_and_params(n_enemies=1, map_h=30, map_w=30,
                                          overrides={"bots": {"nav": True}, "zone": {"enabled": True}})
        state = fresh_state(cfg, params, person=Person.RUSH)
        bank = FakeBank(_edge_pocket_grid(), nav_cfg=cfg)
        state.ent_alive[0, 0] = False
        state.ent_pos[0, 1] = torch.tensor([9.5, 15.5])
        state.zone_lo[0] = torch.tensor([8.0, 8.0])
        state.zone_hi[0] = torch.tensor([22.0, 22.0])
        direction, weight = policy.zone_avoid_contribution(state, cfg, bank, params)
        track = _walk_bot(state, bank, params, cfg, gen, 200, until=lambda s: False)
        return direction[0, 1], weight[0, 1].item(), track

    _direction, weight, track = run()
    assert weight == 0.0
    assert float(track[:, 0].min()) >= 8.0            # never in the gas

    monkeypatch.setattr(nav, "centre_path_inside",
                        lambda bank, map_id, pos, *_: torch.ones(pos.shape[:-1], dtype=torch.bool))
    direction, weight, track = run()
    assert weight > 0.0
    assert direction[0].item() < 0.0                  # along the field, west
    assert float(track[:, 0].min()) < 8.0


def _gas_corner_grid():
    """60x60 with bramble_bend's corner: wall along x=11 (y 10-14) and along y=15 (x 8-11). With
    the safe rect at (10, 10)-(50, 50), every way from the corner to the map centre runs through
    the gas, north round (11, 9) or west round (7, 15)."""
    tiles = grid(60, 60)
    tiles[10:15, 11] = Tile.WALL
    tiles[15, 8:12] = Tile.WALL
    return tiles


def test_a_bot_in_a_gas_edge_corner_is_not_held_there_by_zone_avoidance_under_nav(monkeypatch):
    """The second CAMPER zone death of 2026-10-07, on bramble_bend: where the centre field's path
    left the rect, the avoid push aimed straight at the rect's centre instead, and at about 40 on a
    60x60 map it beat the bot's seek (about 12) and held it in this corner for 7 s, until the gas
    arrived. With no push in a pocket, a bot closing on an enemy 4 tiles north reaches it; with
    the straight push back, it stays in the corner."""
    hero = torch.tensor([10.5, 10.6])

    def run():
        cfg, params, gen = cfg_and_params(n_enemies=1, map_h=60, map_w=60,
                                          overrides={"bots": {"nav": True, "sight_tiles": 0.0},
                                                     "zone": {"enabled": True}})
        state = fresh_state(cfg, params, enemy_kind=Kind.BOT_MELEE, person=Person.RUSH)
        bank = FakeBank(_gas_corner_grid(), nav_cfg=cfg)
        state.ent_pos[0, 0] = hero
        state.ent_pos[0, 1] = torch.tensor([10.55, 14.55])
        state.zone_lo[0] = torch.tensor([10.0, 10.0])
        state.zone_hi[0] = torch.tensor([50.0, 50.0])
        assert _mode_of(state, bank, params, cfg, gen) is Mode.CLOSE
        _direction, weight = policy.zone_avoid_contribution(state, cfg, bank, params)
        track = _walk_bot(state, bank, params, cfg, gen, 200,
                          until=lambda s: float((s.ent_pos[0, 1] - hero).norm()) < 1.0)
        return weight[0, 1].item(), track

    weight, track = run()
    assert weight == 0.0
    assert float((track[-1] - hero).norm()) < 1.0

    straight = policy.zone_avoid_contribution
    monkeypatch.setattr(policy, "zone_avoid_contribution",
                        lambda state, cfg, bank=None, params=None: straight(state, cfg))
    weight, track = run()
    assert weight > 1.5
    assert len(track) == 200
    assert float(track[:, 1].min()) > 14.0            # held in the corner


def test_kite_without_a_line_of_sight_closes_under_nav():
    """A kiter holding range behind a wall never had a shot. Under nav it closes along its path
    until it has a line; with one it holds range as before, and without nav it holds either way."""
    for nav_on, walled, want in ((False, True, Mode.HOLD_RANGE),
                                 (True, True, Mode.CLOSE),
                                 (True, False, Mode.HOLD_RANGE)):
        cfg, params, gen = cfg_and_params(n_enemies=1, map_h=40, map_w=40,
                                          overrides={"bots": {"nav": nav_on}})
        state = fresh_state(cfg, params, person=Person.KITE)
        tiles = grid(40, 40)
        if walled:
            tiles[17:24, 8] = Tile.WALL   # across the line between them
        bank = FakeBank(tiles, nav_cfg=cfg if nav_on else None)
        attack_range = params.attack_range[0, int(Kind.BOT_SNIPER)].item()
        desired = params.desired_range_fraction[0, int(Kind.BOT_SNIPER)].item() * attack_range
        state.ent_pos[0, 0] = torch.tensor([5.0, 20.0])
        state.ent_pos[0, 1] = torch.tensor([5.0 + desired, 20.0])   # in the band: holds if it can

        _vis, tgt = build_targeting(state, bank, params, cfg)
        assert bool(tgt.enemy_los[0, 1]) is not walled
        assert _mode_of(state, bank, params, cfg, gen) is want, (nav_on, walled)


def test_without_nav_path_toward_is_the_straight_offset_and_the_zone_terms_ignore_the_bank():
    cfg, params, gen = cfg_and_params(n_enemies=1, overrides={"zone": {"enabled": True}})
    state = fresh_state(cfg, params)
    bank = FakeBank(_one_bush_grid())
    state.ent_pos[0, 0] = torch.tensor([1.5, 3.5])
    state.ent_pos[0, 1] = torch.tensor([10.5, 17.5])
    state.zone_lo[0] = torch.tensor([4.0, 4.0])
    state.zone_hi[0] = torch.tensor([16.0, 16.0])
    goal = torch.tensor([[[12.0, 3.0], [2.0, 9.0]]])
    assert torch.equal(policy.path_toward(state, bank, params, cfg, goal), goal - state.ent_pos)
    for term in (policy.zone_contribution, policy.zone_avoid_contribution):
        for with_bank, without in zip(term(state, cfg, bank), term(state, cfg)):
            assert torch.equal(with_bank, without)
    assert torch.equal(policy.zone_clearance(state, cfg, bank), policy.zone_clearance(state, cfg))


def test_under_nav_a_goal_no_path_reaches_is_aimed_at_straight():
    """path_toward's last rule: where both straight walks are blocked, the field gives no step and
    the bot is not on its anchor's tile, no path leads to the goal (here a sealed box, the goal
    either side of its wall), and the bot pushes straight at it as it did before nav, rather than
    standing on its tile's centre as a bot on its anchor's tile does."""
    cfg, params, gen = cfg_and_params(n_enemies=1, map_h=20, map_w=20, overrides={"bots": {"nav": True}})
    tiles = grid(20, 20)
    tiles[12, 12:19] = Tile.WALL
    tiles[12:19, 12] = Tile.WALL            # with the map's own walls, a box sealed round 13-18
    state = fresh_state(cfg, params)
    bank = FakeBank(tiles, nav_cfg=cfg)
    state.ent_pos[0, 0] = torch.tensor([10.5, 15.5])
    state.ent_pos[0, 1] = torch.tensor([15.5, 15.5])
    goal = torch.tensor([[[15.7, 16.2], [9.7, 14.6]]])     # each 5-6 tiles away, through the wall
    assert torch.equal(policy.path_toward(state, bank, params, cfg, goal), goal - state.ent_pos)


def _exit_pocket_grid():
    """30x30, for the safe rect (8, 8)-(22, 22): a U of wall opening west, its back along x=14
    (y 12-19) and its arms along y=12 and y=19 (x 10-14). A bush at (12, 15) deep inside, 4.5
    tiles from the gas, whose way to the map centre runs out through the opening and round an arm,
    as near as x=9.5, 1.5 from the gas; a second bush at (18, 15), behind the back wall."""
    tiles = grid(30, 30)
    tiles[12:20, 14] = Tile.WALL
    tiles[12, 10:15] = Tile.WALL
    tiles[19, 10:15] = Tile.WALL
    tiles[15, 12] = Tile.BUSH
    tiles[15, 18] = Tile.BUSH
    return tiles


def _exit_pocket_scene(nav_on):
    """A CAMPER sitting in the pocket's bush, the hero dead, the rect at (8, 8)-(22, 22)."""
    cfg, params, gen = cfg_and_params(n_enemies=1, map_h=30, map_w=30,
                                      overrides={"bots": {"nav": nav_on}, "zone": {"enabled": True}})
    state = fresh_state(cfg, params, person=Person.CAMPER)
    bank = FakeBank(_exit_pocket_grid(), nav_cfg=cfg if nav_on else None)
    state.ent_alive[0, 0] = False
    state.ent_pos[0, 1] = torch.tensor([12.5, 15.5])
    state.zone_lo[0] = torch.tensor([8.0, 8.0])
    state.zone_hi[0] = torch.tensor([22.0, 22.0])
    return cfg, params, gen, state, bank


def test_a_camper_whose_way_out_nears_the_gas_leaves_its_bush_only_under_nav():
    """User decision, 2026-10-07 (#27), from the bramble_bend death: a CAMPER 2.1 tiles inside
    the edge held its bush while its only exit was already gas. Here its own tile has 4.5 tiles of
    room but its way out passes 1.5 from the gas, and under nav that is its clearance: under the
    2.0 flee threshold, so it leaves while the exit is open, the avoid push (ramped on the same
    1.5 against the 4-tile margin: 2.0 x (1 - 1.5 / 4)) walking it out along that way, round to
    the bush behind the back wall and never into the gas. Without nav it holds."""
    for nav_on, clearance, push in ((False, 4.5, 0.0), (True, 1.5, 1.25)):
        cfg, params, gen, state, bank = _exit_pocket_scene(nav_on)
        assert policy.zone_clearance(state, cfg, bank)[0, 1].item() == clearance
        assert policy.zone_avoid_contribution(state, cfg, bank, params)[1][0, 1].item() == push
        assert (_mode_of(state, bank, params, cfg, gen) is Mode.HOLD_STILL) is not nav_on

        track = _walk_bot(state, bank, params, cfg, gen, 400,
                          until=lambda s: float(s.ent_pos[0, 1, 0]) > 15.0
                          and bool(perception.in_bush(s, bank)[0, 1]))
        if nav_on:
            assert len(track) < 400
            assert float(track[:, 0].min()) >= 8.0      # never in the gas
        else:
            assert len(track) == 400
            assert torch.equal(track[-1], torch.tensor([12.5, 15.5]))


def test_a_bush_or_waypoint_whose_way_out_nears_the_gas_is_never_chosen_under_nav():
    """The same rule for a destination: the pocket's bush is 4.5 tiles from the gas but its way
    out 1.5, so under nav neither bush_scan nor hunt_waypoint offers it, and a hunter is sent to
    the waypoint behind the back wall instead. Without nav both offer it."""
    for nav_on in (False, True):
        cfg, _params, _gen, state, bank = _exit_pocket_scene(nav_on)
        bank.bush_wp = torch.tensor([[[12.5, 15.5], [18.5, 15.5]]])
        bank.n_bush_wp = torch.tensor([2], dtype=torch.int64)
        state.ent_pos[0, 1] = torch.tensor([11.5, 13.5])   # in the pocket, off the bush
        zone_lo, zone_hi, _ = policy.zone_rect(state)
        margin = cfg.bots_camper_zone_flee_tiles
        scan = perception.bush_scan(state.ent_pos, state.map_id, bank, cfg,
                                    zone_lo=zone_lo, zone_hi=zone_hi, zone_margin=margin)
        hunt = perception.hunt_waypoint(state.ent_pos, state.map_id, bank, cfg,
                                        state.ent_hunt_seen, zone_lo=zone_lo, zone_hi=zone_hi,
                                        zone_margin=margin)
        assert bool(scan.found[0, 1]) is not nav_on
        assert hunt.pos[0, 1].tolist() == ([18.5, 15.5] if nav_on else [12.5, 15.5])


def _back_pocket_grid():
    """30x30: a U of wall opening east, its back along x=10 (y 13-18) and its arms along y=13
    and y=18 (x 10-13). A bot inside at (11.5, 15.5) has its enemy due east, past the opening,
    and the wall at its back."""
    tiles = grid(30, 30)
    tiles[13:19, 10] = Tile.WALL
    tiles[13, 10:14] = Tile.WALL
    tiles[18, 10:14] = Tile.WALL
    return tiles


def test_a_bot_backing_off_with_its_back_in_a_pocket_walks_out_only_under_nav():
    """User decision, 2026-10-07 (#28): a bot backing off, a HUNTER in RETREAT or a KITE too close
    in HOLD_RANGE, fled straight away from its enemy and pinned itself in the back of the pocket.
    Under nav it heads for bots/policy.retreat_goal, six tiles past itself, along the path: out of
    the pocket and round an arm. The hunter gets clear away; the kiter reaches its band (5.3
    tiles, 6.8 - 1.5 for the sniper), which pinned it never does."""
    enemy = torch.tensor([16.5, 15.5])
    for person, hp_fraction, backing, far in ((Person.HUNTER, 0.1, Mode.RETREAT, 10.0),
                                             (Person.KITE, 1.0, Mode.HOLD_RANGE, 5.3)):
        for nav_on in (False, True):
            cfg, params, gen = cfg_and_params(n_enemies=1, map_h=30, map_w=30,
                                              overrides={"bots": {"nav": nav_on, "sight_tiles": 0.0}})
            state = fresh_state(cfg, params, person=person)
            bank = FakeBank(_back_pocket_grid(), nav_cfg=cfg if nav_on else None)
            state.ent_pos[0, 0] = enemy
            state.ent_pos[0, 1] = torch.tensor([11.5, 15.5])
            state.ent_hp[0, 1] = hp_fraction * state.ent_max_hp[0, 1]
            assert _mode_of(state, bank, params, cfg, gen) is backing

            track = _walk_bot(state, bank, params, cfg, gen, 400, until=lambda s: False)
            left = bool(((track[:, 0] < 10.0) | (track[:, 0] > 14.0)).any())
            assert left is nav_on, (person, nav_on)
            assert (float((track - enemy).norm(dim=-1).max()) >= far) is nav_on, (person, nav_on)


def test_backing_off_on_open_ground_steers_as_it_did_without_nav():
    """Where the straight walk to the retreat goal is clear, the path is that line, and the flee
    keeps the old offset's length, so a HUNTER's retreat and a KITE's back-off steer as before to
    float rounding."""
    for person, hp_fraction, enemy_at in ((Person.HUNTER, 0.1, [24.0, 21.0]),
                                          (Person.KITE, 1.0, [23.0, 20.0])):
        moves = []
        for nav_on in (False, True):
            cfg, params, gen = cfg_and_params(n_enemies=1, map_h=40, map_w=40,
                                              overrides={"bots": {"nav": nav_on}})
            state = fresh_state(cfg, params, person=person)
            bank = FakeBank(grid(40, 40), nav_cfg=cfg if nav_on else None)
            state.ent_pos[0, 0] = torch.tensor(enemy_at)
            state.ent_pos[0, 1] = torch.tensor([20.0, 19.0])
            state.ent_hp[0, 1] = hp_fraction * state.ent_max_hp[0, 1]
            move, mode = _move(state, bank, params, cfg, gen)
            assert Mode(int(mode[0, 1])) in (Mode.RETREAT, Mode.HOLD_RANGE)
            moves.append(move[0, 1])
        assert torch.allclose(moves[0], moves[1], atol=1e-6), person


def test_the_retreat_goal_is_six_tiles_away_inside_the_safe_rect_and_on_the_map():
    """bots/policy.retreat_goal, as literals: six tiles past the bot straight away from the
    threat; with the gas on, one tile inside an active rect, or half the width of a rect narrower
    than two (its centre line); never past the centres of the map's edge tiles; and no rect clamp
    while the rect is the zero-area "no zone yet" one, or with the gas off."""
    def goal(zone_enabled, pos, threat, lo=(0.0, 0.0), hi=(0.0, 0.0)):
        cfg, params, _gen = cfg_and_params(n_enemies=1, map_h=40, map_w=40,
                                           overrides={"zone": {"enabled": zone_enabled}})
        state = fresh_state(cfg, params)
        state.ent_pos[0, 1] = torch.tensor(pos)
        state.zone_lo[0] = torch.tensor(lo)
        state.zone_hi[0] = torch.tensor(hi)
        out = policy.retreat_goal(state, cfg, torch.tensor(threat).expand_as(state.ent_pos))
        return [round(v, 4) for v in out[0, 1].tolist()]

    away = ([20.0, 20.0], [23.0, 24.0])          # (-0.6, -0.8) x 6 past (20, 20)
    assert goal(True, *away) == [16.4, 15.2]
    assert goal(True, *away, lo=(8.0, 8.0), hi=(32.0, 32.0)) == [16.4, 15.2]
    assert goal(True, *away, lo=(18.0, 17.0), hi=(32.0, 32.0)) == [19.0, 18.0]
    assert goal(True, *away, lo=(18.0, 8.0), hi=(19.0, 32.0)) == [18.5, 15.2]
    assert goal(False, *away, lo=(18.0, 17.0), hi=(32.0, 32.0)) == [16.4, 15.2]
    assert goal(True, [3.0, 20.0], [8.0, 20.0]) == [0.5, 20.0]
    assert goal(True, [38.0, 37.0], [34.0, 34.0]) == [39.5, 39.5]


def test_a_kite_strafing_into_a_wall_turns_round_only_under_nav():
    """User decision, 2026-10-07 (#30): inside its range band only the strafe moves a KITE, and
    its slot's fixed sense held it against a wall until its enemy moved. The wall here lies along
    the hero's own row, so slot 1's orbit meets its face almost head-on. Without nav the kiter
    pins there; under nav it turns round 1 tile short (personality.advance_strafe), orbits the
    other way and comes round to the wall's far side."""
    for nav_on in (False, True):
        cfg, params, gen = cfg_and_params(n_enemies=1, map_h=40, map_w=40,
                                          overrides={"bots": {"nav": nav_on, "sight_tiles": 0.0}})
        state = fresh_state(cfg, params, person=Person.KITE)
        tiles = grid(40, 40)
        tiles[20, 24:32] = Tile.WALL
        bank = FakeBank(tiles, nav_cfg=cfg if nav_on else None)
        state.ent_pos[0, 0] = torch.tensor([20.0, 20.0])
        state.ent_pos[0, 1] = torch.tensor([26.8, 17.0])
        assert _mode_of(state, bank, params, cfg, gen) is Mode.HOLD_RANGE

        track = _walk_bot(state, bank, params, cfg, gen, 400, until=lambda s: False)
        stalled = float(((track[1:] - track[:-1]).norm(dim=-1) < 0.5 * cfg.dt).float().mean())
        if nav_on:
            assert stalled < 0.05
            assert float(track[:, 1].max()) > 21.0   # round to the wall's far side
        else:
            assert stalled > 0.8
            assert float(track[:, 1].max()) < 20.0   # pinned on the near face
            assert not state.ent_strafe_sign.any()   # never written without nav


def test_the_strafe_sense_turns_round_at_a_wall_only_with_the_way_back_open_and_holds():
    """personality.advance_strafe, one bot at a time. Slot 1 east of its enemy strafes +y (its
    slot sense, -1, which ent_strafe_sign's 0 reads as). A wall 1 tile ahead with open ground behind
    turns it round, and the new sense holds once the wall is behind it; walls both ways, or a bot
    not strafing this tick, keep the sense it has."""
    def sense(rows, strafing=True, stored=0.0):
        cfg, params, _gen = cfg_and_params(n_enemies=1, overrides={"bots": {"nav": True}})
        state = fresh_state(cfg, params, person=Person.KITE)
        tiles = grid(20, 20)
        for row in rows:
            tiles[row, :] = Tile.WALL
        state.ent_pos[0, 0] = torch.tensor([5.0, 10.0])
        state.ent_pos[0, 1] = torch.tensor([12.0, 10.0])
        state.ent_strafe_sign[0, 1] = stored
        enemy = state.ent_pos[:, :1].expand_as(state.ent_pos)
        live = torch.full(state.ent_strafe_sign.shape, strafing)
        out = personality.advance_strafe(state, FakeBank(tiles), cfg, enemy, live)
        assert float(state.ent_strafe_sign[0, 1]) == float(out[0, 1])
        return float(out[0, 1])

    assert sense([]) == -1.0                       # the slot's sense, now stored
    assert sense([11]) == 1.0                      # wall ahead, way back open: turns round
    assert sense([11], stored=1.0) == 1.0          # already turned: the wall is behind it now
    assert sense([9], stored=1.0) == -1.0          # and turns again at the next wall
    assert sense([9, 11]) == -1.0                  # walled both ways: keeps its sense
    assert sense([11], strafing=False) == -1.0     # not strafing: never turns


def test_an_in_band_kite_on_open_ground_strafes_as_it_did_without_nav():
    """Where no wall is near, advance_strafe keeps each slot's sense, so an in-band kiter's
    first move under nav is the old one exactly."""
    moves = []
    for nav_on in (False, True):
        cfg, params, gen = cfg_and_params(n_enemies=2, map_h=40, map_w=40,
                                          overrides={"bots": {"nav": nav_on}})
        state = fresh_state(cfg, params, person=Person.KITE)
        bank = FakeBank(grid(40, 40), nav_cfg=cfg if nav_on else None)
        state.ent_pos[0, 0] = torch.tensor([20.0, 20.0])
        state.ent_pos[0, 1] = torch.tensor([26.8, 20.0])
        state.ent_pos[0, 2] = torch.tensor([20.0, 13.2])
        move, mode = _move(state, bank, params, cfg, gen)
        assert (mode[0, 1:] == int(Mode.HOLD_RANGE)).all()
        moves.append(move[0, 1:])
    assert torch.allclose(moves[0], moves[1], atol=1e-6)


def test_campers_and_trappers_play_as_hunters_once_four_players_are_left():
    """User decision, 2026-10-06: "campers should switch to hunting in final 4". The hero counts
    as a player, and 0 (default.yaml) never switches."""
    cfg, params, gen = cfg_and_params(n_enemies=4, overrides={"bots": {"endgame_players": 4}})
    state = fresh_state(cfg, params)
    persons = (Person.RUSH, Person.CAMPER, Person.TRAPPER, Person.KITE, Person.HUNTER)
    for e, person in enumerate(persons):
        state.ent_person[0, e] = int(person)
    assert torch.equal(policy.effective_person(state, cfg), state.ent_person)   # 5 alive

    state.ent_alive[0, 4] = False                                               # 4 alive
    played = [Person(int(p)) for p in policy.effective_person(state, cfg)[0]]
    assert played == [Person.RUSH, Person.HUNTER, Person.HUNTER, Person.KITE, Person.HUNTER]

    never, _params, _gen = cfg_and_params(n_enemies=4)
    assert torch.equal(policy.effective_person(state, never), state.ent_person)


def test_an_endgame_camper_leaves_its_bush_to_hunt():
    cfg, params, gen = cfg_and_params(n_enemies=4, map_h=40, map_w=40,
                                      overrides={"bots": {"endgame_players": 4}})
    state = fresh_state(cfg, params, person=Person.CAMPER)
    bank = _waypoint_bank(cfg, (13, 10), (30, 30))
    state.ent_pos[0, 1] = torch.tensor([13.5, 10.5])   # in its bush
    # Everyone else beyond the 14-tile sight: nothing to react to.
    for e, xy in ((0, (35.0, 35.0)), (2, (35.0, 5.0)), (3, (5.0, 35.0)), (4, (30.0, 25.0))):
        state.ent_pos[0, e] = torch.tensor(xy)
    assert _mode_of(state, bank, params, cfg, gen) is Mode.HOLD_STILL

    state.ent_alive[0, 4] = False
    assert _mode_of(state, bank, params, cfg, gen) is Mode.HUNT_BUSH


def test_an_endgame_camper_fires_without_being_seen():
    """fire_allowed's veto is a CAMPER's; a camper the endgame plays as a hunter fires freely."""
    for endgame_players, fires in ((0, False), (4, True)):
        cfg, params, gen = cfg_and_params(n_enemies=1,
                                          overrides={"bots": {"endgame_players": endgame_players}})
        state = fresh_state(cfg, params, person=Person.CAMPER)
        bank = FakeBank(_one_bush_grid())
        bush_reveal = params.bush_reveal_radius[0].item()
        state.ent_pos[0, 1] = BUSH_E.clone()
        state.ent_pos[0, 0] = torch.tensor([13.5 - (bush_reveal + 2.0), 10.5])

        _vis, tgt = build_targeting(state, bank, params, cfg)
        assert bool(tgt.has_enemy[0, 1]) and not bool(tgt.seen_by_other[0, 1])
        aggression = torch.ones_like(state.ent_hp)   # below the 1.25 fire-on-sight line
        assert bool(personality.fire_allowed(state, tgt, aggression, cfg)[0, 1]) is fires
