"""Bot personality tests (Step 41): one section per personality, then the rules that must hold
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
    assert not bool(personality.fire_allowed(state, tgt, cfg)[0, 1])

    # Step inside the reveal radius and the camper is exposed -- and opens fire.
    state.ent_pos[0, 0] = torch.tensor([13.5 - (bush_reveal - 0.5), 10.5])
    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert bool(tgt.seen_by_other[0, 1])
    assert bool(personality.fire_allowed(state, tgt, cfg)[0, 1])


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
    assert bool(personality.fire_allowed(state, tgt, cfg)[0, 1])


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
    assert bool(personality.fire_allowed(state, tgt, cfg)[0, 1])  # shoots anyway


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
