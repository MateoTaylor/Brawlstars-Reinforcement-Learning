import dataclasses
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import load_config, build_params
from brawl_sim.constants import Kind, Tile
from brawl_sim.core.state import allocate
from brawl_sim.bots import perception
from tests.bot_fixtures import FakeBank

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


# The shared fixture also carries blocks_unit and derives bush waypoints from `tiles` the way a
# real MapBank does -- both of which perception.bush_scan/hunt_waypoint now need.
_FakeBank = FakeBank


def _grid(h, w, fill=Tile.FLOOR):
    t = torch.full((h, w), int(fill), dtype=torch.int64)
    t[0, :] = Tile.WALL
    t[-1, :] = Tile.WALL
    t[:, 0] = Tile.WALL
    t[:, -1] = Tile.WALL
    return t


def _cfg_and_params(n_envs=1, map_h=20, map_w=20):
    cfg = load_config(CONFIGS / "default.yaml", overrides={"world": {"map_h": map_h, "map_w": map_w}})
    spec = {
        **yaml.safe_load((CONFIGS / "default.yaml").read_text()),
        **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    return cfg, params


def _fresh_state(cfg, params, n_envs=1):
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[:, 0] = int(Kind.HERO_MORTIS)
    for e in range(1, cfg.n_entities):
        state.ent_kind[:, e] = int(Kind.BOT_SNIPER)
    state.map_id.fill_(0)
    state.ent_target.fill_(-1)
    return state


# ---- in_bush -----------------------------------------------------------------

def test_in_bush_true_only_on_bush_tile():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    tiles[10, 10] = Tile.BUSH
    bank = _FakeBank(tiles)

    state.ent_pos[0, 0] = torch.tensor([10.5, 10.5])  # inside the bush tile
    state.ent_pos[0, 1] = torch.tensor([5.5, 5.5])    # open floor
    result = perception.in_bush(state, bank)
    assert bool(result[0, 0])
    assert not bool(result[0, 1])


# ---- visibility (acceptance) --------------------------------------------------

def test_bot_in_bush_five_tiles_away_not_visible_at_1_5_tiles_is():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    tiles[10, 10] = Tile.BUSH
    bank = _FakeBank(tiles)

    state.ent_pos[0, 0] = torch.tensor([5.5, 10.5])  # hero (observer)
    state.ent_pos[0, 1] = torch.tensor([10.5, 10.5])  # in bush, 5 tiles away
    bush_reveal_radius = params.bush_reveal_radius[0].item()
    assert bush_reveal_radius < 5.0  # sanity: default should make 5 tiles "hidden"

    vis = perception.visibility(state, bank, params, cfg)
    assert not bool(vis[0, 0, 1])

    state.ent_pos[0, 0] = torch.tensor([10.5 - 1.5, 10.5])  # now 1.5 tiles away
    vis = perception.visibility(state, bank, params, cfg)
    assert bool(vis[0, 0, 1])


def test_bush_hidden_entity_visible_after_recent_attack():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    tiles[10, 10] = Tile.BUSH
    bank = _FakeBank(tiles)

    state.ent_pos[0, 0] = torch.tensor([5.5, 10.5])
    state.ent_pos[0, 1] = torch.tensor([10.5, 10.5])  # 5 tiles away, in bush
    state.ent_reveal_t[0, 1] = 1.0  # fired recently

    vis = perception.visibility(state, bank, params, cfg)
    assert bool(vis[0, 0, 1])


def test_wall_never_blocks_visibility_but_blocks_raw_los():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    tiles[10, 8] = Tile.WALL  # between observer and target, neither in a bush
    bank = _FakeBank(tiles)

    state.ent_pos[0, 0] = torch.tensor([5.5, 10.5])
    state.ent_pos[0, 1] = torch.tensor([11.5, 10.5])

    vis = perception.visibility(state, bank, params, cfg)
    los = perception.raw_los(state, bank, cfg)
    assert bool(vis[0, 0, 1])   # targeting ignores the wall
    assert not bool(los[0, 0, 1])  # physical LOS is blocked


def test_target_los_matches_the_raw_los_column_it_replaced():
    """bot_overhaul.md Step A2: the bot phase swapped an (N,E,E) `raw_los` -- of which
    bots/policy.targeting read exactly one column -- for an (N,E) `target_los`. This pins that
    they agree on every row whose answer is actually consumed.

    Rows with no target (`ent_target < 0`) are excluded: `target_los` clamps the index to 0 and
    measures a meaningless ray there, which is fine precisely because `targeting` resolves those
    rows to `has_enemy=False` and `fire_gate` drops them. That exclusion is the claim, so the test
    also asserts such rows exist -- otherwise it would be silently checking nothing.
    """
    cfg, params = _cfg_and_params(n_envs=6)
    torch.manual_seed(7)

    saw_untargeted = False
    for _ in range(10):
        state = _fresh_state(cfg, params, n_envs=6)
        tiles = _grid(20, 20)
        wall = torch.rand(20, 20) < 0.20
        wall[0, :] = wall[-1, :] = wall[:, 0] = wall[:, -1] = False
        tiles[wall] = int(Tile.WALL)
        bank = _FakeBank(tiles)

        state.ent_pos.copy_(1.5 + torch.rand(6, cfg.n_entities, 2) * 17.0)
        # Kill some entities outright. A dead observer's whole `vis` row is False, so
        # select_target leaves it at -1 -- on a 20x20 map with sight_tiles=14 every LIVING entity
        # can always see someone, so this is the only reliable way to generate the untargeted
        # rows whose exclusion this test exists to check.
        state.ent_alive.copy_(torch.rand(6, cfg.n_entities) > 0.3)
        vis = perception.bot_visibility(state, perception.visibility(state, bank, params, cfg), cfg)
        perception.select_target(state, vis, cfg)

        got = perception.target_los(state, bank, cfg)
        full = perception.raw_los(state, bank, cfg)
        idx = torch.clamp(state.ent_target, min=0)
        want = torch.gather(full, 2, idx.unsqueeze(-1)).squeeze(-1)

        has_target = state.ent_target >= 0
        saw_untargeted = saw_untargeted or bool((~has_target).any())
        assert torch.equal(got[has_target], want[has_target]), (
            "target_los disagrees with the raw_los column it replaced on a row that has a target"
        )

    assert saw_untargeted, "no untargeted rows were generated; the exclusion is untested"


def test_visibility_self_true_when_alive_false_when_dead():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)
    vis = perception.visibility(state, bank, params, cfg)
    assert torch.all(torch.diagonal(vis[0]))

    state.ent_alive[0, 1] = False
    vis = perception.visibility(state, bank, params, cfg)
    assert not bool(vis[0, 1, 1])
    assert not torch.any(vis[0, 1])  # dead observer sees nothing


def test_visibility_requires_both_alive():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)
    state.ent_pos[0, 0] = torch.tensor([5.5, 5.5])
    state.ent_pos[0, 1] = torch.tensor([6.0, 5.5])
    state.ent_alive[0, 1] = False

    vis = perception.visibility(state, bank, params, cfg)
    assert not bool(vis[0, 0, 1])


# ---- bot_visibility (the sight limit) -------------------------------------------

def test_bot_visibility_clips_by_range_while_visibility_itself_does_not():
    """visibility() answers "is this entity concealed" and has no range limit by design -- the
    camera is a fixed bird's-eye view. bot_visibility() is what bots are allowed to ACT on."""
    cfg, params = _cfg_and_params(map_h=40, map_w=40)
    state = _fresh_state(cfg, params)
    bank = _FakeBank(_grid(40, 40))

    far = cfg.bots_sight_tiles + 3.0
    state.ent_pos[0, 0] = torch.tensor([5.0, 20.0])
    state.ent_pos[0, 1] = torch.tensor([5.0 + far, 20.0])

    vis = perception.visibility(state, bank, params, cfg)
    assert bool(vis[0, 1, 0])  # perception: the hero is in the open, so it is not concealed
    bot_vis = perception.bot_visibility(state, vis, cfg)
    assert not bool(bot_vis[0, 1, 0])  # but it is beyond what a bot may react to

    state.ent_pos[0, 1] = torch.tensor([5.0 + cfg.bots_sight_tiles - 1.0, 20.0])
    vis = perception.visibility(state, bank, params, cfg)
    assert bool(perception.bot_visibility(state, vis, cfg)[0, 1, 0])


def test_bot_visibility_never_widens_what_visibility_allowed():
    """A bush-concealed entity stays concealed no matter how close it is standing."""
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    tiles[10, 10] = Tile.BUSH
    bank = _FakeBank(tiles)
    bush_reveal = params.bush_reveal_radius[0].item()

    state.ent_pos[0, 1] = torch.tensor([10.5, 10.5])                       # in the bush
    state.ent_pos[0, 0] = torch.tensor([10.5 - (bush_reveal + 1.0), 10.5])  # close, but not enough
    vis = perception.visibility(state, bank, params, cfg)
    bot_vis = perception.bot_visibility(state, vis, cfg)
    assert not bool(vis[0, 0, 1]) and not bool(bot_vis[0, 0, 1])
    assert torch.all(bot_vis <= vis)  # a pure narrowing, never a widening


def test_sight_tiles_zero_disables_the_limit():
    cfg, params = _cfg_and_params(map_h=40, map_w=40)
    cfg = dataclasses.replace(cfg, bots_sight_tiles=0.0)
    state = _fresh_state(cfg, params)
    bank = _FakeBank(_grid(40, 40))
    state.ent_pos[0, 0] = torch.tensor([2.0, 20.0])
    state.ent_pos[0, 1] = torch.tensor([38.0, 20.0])  # 36 tiles apart

    vis = perception.visibility(state, bank, params, cfg)
    assert torch.equal(perception.bot_visibility(state, vis, cfg), vis)


def test_select_target_drops_a_target_that_walks_out_of_sight():
    """Stickiness must not outlive the sight limit -- otherwise the range check would apply only to
    ACQUIRING a target, and a bot could keep tracking one across the whole map."""
    cfg, params = _cfg_and_params(map_h=40, map_w=40)
    state = _fresh_state(cfg, params)
    bank = _FakeBank(_grid(40, 40))

    state.ent_pos[0, 1] = torch.tensor([20.0, 20.0])
    state.ent_pos[0, 0] = torch.tensor([25.0, 20.0])  # well within sight
    vis = perception.bot_visibility(state, perception.visibility(state, bank, params, cfg), cfg)
    perception.select_target(state, vis, cfg)
    assert state.ent_target[0, 1].item() == 0

    state.ent_pos[0, 0] = torch.tensor([20.0 + cfg.bots_sight_tiles + 5.0, 20.0])
    vis = perception.bot_visibility(state, perception.visibility(state, bank, params, cfg), cfg)
    perception.select_target(state, vis, cfg)
    assert state.ent_target[0, 1].item() == -1  # dropped, not held


# ---- team_id --------------------------------------------------------------------

def test_team_id_matches_kind():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tid = perception.team_id(state.ent_kind, cfg)
    assert torch.equal(tid, state.ent_kind)
    assert tid is not state.ent_kind  # defensive clone, not an alias


# ---- select_target (acceptance: no oscillation) -----------------------------

def test_select_target_picks_nearest_visible():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([12.0, 10.0])  # closer
    state.ent_pos[0, 2] = torch.tensor([15.0, 10.0])  # farther

    vis = perception.visibility(state, bank, params, cfg)
    perception.select_target(state, vis, cfg)
    assert state.ent_target[0, 0].item() == 1


def test_select_target_is_sticky_no_oscillation_between_equidistant():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([12.0, 10.0])  # exactly equidistant
    state.ent_pos[0, 2] = torch.tensor([8.0, 10.0])   # exactly equidistant

    vis = perception.visibility(state, bank, params, cfg)
    perception.select_target(state, vis, cfg)
    first_pick = state.ent_target[0, 0].item()
    assert first_pick in (1, 2)

    # run select_target repeatedly under the exact same equidistant conditions -- a
    # non-sticky implementation could flip-flop between argmin ties; this must not.
    for _ in range(20):
        vis = perception.visibility(state, bank, params, cfg)
        perception.select_target(state, vis, cfg)
        assert state.ent_target[0, 0].item() == first_pick


def test_select_target_switches_when_current_dies():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([11.0, 10.0])
    state.ent_pos[0, 2] = torch.tensor([15.0, 10.0])

    vis = perception.visibility(state, bank, params, cfg)
    perception.select_target(state, vis, cfg)
    assert state.ent_target[0, 0].item() == 1

    state.ent_alive[0, 1] = False
    vis = perception.visibility(state, bank, params, cfg)
    perception.select_target(state, vis, cfg)
    assert state.ent_target[0, 0].item() == 2


def test_select_target_no_candidate_gives_negative_one():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)
    for e in range(1, cfg.n_entities):
        state.ent_alive[0, e] = False

    vis = perception.visibility(state, bank, params, cfg)
    perception.select_target(state, vis, cfg)
    assert state.ent_target[0, 0].item() == -1


# ---- incoming_threat --------------------------------------------------------------

def test_incoming_threat_points_away_from_approaching_projectile():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])

    state.prj_alive[0, 0] = True
    state.prj_owner[0, 0] = 1  # not entity 0
    state.prj_pos[0, 0] = torch.tensor([5.0, 10.0])
    state.prj_vel[0, 0] = torch.tensor([1.0, 0.0])  # heading straight at entity 0

    threat = perception.incoming_threat(state, params, cfg)
    assert threat[0, 0, 0].item() > 0.0  # points from the projectile (west) toward the hero (east)


def test_incoming_threat_ignores_own_projectiles():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.prj_alive[0, 0] = True
    state.prj_owner[0, 0] = 0  # entity 0's own shot
    state.prj_pos[0, 0] = torch.tensor([5.0, 10.0])
    state.prj_vel[0, 0] = torch.tensor([1.0, 0.0])

    threat = perception.incoming_threat(state, params, cfg)
    assert torch.allclose(threat[0, 0], torch.zeros(2))


def test_incoming_threat_no_projectiles_is_zero():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    threat = perception.incoming_threat(state, params, cfg)
    assert torch.allclose(threat, torch.zeros_like(threat))
    assert not torch.any(torch.isnan(threat))


# ---- nearest_alive --------------------------------------------------------------

def test_nearest_alive_basic():
    points_pos = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [3.0, 0.0]]])
    points_alive = torch.tensor([[True, True, True]])
    from_pos = torch.tensor([[[2.0, 0.0]]])
    idx, dist = perception.nearest_alive(points_pos, points_alive, from_pos)
    assert idx[0, 0].item() == 2
    assert abs(dist[0, 0].item() - 1.0) < 1e-4


def test_nearest_alive_skips_dead_points():
    points_pos = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [3.0, 0.0]]])
    points_alive = torch.tensor([[True, True, False]])  # nearest (idx 2) is dead
    from_pos = torch.tensor([[[2.0, 0.0]]])
    idx, dist = perception.nearest_alive(points_pos, points_alive, from_pos)
    assert idx[0, 0].item() == 0
    assert abs(dist[0, 0].item() - 2.0) < 1e-4


# ---- in_zone / nearest_safe_point --------------------------------------------------

def test_in_zone_outside_rect_is_true():
    zone_lo = torch.tensor([[5.0, 5.0]])
    zone_hi = torch.tensor([[15.0, 15.0]])
    inside = torch.tensor([[10.0, 10.0]])
    outside = torch.tensor([[20.0, 10.0]])
    assert not bool(perception.in_zone(inside, zone_lo, zone_hi)[0])
    assert bool(perception.in_zone(outside, zone_lo, zone_hi)[0])


def test_nearest_safe_point_clamps_into_rect():
    zone_lo = torch.tensor([[5.0, 5.0]])
    zone_hi = torch.tensor([[15.0, 15.0]])
    pos = torch.tensor([[20.0, 2.0]])
    nearest = perception.nearest_safe_point(pos, zone_lo, zone_hi)
    assert torch.allclose(nearest, torch.tensor([[15.0, 5.0]]))


def test_nearest_safe_point_inside_rect_is_unchanged():
    zone_lo = torch.tensor([[5.0, 5.0]])
    zone_hi = torch.tensor([[15.0, 15.0]])
    pos = torch.tensor([[10.0, 12.0]])
    nearest = perception.nearest_safe_point(pos, zone_lo, zone_hi)
    assert torch.allclose(nearest, pos)


# ---- zone_clearance -----------------------------------------------------------------

def test_zone_clearance_is_distance_to_the_nearest_edge_from_inside():
    zone_lo = torch.tensor([[5.0, 5.0]])
    zone_hi = torch.tensor([[15.0, 15.0]])
    # 2 from the west edge, 8 from the east, 4 from the north, 6 from the south -> 2.
    pos = torch.tensor([[7.0, 9.0]])
    assert perception.zone_clearance(pos, zone_lo, zone_hi)[0].item() == 2.0
    # dead center of a 10x10 rect
    center = torch.tensor([[10.0, 10.0]])
    assert perception.zone_clearance(center, zone_lo, zone_hi)[0].item() == 5.0


def test_zone_clearance_clamps_to_zero_outside_rather_than_going_negative():
    zone_lo = torch.tensor([[5.0, 5.0]])
    zone_hi = torch.tensor([[15.0, 15.0]])
    outside = torch.tensor([[20.0, 10.0]])
    assert perception.zone_clearance(outside, zone_lo, zone_hi)[0].item() == 0.0


# ---- bush_scan ----------------------------------------------------------------------

def _naive_bush_scan(pos, tiles, radius):
    """Plain-python nearest-bush-tile-center, looping over every tile in the map. Deliberately
    shares no code with perception.bush_scan -- it is the independent oracle for the batched
    (N,E,K) gather, whose whole reason for existing is speed."""
    h, w = tiles.shape
    best_dist, best_pos = float("inf"), None
    px, py = float(pos[0]), float(pos[1])
    ix, iy = int(px // 1), int(py // 1)
    for ty in range(h):
        for tx in range(w):
            if int(tiles[ty, tx]) != int(Tile.BUSH):
                continue
            if (tx - ix) ** 2 + (ty - iy) ** 2 > radius * radius:
                continue
            d = ((px - (tx + 0.5)) ** 2 + (py - (ty + 0.5)) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best_pos = d, (tx + 0.5, ty + 0.5)
    return best_pos, best_dist


def test_bush_scan_matches_a_naive_per_offset_reference():
    cfg, params = _cfg_and_params(n_envs=1)
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    for ty, tx in ((5, 5), (5, 6), (12, 3), (9, 14), (16, 16), (2, 11)):
        tiles[ty, tx] = Tile.BUSH
    bank = _FakeBank(tiles)

    torch.manual_seed(0)
    for _ in range(25):
        state.ent_pos.uniform_(1.5, 18.5)
        scan = perception.bush_scan(state.ent_pos, state.map_id, bank, cfg)
        for e in range(cfg.n_entities):
            expected_pos, expected_dist = _naive_bush_scan(
                state.ent_pos[0, e], tiles, cfg.bots_bush_search_tiles,
            )
            if expected_pos is None:
                assert not bool(scan.found[0, e])
                continue
            assert bool(scan.found[0, e])
            assert abs(scan.dist[0, e].item() - expected_dist) < 1e-4
            assert torch.allclose(scan.pos[0, e], torch.tensor(expected_pos), atol=1e-4)


def test_bush_scan_includes_the_tile_the_entity_is_standing_on():
    """Regression: the per-offset ancestor of this function excluded offset (0,0), which made a
    bot standing in an isolated bush believe no cover existed and walk out of it."""
    cfg, params = _cfg_and_params(n_envs=1)
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    tiles[10, 10] = Tile.BUSH  # the only bush, nothing else within the search radius
    bank = _FakeBank(tiles)

    state.ent_pos[0, 1] = torch.tensor([10.5, 10.5])
    scan = perception.bush_scan(state.ent_pos, state.map_id, bank, cfg)
    assert bool(scan.found[0, 1])
    assert scan.dist[0, 1].item() == 0.0
    assert torch.allclose(scan.pos[0, 1], torch.tensor([10.5, 10.5]))


def test_bush_scan_excludes_bushes_inside_the_zone_and_within_its_margin():
    cfg, params = _cfg_and_params(n_envs=1)
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    tiles[10, 12] = Tile.BUSH  # (12.5, 10.5)
    tiles[10, 8] = Tile.BUSH   # (8.5, 10.5)
    bank = _FakeBank(tiles)
    # NOT (10.5, 10.5): that is exactly 2.0 tiles from BOTH bushes, so which one argmin returns is
    # a tie-break detail rather than the thing under test.
    state.ent_pos[0, 1] = torch.tensor([11.5, 10.5])  # 1.0 from the east bush, 3.0 from the west

    zone_lo = torch.tensor([[2.0, 2.0]]).unsqueeze(1)    # (N,1,2), as bots/policy.zone_rect gives
    zone_hi = torch.tensor([[13.0, 18.0]]).unsqueeze(1)   # east bush clearance = 0.5

    # No margin: the east bush is still technically safe, and it is the nearer of the two.
    scan = perception.bush_scan(state.ent_pos, state.map_id, bank, cfg,
                                zone_lo=zone_lo, zone_hi=zone_hi, zone_margin=0.0)
    assert torch.allclose(scan.pos[0, 1], torch.tensor([12.5, 10.5]))

    # With a 2-tile margin it is rejected and the deeper-inside bush wins.
    scan = perception.bush_scan(state.ent_pos, state.map_id, bank, cfg,
                                zone_lo=zone_lo, zone_hi=zone_hi, zone_margin=2.0)
    assert torch.allclose(scan.pos[0, 1], torch.tensor([8.5, 10.5]))


# ---- hunt_waypoint ------------------------------------------------------------------

def _waypoint_bank(tiles, points):
    bank = _FakeBank(tiles)
    bank.bush_wp = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
    bank.n_bush_wp = torch.tensor([len(points)], dtype=torch.int64)
    return bank


def test_hunt_waypoint_picks_the_nearest_unvisited_and_skips_visited_bits():
    cfg, params = _cfg_and_params(n_envs=1)
    state = _fresh_state(cfg, params)
    bank = _waypoint_bank(_grid(20, 20), [[16.5, 10.5], [3.5, 10.5], [10.5, 3.5]])
    state.ent_pos[0, 1] = torch.tensor([13.0, 10.5])  # 3.5 / 9.5 / 7.5 away respectively

    seen = torch.zeros((1, cfg.n_entities), dtype=torch.int64)
    hunt = perception.hunt_waypoint(state.ent_pos, state.map_id, bank, cfg, seen)
    assert bool(hunt.found[0, 1])
    assert int(hunt.idx[0, 1]) == 0
    assert torch.allclose(hunt.pos[0, 1], torch.tensor([16.5, 10.5]))

    seen[0, 1] = 1 << 0  # waypoint 0 already visited -> the third one is next-nearest
    hunt = perception.hunt_waypoint(state.ent_pos, state.map_id, bank, cfg, seen)
    assert int(hunt.idx[0, 1]) == 2
    assert torch.allclose(hunt.pos[0, 1], torch.tensor([10.5, 3.5]))


def test_hunt_waypoint_reports_all_visited_distinctly_from_no_waypoints():
    """`found` False means "nowhere to go next"; `any_valid` is what separates "I have swept
    everything" (reset the mask) from "this map has no bushes" (go wander)."""
    cfg, params = _cfg_and_params(n_envs=1)
    state = _fresh_state(cfg, params)

    bank = _waypoint_bank(_grid(20, 20), [[16.5, 10.5], [3.5, 10.5]])
    all_seen = torch.full((1, cfg.n_entities), 0b11, dtype=torch.int64)
    hunt = perception.hunt_waypoint(state.ent_pos, state.map_id, bank, cfg, all_seen)
    assert not bool(hunt.found[0, 1])
    assert bool(hunt.any_valid[0, 1])  # they exist, they are just all visited

    empty = _FakeBank(_grid(20, 20))
    empty.bush_wp = torch.zeros((1, 0, 2))
    empty.n_bush_wp = torch.zeros((1,), dtype=torch.int64)
    seen = torch.zeros((1, cfg.n_entities), dtype=torch.int64)
    hunt = perception.hunt_waypoint(state.ent_pos, state.map_id, empty, cfg, seen)
    assert not bool(hunt.found[0, 1])
    assert not bool(hunt.any_valid[0, 1])


def test_hunt_waypoint_ignores_padded_slots_beyond_the_maps_own_count():
    cfg, params = _cfg_and_params(n_envs=1)
    state = _fresh_state(cfg, params)
    bank = _waypoint_bank(_grid(20, 20), [[16.5, 10.5], [3.5, 10.5]])
    # Pad with a (0,0) slot -- _pad_slots' real zero padding -- and claim only 1 is valid. The
    # padded slots must not be selectable even though they are much nearer than the real one.
    bank.bush_wp = torch.tensor([[[16.5, 10.5], [0.0, 0.0], [0.0, 0.0]]])
    bank.n_bush_wp = torch.tensor([1], dtype=torch.int64)
    state.ent_pos[0, 1] = torch.tensor([2.0, 2.0])  # far closer to (0,0) than to (16.5,10.5)

    seen = torch.zeros((1, cfg.n_entities), dtype=torch.int64)
    hunt = perception.hunt_waypoint(state.ent_pos, state.map_id, bank, cfg, seen)
    assert int(hunt.idx[0, 1]) == 0
    assert torch.allclose(hunt.pos[0, 1], torch.tensor([16.5, 10.5]))


def test_hunt_waypoint_skips_waypoints_the_zone_has_reached():
    cfg, params = _cfg_and_params(n_envs=1)
    state = _fresh_state(cfg, params)
    bank = _waypoint_bank(_grid(20, 20), [[16.5, 10.5], [8.5, 10.5]])
    state.ent_pos[0, 1] = torch.tensor([13.0, 10.5])
    seen = torch.zeros((1, cfg.n_entities), dtype=torch.int64)

    zone_lo = torch.tensor([[2.0, 2.0]]).unsqueeze(1)
    zone_hi = torch.tensor([[17.0, 18.0]]).unsqueeze(1)  # (16.5,10.5) has only 0.5 clearance
    hunt = perception.hunt_waypoint(state.ent_pos, state.map_id, bank, cfg, seen,
                                    zone_lo=zone_lo, zone_hi=zone_hi, zone_margin=2.0)
    assert torch.allclose(hunt.pos[0, 1], torch.tensor([8.5, 10.5]))


def test_bush_scan_rejects_a_wrongly_shaped_zone_rect():
    cfg, params = _cfg_and_params(n_envs=1)
    state = _fresh_state(cfg, params)
    bank = _FakeBank(_grid(20, 20))
    bad_lo = torch.zeros((1, 1, 1, 2))  # rank 4: an already-unsqueezed rect, the natural mistake
    bad_hi = torch.full((1, 1, 1, 2), 13.0)
    with pytest.raises(ValueError, match="zone bounds"):
        perception.bush_scan(state.ent_pos, state.map_id, bank, cfg,
                             zone_lo=bad_lo, zone_hi=bad_hi)


# ---- batched leading-dim smoke test -------------------------------------------

def test_batched_smoke():
    cfg, params = _cfg_and_params(n_envs=4)
    state = _fresh_state(cfg, params, n_envs=4)
    tiles = _grid(20, 20)
    tiles[10, 10] = Tile.BUSH
    bank = _FakeBank(tiles)
    state.ent_pos.uniform_(2, 18)

    vis = perception.visibility(state, bank, params, cfg)
    los = perception.raw_los(state, bank, cfg)
    perception.select_target(state, vis, cfg)
    threat = perception.incoming_threat(state, params, cfg)

    assert vis.shape == (4, cfg.n_entities, cfg.n_entities)
    assert los.shape == (4, cfg.n_entities, cfg.n_entities)
    assert state.ent_target.shape == (4, cfg.n_entities)
    assert threat.shape == (4, cfg.n_entities, 2)
    assert not torch.any(torch.isnan(threat))
