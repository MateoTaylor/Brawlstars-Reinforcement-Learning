"""maps/nav.py's pathfinding tables, checked against what they promise: shortest 8-connected paths
with no wall steps and no corner cuts (against networkx's Dijkstra, not against themselves), every
walk ending at its goal, ties taken along the straight bearing, a clear straight walk from every
goal's anchor on the real map pool, which a body following policy.path_toward turns into every
goal reached, the exact walk test against a geometric reference, the centre path's box against a
plain walk, and the corner rule against terrain.resolve_move. Bot behaviour under `bots.nav` is
in test_personality.py."""
import math
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch
import yaml

from brawl_sim.bots import policy
from brawl_sim.config import body_radius_tiles, load_config
from brawl_sim.core import geometry as geo
from brawl_sim.core import terrain
from brawl_sim.maps import loader, nav

CONFIGS_DEFAULT = "configs/default.yaml"
# Three of the pool's most walled maps: the walk test is the expensive one.
WALK_MAPS = ("skull_creek", "hot_maze", "feast_or_famine")


@pytest.fixture(scope="module")
def pool():
    """The bank as the env builds it: the bots' body radius from default.yaml."""
    cfg = load_config(CONFIGS_DEFAULT, overrides={"bots": {"nav": True}})
    radius = body_radius_tiles(yaml.safe_load(open(CONFIGS_DEFAULT).read()))
    return cfg, loader.build_map_bank(cfg, "cpu", body_radius=radius)


def _octile_graph(free: np.ndarray) -> nx.Graph:
    """The reference: walkable tiles, orthogonal edges cost 1, a diagonal costs sqrt(2) and
    exists only when its whole 2x2 block is walkable."""
    h, w = free.shape
    g = nx.Graph()
    for y in range(h):
        for x in range(w):
            if not free[y, x]:
                continue
            g.add_node((x, y))
            if x + 1 < w and free[y, x + 1]:
                g.add_edge((x, y), (x + 1, y), weight=1.0)
            if y + 1 < h and free[y + 1, x]:
                g.add_edge((x, y), (x, y + 1), weight=1.0)
            if x + 1 < w and y + 1 < h and free[y, x + 1] and free[y + 1, x] and free[y + 1, x + 1]:
                g.add_edge((x, y), (x + 1, y + 1), weight=math.sqrt(2.0))
            if x >= 1 and y + 1 < h and free[y, x - 1] and free[y + 1, x] and free[y + 1, x - 1]:
                g.add_edge((x, y), (x - 1, y + 1), weight=math.sqrt(2.0))
    return g


def _free(cfg, name):
    tiles = loader.load_map_csv(loader.CSV_DIR / f"{name}.csv", cfg)
    return ~loader.TILE_BLOCKS_UNIT.numpy()[tiles]


def test_distance_fields_match_dijkstra_on_a_walled_map(pool):
    cfg, _bank = pool
    free_np = _free(cfg, "skull_creek")
    g = _octile_graph(free_np)
    anchor_xy = nav.anchors(free_np, cfg.bots_nav_anchor_tiles)
    picks = anchor_xy[np.linspace(0, len(anchor_xy) - 1, 6).round().astype(int)]
    centre = nav.centre_tiles(free_np)

    h, w = free_np.shape
    sources = torch.zeros((len(picks) + 1, h, w), dtype=torch.bool)
    for i, (x, y) in enumerate(picks):
        sources[i, y, x] = True
    sources[-1, torch.as_tensor(centre[:, 1]), torch.as_tensor(centre[:, 0])] = True
    dist = nav.distance_fields(torch.as_tensor(free_np), sources)

    refs = [nx.single_source_dijkstra_path_length(g, (int(x), int(y))) for x, y in picks]
    refs.append(nx.multi_source_dijkstra_path_length(g, {(int(x), int(y)) for x, y in centre}))
    for s, ref in enumerate(refs):
        want = torch.full((h, w), float("inf"))
        for (x, y), d in ref.items():
            want[y, x] = d
        assert torch.equal(torch.isinf(dist[s]), torch.isinf(want))
        finite = torch.isfinite(want)
        assert torch.allclose(dist[s][finite], want[finite], atol=1e-4)


def _walk(codes: torch.Tensor, free: torch.Tensor):
    """Walks every (source, start tile) pair along `codes` (S, H, W) at once. Asserts every step
    lands on a walkable tile and every diagonal clears its 2x2 block. Returns the end tiles and
    each walk's summed step cost."""
    s_count, h, w = codes.shape
    off = torch.tensor(nav.OFFSETS)
    src = torch.arange(s_count).view(-1, 1, 1).expand(s_count, h, w)
    y = torch.arange(h).view(1, -1, 1).expand(s_count, h, w).clone()
    x = torch.arange(w).view(1, 1, -1).expand(s_count, h, w).clone()
    cost = torch.zeros((s_count, h, w), dtype=torch.float64)
    for _ in range(h * w):
        code = codes[src, y, x].long()
        moving = code > 0
        if not moving.any():
            return y, x, cost
        dx, dy = off[code, 0], off[code, 1]
        ny, nx_ = y + dy, x + dx
        assert free[ny, nx_][moving].all(), "a step into a wall"
        diag = moving & (dx != 0) & (dy != 0)
        assert (free[y, nx_] & free[ny, x])[diag].all(), "a diagonal cut a wall corner"
        cost += torch.where(diag, math.sqrt(2.0), moving.double())
        y, x = ny, nx_
    raise AssertionError("a walk never ended: the field has a loop")


@pytest.mark.parametrize("name", WALK_MAPS)
def test_every_walk_reaches_its_goal_by_a_shortest_legal_path(pool, name):
    """Every real slot: the cell anchors, the orphans (an anchor each, past them) and the centre."""
    cfg, bank = pool
    m = cfg.map_names.index(name)
    free = ~bank.blocks_unit[m]
    centre = nav.centre_tiles(free.numpy())
    h, w = free.shape
    real = torch.nonzero(bank.nav_anchor_tile[m] >= 0).flatten()
    tiles = bank.nav_anchor_tile[m, real]
    n = len(real)
    codes = bank.nav_next[m, real.tolist() + [nav.centre_slot(bank)]]

    sources = torch.zeros((n + 1, h, w), dtype=torch.bool)
    sources[torch.arange(n), tiles // w, tiles % w] = True
    sources[n, torch.as_tensor(centre[:, 1]), torch.as_tensor(centre[:, 0])] = True
    dist = nav.distance_fields(free, sources)

    end_y, end_x, cost = _walk(codes, free)
    reach = torch.isfinite(dist)
    src = torch.arange(n + 1).view(-1, 1, 1).expand_as(end_y)
    assert sources[src, end_y, end_x][reach].all(), "a walk stopped short of its goal"
    assert torch.allclose(cost[reach], dist[reach].double(), atol=1e-3), "a walk was not shortest"
    assert (codes[~reach] == 0).all()


def test_open_ground_walks_the_straight_line_not_a_dog_leg():
    """From 15 tiles east and 6 south of the goal, a diagonal-first dog-leg strays 3.3 tiles off
    the line; the bearing tie-break keeps every tile within one."""
    h = w = 30
    free = torch.ones((h, w), dtype=torch.bool)
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
    goal = (5, 10)
    sources = torch.zeros((1, h, w), dtype=torch.bool)
    sources[0, goal[1], goal[0]] = True
    dist = nav.distance_fields(free, sources)
    codes = nav.step_codes(free, dist, torch.tensor([[goal[0] + 0.5, goal[1] + 0.5]]))

    x, y = 20, 16
    line = torch.tensor([x - goal[0], y - goal[1]], dtype=torch.float32)
    worst = 0.0
    while codes[0, y, x] > 0:
        dx, dy = nav.OFFSETS[int(codes[0, y, x])]
        x, y = x + dx, y + dy
        rel = torch.tensor([x - goal[0], y - goal[1]], dtype=torch.float32)
        worst = max(worst, abs(float(rel[0] * line[1] - rel[1] * line[0])) / float(line.norm()))
    assert (x, y) == goal
    assert worst <= 1.0


def test_every_goal_tile_on_the_pool_has_a_clear_walk_from_its_anchor(pool):
    """The handover in policy.path_toward: a bot on a goal's anchor walks straight to the goal
    tile's centre, so that walk must be clear for the body, by the exact test the bots walk by,
    on all 102,935 walkable tiles: a tile no neighbouring anchor's walk reaches is an anchor of
    its own (56 on the pool, 2026-10-07). The centre line is checked by terrain.march too, and
    the body by terrain.body_travel, the circle test walking itself uses, which the edge lines
    approximate: by it, choosing anchors by the centre line left 3,105 tiles the body cannot walk
    to."""
    cfg, bank = pool
    radius = body_radius_tiles(yaml.safe_load(open(CONFIGS_DEFAULT).read()))
    total = stuck = 0
    for m, name in enumerate(cfg.map_names):
        free = ~bank.blocks_unit[m]
        ys, xs = torch.nonzero(free, as_tuple=True)
        tile = bank.nav_anchor_tile[m, bank.nav_anchor_of[m, ys, xs]]
        assert bool((tile >= 0).all()), name
        start = torch.stack([tile % cfg.map_w, tile // cfg.map_w], dim=1).float() + 0.5
        goal = torch.stack([xs, ys], dim=1).float() + 0.5
        delta = goal - start
        dist = geo.safe_norm(delta, dim=-1)
        map_id = torch.full((len(xs),), m)
        walk = nav.walk_blocked(bank.blocks_unit, map_id, start, delta, radius, cfg, max_tiles=8.0)
        assert not walk.any(), f"{name}: {int(walk.sum())} goal tiles the body cannot walk to"
        hit, _, _ = terrain.march(bank.blocks_unit, map_id, start, delta, dist, cfg)
        assert not hit.any(), f"{name}: {int(hit.sum())} goal tiles with a blocked anchor walk"
        travel = terrain.body_travel(bank.blocks_unit, map_id, start, delta, dist,
                                     torch.tensor(radius), cfg)
        total += len(xs)
        stuck += int((travel < dist).sum())
    assert stuck <= total // 1000, f"{stuck} of {total} goal tiles the body cannot walk to"


@pytest.mark.parametrize("name", ["crescent_lakes", "bush_halo"])
def test_a_body_following_path_toward_reaches_every_walkable_goal_tile(pool, name):
    """policy.path_toward end to end, on the two maps where a 2026-10-07 census of the pool
    (a goal on every walkable tile) left the most goals unreached: 8 each, of 59. A body starts
    at least 9 tiles from its goal in the same component and walks the offset path_toward gives
    through terrain.resolve_move at 2.4 tiles/s; start and goal sit a little off their tiles'
    centres, as a body's own would. Each of those 59 had stopped a tile or two short of its
    goal's anchor, swapping between the field, which led back to the anchor, and the straight
    walk, which led away from it, every tick for good."""
    cfg, bank = pool
    m = cfg.map_names.index(name)
    free = (~bank.blocks_unit[m]).numpy()
    radius = body_radius_tiles(yaml.safe_load(open(CONFIGS_DEFAULT).read()))
    part = {}
    for i, tiles in enumerate(nx.connected_components(_octile_graph(free))):
        part.update(dict.fromkeys(tiles, i))
    yx = np.argwhere(free)
    label = np.array([part[(x, y)] for y, x in yx])
    rng = np.random.default_rng(0)
    slack = 0.5 - radius - 1e-3
    starts, goals = [], []
    for gy, gx in yx:
        far = np.nonzero((label == part[(gx, gy)]) & (np.hypot(yx[:, 1] - gx, yx[:, 0] - gy) >= 9.0))[0]
        if len(far):
            sy, sx = yx[far[rng.integers(len(far))]]
            starts.append(np.array([sx, sy]) + 0.5 + rng.uniform(-slack, slack, 2))
            goals.append(np.array([gx, gy]) + 0.5 + rng.uniform(-slack, slack, 2))

    state = SimpleNamespace(ent_pos=torch.as_tensor(np.array(starts), dtype=torch.float32).unsqueeze(0),
                            map_id=torch.tensor([m]))
    goal = torch.as_tensor(np.array(goals), dtype=torch.float32).unsqueeze(0)
    params = SimpleNamespace(unit_radius=torch.tensor([radius]))
    done = torch.zeros(goal.shape[:2], dtype=torch.bool)
    for _ in range(1000):
        aim = policy.path_toward(state, bank, params, cfg, goal)
        step = geo.normalize(aim) * (2.4 * cfg.dt) * (~done).unsqueeze(-1)
        state.ent_pos = terrain.resolve_move(bank.blocks_unit, state.map_id, state.ent_pos, step,
                                             params.unit_radius.view(-1, 1), cfg)
        done |= geo.safe_norm(state.ent_pos - goal, dim=-1) < 0.3
        if bool(done.all()):
            break
    short = torch.nonzero(~done[0]).flatten()[:5].tolist()
    assert bool(done.all()), (f"{name}: {int((~done).sum())} of {done.numel()} goals never reached, "
                              f"e.g. {[tuple(goal[0, i].tolist()) for i in short]}")


def test_an_orphan_tile_is_its_own_anchor_and_every_field_ends_on_its_anchor(pool):
    """The slots past a map's cell anchors are the orphans: each sits on a walkable tile that
    reads its own field. Every anchor field gives no step on the tile nav_anchor_tile names,
    which is what nav.at_anchor compares against (the walk test above checks that each path ends
    there); the centre field names none."""
    cfg, bank = pool
    orphans = 0
    for m, name in enumerate(cfg.map_names):
        free = ~bank.blocks_unit[m]
        n_cell = len(nav.anchors(free.numpy(), cfg.bots_nav_anchor_tiles))
        tiles = bank.nav_anchor_tile[m]
        real = torch.nonzero(tiles >= 0).flatten()
        assert int(tiles[nav.centre_slot(bank)]) == -1
        assert bool((bank.nav_next[m, real, tiles[real] // cfg.map_w, tiles[real] % cfg.map_w] == 0).all())
        for a in real[n_cell:].tolist():
            x, y = int(tiles[a]) % cfg.map_w, int(tiles[a]) // cfg.map_w
            assert bool(free[y, x]) and int(bank.nav_anchor_of[m, y, x]) == a, (name, a)
            orphans += 1
    assert orphans > 0                      # the pool has some, or this checks nothing


def _crosses(p0: np.ndarray, d: np.ndarray, h: int, w: int) -> np.ndarray:
    """(n, h, w) bool, the reference for nav.segment_blocked: segment i overlaps tile (y, x) for
    a positive length, found by clipping it to each tile's square (Liang-Barsky), the tile it
    starts in left out."""
    def span(p, dd, lo):
        inside = (p > lo) & (p < lo + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (lo - p) / dd, (lo + 1 - p) / dd
        t0 = np.where(dd == 0, np.where(inside, -np.inf, np.inf), np.minimum(a, b))
        t1 = np.where(dd == 0, np.where(inside, np.inf, -np.inf), np.maximum(a, b))
        return t0, t1

    px, py = p0[:, 0, None, None], p0[:, 1, None, None]
    xs, ys = np.arange(w)[None, None, :], np.arange(h)[None, :, None]
    x0, x1 = span(px, d[:, 0, None, None], xs)
    y0, y1 = span(py, d[:, 1, None, None], ys)
    overlap = np.maximum(np.maximum(x0, y0), 0.0) < np.minimum(np.minimum(x1, y1), 1.0)
    return overlap & ~((np.floor(px) == xs) & (np.floor(py) == ys))


def test_a_segment_is_blocked_exactly_where_it_enters_a_blocked_tile():
    """nav.segment_blocked against a geometric reference on 4,000 random segments up to 8 tiles
    long over a quarter-walled map. terrain.march, sampling every half tile, misses some corners
    it clips (2026-10-07: a sampled walk read clear from one spot and blocked from the next, and
    a bot swapped between the walk and its path every tick for good); the exact test misses none,
    a tail of a clear segment is clear, and leaving the map is blocked."""
    cfg = load_config(CONFIGS_DEFAULT, overrides={"world": {"map_h": 30, "map_w": 30}})
    rng = np.random.default_rng(0)
    mask = rng.random((30, 30)) < 0.25
    free_xy = np.argwhere(~mask)[:, ::-1]                                  # (x, y)
    p0 = free_xy[rng.integers(len(free_xy), size=4000)] + rng.random((4000, 2))
    ang = rng.random(4000) * 2 * math.pi
    p1 = np.clip(p0 + np.stack([np.cos(ang), np.sin(ang)], 1) * rng.random((4000, 1)) * 8, 0.01, 29.99)
    p0, d = p0.astype(np.float32), (p1 - p0).astype(np.float32)
    want = (_crosses(p0.astype(np.float64), d.astype(np.float64), 30, 30) & mask).any(axis=(1, 2))

    bank, map0 = torch.as_tensor(mask).unsqueeze(0), torch.zeros((), dtype=torch.int64)
    tp0, td = torch.as_tensor(p0), torch.as_tensor(d)
    got = nav.segment_blocked(bank, map0, tp0, td, cfg, 8.0).numpy()
    assert np.array_equal(got, want), int((got != want).sum())
    sampled, _, _ = terrain.march(bank, map0, tp0, td, geo.safe_norm(td, dim=-1), cfg, max_tiles=8.0)
    assert not bool((sampled & ~torch.as_tensor(got)).any())               # never stricter
    assert bool((torch.as_tensor(got) & ~sampled).any())                   # and it does miss some
    for frac in (0.25, 0.5, 0.9):
        tail = nav.segment_blocked(bank, map0, tp0 + td * frac, td * (1 - frac), cfg, 8.0).numpy()
        assert not (~got & tail).any(), frac
    out = nav.segment_blocked(torch.zeros_like(bank), map0, torch.tensor([[1.5, 1.5], [1.5, 1.5]]),
                              torch.tensor([[-2.0, 0.0], [2.0, 0.0]]), cfg, 8.0)
    assert out.tolist() == [True, False]


@pytest.mark.parametrize("name", ["open", "skull_creek"])
def test_the_centre_field_ends_at_the_walkable_tiles_nearest_the_centre(pool, name):
    """Open's centre is walkable; Skull Creek's is walled, and its walks end on the nearest
    walkable ring instead, the radius widened a whole tile at a time from 1."""
    cfg, bank = pool
    m = cfg.map_names.index(name)
    free = ~bank.blocks_unit[m]
    codes = bank.nav_next[m, nav.centre_slot(bank)].unsqueeze(0)
    end_y, end_x, _ = _walk(codes, free)

    def off_centre(x, y):
        return torch.hypot(x.float() + 0.5 - cfg.map_w / 2, y.float() + 0.5 - cfg.map_h / 2)

    ys, xs = torch.nonzero(free, as_tuple=True)
    radius = max(1.0, math.ceil(float(off_centre(xs, ys).min())))
    assert off_centre(end_x[0][free], end_y[0][free]).max() <= radius


def test_the_centre_box_bounds_the_tiles_each_centre_walk_visits(pool):
    """nav_centre_box (pointer doubling) against a plain walk: every tile of every pool map steps
    along the centre field to its end, keeping its x and y range on the way."""
    cfg, bank = pool
    off = torch.tensor(nav.OFFSETS)
    for m, name in enumerate(cfg.map_names):
        codes = bank.nav_next[m, nav.centre_slot(bank)]
        h, w = codes.shape
        y = torch.arange(h).view(-1, 1).expand(h, w).clone()
        x = torch.arange(w).view(1, -1).expand(h, w).clone()
        lo_x, lo_y, hi_x, hi_y = x.clone(), y.clone(), x.clone(), y.clone()
        for _ in range(h * w):
            code = codes[y, x].long()
            if not (code > 0).any():
                break
            x, y = x + off[code, 0], y + off[code, 1]
            lo_x, lo_y = torch.minimum(lo_x, x), torch.minimum(lo_y, y)
            hi_x, hi_y = torch.maximum(hi_x, x), torch.maximum(hi_y, y)
        want = torch.stack([lo_x, lo_y, hi_x, hi_y], dim=-1)
        assert torch.equal(bank.nav_centre_box[m].long(), want), name


def _tile_centres(h, w):
    """(1, h*w, 2): every tile's centre, as one env's entity axis, row-major."""
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack([xs, ys], dim=-1).reshape(1, -1, 2).float() + 0.5


def _rect(lo, hi):
    return torch.tensor([[lo]]), torch.tensor([[hi]])


def test_the_centre_path_dip_is_the_room_its_walk_gives_up(pool):
    """centre_path_dip (four numbers a tile, from the box) against a plain walk: for every tile
    of three walled maps, the room its own centre has inside a centred rect, less the least room of
    any tile centre its centre walk visits."""
    cfg, bank = pool
    lo, hi = (8.0, 8.0), (cfg.map_w - 8.0, cfg.map_h - 8.0)
    off = torch.tensor(nav.OFFSETS)

    def room(x, y):
        cx, cy = x + 0.5, y + 0.5
        return torch.minimum(torch.minimum(cx - lo[0], hi[0] - cx),
                             torch.minimum(cy - lo[1], hi[1] - cy))

    for name in WALK_MAPS:
        m = cfg.map_names.index(name)
        codes = bank.nav_next[m, nav.centre_slot(bank)]
        h, w = codes.shape
        y = torch.arange(h).view(-1, 1).expand(h, w).clone()
        x = torch.arange(w).view(1, -1).expand(h, w).clone()
        start = room(x, y)
        least = start.clone()
        for _ in range(h * w):
            code = codes[y, x].long()
            if not (code > 0).any():
                break
            x, y = x + off[code, 0], y + off[code, 1]
            least = torch.minimum(least, room(x, y))

        dip = nav.centre_path_dip(bank, torch.tensor([m]), _tile_centres(h, w), *_rect(lo, hi), cfg)
        assert torch.equal(dip.reshape(h, w), start - least), name
        assert bool((dip > 0).any()), name     # the map has pockets for the rect, or this is idle


def test_the_centre_path_dip_is_zero_on_open_ground_and_for_a_rect_off_centre(pool):
    """On open ground a centre walk only ever moves inward, so it gives up no room and a gas rule
    that subtracts the dip reads exactly as before. A rect off the map's centre has no dip on a
    walled map either: the centre field is not its way out."""
    cfg = load_config(CONFIGS_DEFAULT, overrides={"world": {"map_h": 30, "map_w": 30},
                                                  "bots": {"nav": True}})
    blocked = torch.ones((1, 30, 30), dtype=torch.bool)
    blocked[:, 1:-1, 1:-1] = False
    radius = body_radius_tiles(yaml.safe_load(open(CONFIGS_DEFAULT).read()))
    *_, box = nav.build_tables(blocked, cfg, "cpu", radius)
    open_bank = SimpleNamespace(nav_centre_box=box)
    for lo, hi in (((8.0, 8.0), (22.0, 22.0)), ((3.0, 3.0), (27.0, 27.0))):
        dip = nav.centre_path_dip(open_bank, torch.tensor([0]), _tile_centres(30, 30),
                                  *_rect(lo, hi), cfg)
        assert torch.equal(dip, torch.zeros_like(dip)), (lo, hi)

    pool_cfg, bank = pool
    m = pool_cfg.map_names.index(WALK_MAPS[0])
    lo, hi = (10.0, 8.0), (pool_cfg.map_w - 6.0, pool_cfg.map_h - 8.0)   # 2 tiles east of centre
    dip = nav.centre_path_dip(bank, torch.tensor([m]),
                              _tile_centres(pool_cfg.map_h, pool_cfg.map_w), *_rect(lo, hi), pool_cfg)
    assert torch.equal(dip, torch.zeros_like(dip))


def _one_field(blocked: torch.Tensor, step: tuple[int, int]):
    """A one-map bank whose only field steps every tile by `step`: at a tile, nav.step_dir reads
    just that step and the unit-blocking plane."""
    h, w = blocked.shape
    return SimpleNamespace(
        nav_next=torch.full((1, 1, h, w), nav.OFFSETS.index(step), dtype=torch.uint8),
        nav_offsets=torch.tensor(nav.OFFSETS, dtype=torch.float32),
        blocks_unit=blocked.unsqueeze(0))


def _water_at_9_11():
    cfg = load_config(CONFIGS_DEFAULT, overrides={"world": {"map_h": 20, "map_w": 20}})
    radius = body_radius_tiles(yaml.safe_load(open(CONFIGS_DEFAULT).read()))
    water = torch.zeros((20, 20), dtype=torch.bool)
    water[11, 9] = True
    return cfg, radius, water


def test_a_body_overhanging_a_blocked_corner_straightens_up_before_the_step():
    """nav.step_dir's corner rule, on hand-made fields. Water at (9, 11) lies below the tile
    (9, 10) a bot in (10, 10) steps west into. A body overhanging row 11 first moves north to its
    tile's centre row. It aims at the next tile's centre when it no longer overhangs, when the
    water is gone, or as a centre line (radius 0). A NW step overhanging the same row straightens
    the same way, and one that also overhangs column 11 toward a wall at (11, 9) aims at its own
    tile's centre."""
    cfg, r, water = _water_at_9_11()
    map0 = torch.tensor(0)

    def aim(blocked, step, pos, radius=r):
        return nav.step_dir(_one_field(blocked, step), map0, torch.tensor(pos), 0, cfg, radius)

    def toward(pos, target):
        return geo.normalize(torch.tensor(target) - torch.tensor(pos))

    north = torch.tensor([0.0, -1.0])
    low = (10.11, 10.67)                                   # fy + r = 1.07: in row 11
    assert torch.allclose(aim(water, (-1, 0), low), north)
    assert torch.allclose(aim(water, (-1, 0), low, 0.0), toward(low, (9.5, 10.5)))
    assert torch.allclose(aim(torch.zeros_like(water), (-1, 0), low), toward(low, (9.5, 10.5)))
    level = (10.11, 10.55)                                 # fy + r = 0.95: clear of it
    assert torch.allclose(aim(water, (-1, 0), level), toward(level, (9.5, 10.5)))
    assert torch.allclose(aim(water, (-1, -1), low), north)

    both = water.clone()
    both[9, 11] = True
    corner = (10.75, 10.65)                                # in row 11 and column 11
    assert torch.allclose(aim(both, (-1, -1), corner), toward(corner, (10.5, 10.5)))
    assert torch.allclose(aim(both, (-1, -1), corner, 0.0), toward(corner, (9.5, 9.5)))


def test_a_body_pushed_sideways_past_a_blocked_corner_is_pinned_only_without_the_rule():
    """The 20.6 s pin of 2026-10-07, under terrain.resolve_move: a bot overhanging row 11 steps
    west past the water at (9, 11) while a push south (its strafe, 0.3 against a 1.3-tile seek)
    holds it there. Aimed at the next tile's centre, the push cancels the small northward part of
    the aim and every west step clips the water. Straightened up first, it walks on west."""
    cfg, r, water = _water_at_9_11()
    bank = _one_field(water, (-1, 0))
    map0 = torch.tensor(0)
    tick = 0.13                                            # 2.6 tiles/s at dt 0.05
    strafe = torch.tensor([0.0, 0.3])
    for radius, passes in ((0.0, False), (r, True)):
        pos = torch.tensor([10.11, 10.67])
        for _ in range(40):
            push = nav.step_dir(bank, map0, pos, 0, cfg, radius) * 1.3 + strafe
            pos = terrain.resolve_move(bank.blocks_unit, map0, pos, geo.normalize(push) * tick,
                                       torch.tensor(r), cfg)
        assert (float(pos[0]) < 9.0) is passes, radius


def test_tables_are_absent_without_the_flag_and_rebuild_identically():
    cfg = load_config(CONFIGS_DEFAULT, overrides={"world": {"maps": ["open"]}})
    assert not hasattr(loader.build_map_bank(cfg, "cpu"), "nav_next")
    on = load_config(CONFIGS_DEFAULT, overrides={"world": {"maps": ["open"]}, "bots": {"nav": True}})
    first = loader.build_map_bank(on, "cpu")
    nav._CACHE.clear()
    again = loader.build_map_bank(on, "cpu")
    assert torch.equal(first.nav_next, again.nav_next)
    assert torch.equal(first.nav_anchor_of, again.nav_anchor_of)
    assert torch.equal(first.nav_centre_box, again.nav_centre_box)
