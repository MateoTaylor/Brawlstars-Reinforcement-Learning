"""Bot pathfinding tables, built once per map bank and only under `bots.nav` (user decision,
2026-10-06: "bots should get as real pathfinding as possible without compromising training time
seriously"). Nothing here runs per tick except the gather helpers at the bottom, which
bots/policy.path_toward and the zone terms call.

**Anchors.** One per `bots.nav_anchor_tiles`-square map cell, at the walkable tile nearest the
cell's centre (the pick maps/loader.bush_waypoints makes for bushes); a cell with no walkable tile
gets none. Each anchor owns a FLOW FIELD over the whole map: `nav_next[m, a, y, x]` is the
neighbour code (`OFFSETS`) of the first step of a shortest walkable path from tile (x, y) to
anchor a, and 0 at the anchor itself or where no path exists. The last slot (`centre_slot`) leads
to the map's centre tiles instead. core/zone.py shrinks the safe rect about the map centre, so that
one field is a way out of the gas from anywhere, and it drives the zone terms.

**Paths.** 8-connected: an orthogonal step costs 1, a diagonal one sqrt(2), and a diagonal step
needs the whole 2x2 block it crosses to be walkable, since terrain.resolve_move moves x then y and
a body cutting a wall corner snags on it. Open ground holds many equally short paths; the step
taken is the one nearest the straight bearing to the goal, so a bot crosses open ground on the
straight line, not on a diagonal-then-straight dog-leg.

**Goals.** A goal tile reads the field of `nav_anchor_of[m, y, x]`: of the anchors in its own and
the 8 neighbouring cells, the path-nearest one from whose centre the bots' body walks straight to
the tile's centre (`walk_blocked`, the exact test policy.path_toward makes at run time). A
walkable tile with none, an orphan, is made an anchor itself, its field after the cell anchors';
a tile no body stands on (a wall) takes the nearest anchor by straight distance. So every
walkable goal tile's anchor has that walk clear, and path_toward hands over along it: from the
field's last tile to the goal tile's centre, and from there to the goal. Measured 2026-10-07 on
the pool at the bots' 0.4 radius: choosing by the centre line alone left 3,105 of 102,935
walkable goal tiles with a body-blocked walk from their anchor; by the body's walk, sampled
along each edge, 24, which fell back to anchors with a clear centre line. That fell short twice:
a bot on such an anchor had no walk to take, and the sampled test read a walk clear from one
spot and blocked from the next, a tick's step on, so a bot swapped between the field and the
walk every tick for good (59 goals never reached in a census of every walkable tile). The exact
test leaves 56 orphans on the pool, 6 at most on a map (bush_halo), which adds 4 slots.

**Following.** `step_dir` aims at the next tile's centre, and a centre-to-centre step is always
clear for a body narrower than a tile. A body off its tile's centre is not: it can overhang the
tile's edge toward a blocked tile beside the next one and clip that corner on the way, and since
terrain.resolve_move takes an axis's whole step or none of it, the bot stops one step short of the
corner and stays there. So a body overhanging toward such a tile first straightens up, moving
along that axis to its own tile's centre line. Measured 2026-10-07: a bot closing on a target past
a water corner stood 20.6 s against it, its strafe cancelling the small correction the plain aim
had.

**The centre path's box.** `nav_centre_box[m, y, x]` is the bounding box of the tiles the centre
field's path from (x, y) visits, so `centre_path_inside` can tell a pocket: a tile whose shortest
way to the centre leaves the safe rect. Zone avoidance does not push there. Measured 2026-10-07, a
CAMPER in a pocket at the rect's edge died both ways the push was tried. Following the field, it
was pushed one tile into the gas, where the escape term (scaled by its depth) lost to its pull back
to its bush, and it crossed the edge back and forth until the gas killed it. Aimed straight at the
rect's centre, the push held it against the pocket's inner wall for 7 s, until the gas arrived. A
path that stays inside never passes a pocket tile (each later tile's path is part of it), so the
push never walks a bot to a place where it stops.

The same box gives `centre_path_dip`: how much nearer the gas the path comes than its first tile.
Since the rect is axis-aligned, the box's distance to its edge is exactly the closest the path
comes, so four numbers a tile replace walking the path. The bots' gas rules subtract it from a
clearance (user decision, 2026-10-07), which counts the room a bot has along its way out: a pocket
seals when its exit is gassed, not its back. Measured 2026-10-07, bramble_bend: a CAMPER 2.1 tiles
inside the edge held its bush while its only exit, a tile west, was already gas.

**Cost.** The distance fields come from row sweeps, top-down and bottom-up in turn until a sweep
changes nothing: each row takes its vertical and diagonal steps from the row before and its
horizontal ones from a doubling scan, so one sweep settles every path that never turns back
vertically and an arena needs three or four. Built on CPU in float32 whatever the sim's device, so
a CPU test and a CUDA run read identical tables, and cached per map for the life of the process,
so a run's eval envs reuse the train env's tables. Memory: (M, A+1, H, W) uint8, about 52 MB for
36 maps at 3-tile cells; the centre box adds (M, H, W, 4) int16, 29 KB a map.
"""
import math

import numpy as np
import torch

from ..core import geometry as geo
from ..core import terrain

# Neighbour codes, (dx, dy) with y increasing downward. 0 = no step: the tile is the goal anchor
# itself, unwalkable, or cut off from it.
OFFSETS = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))
_SQRT2 = math.sqrt(2.0)
_INF = float("inf")
# Two candidate steps tie within this. Float32 noise on these sums is ~1e-5; the smallest real gap
# between two path lengths on a 60-tile map is 29*sqrt(2) - 41 = 0.012.
_TIE_TOL = 1e-3
# Sources per chunk when deriving the step codes, which keeps each temporary near 1 MB.
_CHUNK = 64

_CACHE: dict = {}


def segment_blocked(mask: torch.Tensor, map_id: torch.Tensor, p0: torch.Tensor,
                    delta: torch.Tensor, cfg, max_tiles: float) -> torch.Tensor:
    """(...) bool: the segment from `p0` along `delta` (each (..., 2)) enters a tile of `mask`
    (M, H, W), off the map counting as blocked (terrain.sample). Exact: it finds every grid line
    the segment crosses and tests the tile on the far side, so no corner slips between samples
    as one can between terrain.march's. The tile the segment starts in is not tested, so a tail
    of a clear segment is clear: a body walking down a clear line stays on one.

    Only the first ceil(max_tiles) + 2 lines along each axis are found (a Python scalar: it
    sizes a tensor dimension), which covers every segment up to `max_tiles` long."""
    k = torch.arange(int(math.ceil(max_tiles)) + 2, device=p0.device, dtype=p0.dtype)
    tiles, valid = [], []
    for a in (0, 1):
        o = 1 - a
        s, d = p0[..., a:a + 1], delta[..., a:a + 1]                      # (..., 1)
        up = d > 0
        # Nearest first. Going down, floor(s) itself: a segment starting on a line crosses it at 0.
        line = torch.where(up, torch.floor(s) + 1.0 + k, torch.floor(s) - k)     # (..., n)
        t = (line - s) / torch.where(d == 0, torch.ones_like(d), d)
        valid.append((d != 0) & (t >= 0) & (t <= 1))
        past = torch.where(up, line, line - 1.0)                           # the tile beyond it
        other = p0[..., o:o + 1] + t * delta[..., o:o + 1]
        beside = torch.where(delta[..., o:o + 1] < 0, torch.ceil(other) - 1.0, torch.floor(other))
        tiles.append(torch.stack((past, beside) if a == 0 else (beside, past), dim=-1))
    centres = torch.cat(tiles, dim=-2) + 0.5                               # (..., 2n, 2)
    return (terrain.sample(mask, map_id, centres, cfg) & torch.cat(valid, dim=-1)).any(dim=-1)


def walk_blocked(blocks_unit: torch.Tensor, map_id: torch.Tensor, p0: torch.Tensor,
                 delta: torch.Tensor, radius, cfg, max_tiles: float) -> torch.Tensor:
    """bool shaped like `delta[..., 0]`: a body of `radius` walking straight from `p0` along the
    whole of `delta` meets a unit-blocking tile. The test is `segment_blocked` along the body's
    two edges, the lines `radius` to either side of the centre line. Two lines are enough while
    the body is narrower than a tile (config.validate): no tile fits between them without
    crossing one. The centre line alone is not: it passes wall corners the body cannot, and a
    push running along a wall face leaves terrain.resolve_move no axis to slide on.

    Exact, where a march's samples can step over a wall corner: measured 2026-10-07, a bot whose
    sampled walk read clear from one spot and blocked from the next, a tick's step on, swapped
    between that walk and the field every tick for good (policy.path_toward). And a tail of a
    clear walk is clear, so a body that takes a clear walk keeps it to the end.

    `radius` scales the (..., 2) side offset: a float, or a tensor that broadcasts against it.
    0 tests the centre line (twice). A walk longer than `max_tiles` is tested only that far, so
    a caller ignores the answer past it."""
    side = geo.perp(geo.normalize(delta)) * radius
    edges = torch.stack([p0 + side, p0 - side], dim=-2)                   # (..., 2, 2)
    return segment_blocked(blocks_unit, map_id, edges, delta.unsqueeze(-2).expand_as(edges),
                           cfg, max_tiles).any(dim=-1)


def anchors(free: np.ndarray, cell_tiles: int) -> np.ndarray:
    """(A, 2) int64 (x, y) tiles, one per `cell_tiles`-square cell that holds a walkable tile: the
    one nearest the cell's centre (first in raster order on a tie). Cells in raster order."""
    h, w = free.shape
    picks = []
    for cy0 in range(0, h, cell_tiles):
        for cx0 in range(0, w, cell_tiles):
            cell = free[cy0:cy0 + cell_tiles, cx0:cx0 + cell_tiles]
            ys, xs = np.nonzero(cell)
            if len(ys) == 0:
                continue
            mid_y, mid_x = cell.shape[0] / 2.0, cell.shape[1] / 2.0
            best = np.argmin((xs + 0.5 - mid_x) ** 2 + (ys + 0.5 - mid_y) ** 2)
            picks.append((cx0 + xs[best], cy0 + ys[best]))
    return np.asarray(picks, dtype=np.int64).reshape(-1, 2)


def centre_tiles(free: np.ndarray) -> np.ndarray:
    """(K, 2) int64 (x, y): the walkable tiles whose centres lie within 1 tile of the map centre,
    the radius widened a tile at a time until one qualifies."""
    if not free.any():
        raise ValueError("map has no walkable tile")
    h, w = free.shape
    ys, xs = np.nonzero(free)
    d = np.hypot(xs + 0.5 - w / 2.0, ys + 0.5 - h / 2.0)
    radius = 1.0
    while not (d <= radius).any():
        radius += 1.0
    keep = d <= radius
    return np.stack([xs[keep], ys[keep]], axis=1).astype(np.int64)


def _row_tables(free: torch.Tensor):
    """Step costs for `distance_fields` (inf where a step is illegal).

    vert[j] (W,): the vertical step between rows j and j+1, per column. diag[j] (W-1,): either
    diagonal across the 2x2 block at columns (i, i+1) of rows (j, j+1), legal only when all four
    tiles are walkable. jumps[y]: per row, the (k, cost) pairs of a doubling scan, a run of k legal
    horizontal steps costing k, rightward then leftward."""
    h, w = free.shape
    pair = free[1:] & free[:-1]
    vert = torch.where(pair, 1.0, _INF)
    diag = torch.where(pair[:, 1:] & pair[:, :-1], _SQRT2, _INF)

    step = free[:, 1:] & free[:, :-1]                    # (H, W-1): x <-> x+1 legal
    run_right = torch.zeros((h, w), dtype=torch.int64)   # legal steps in a row ending at x from the left
    run_left = torch.zeros((h, w), dtype=torch.int64)    # ... from the right
    for x in range(1, w):
        run_right[:, x] = torch.where(step[:, x - 1], run_right[:, x - 1] + 1, 0)
    for x in range(w - 2, -1, -1):
        run_left[:, x] = torch.where(step[:, x], run_left[:, x + 1] + 1, 0)

    ks = [1 << i for i in range((w - 1).bit_length())]   # 1, 2, 4, ... : every run up to w-1
    jumps = []
    for y in range(h):
        right = [(k, torch.where(run_right[y, k:] >= k, float(k), _INF)) for k in ks]
        left = [(k, torch.where(run_left[y, :-k] >= k, float(k), _INF)) for k in ks]
        jumps.append((right, left))
    return vert, diag, jumps


def _close_row(row: torch.Tensor, right, left) -> None:
    """In place: every cell of `row` (S, W) takes the best value over the straight horizontal runs
    reaching it. After jump k a cell has seen every run of up to 2k - 1 steps from that side."""
    for k, cost in right:
        row[:, k:] = torch.minimum(row[:, k:], row[:, :-k] + cost)
    for k, cost in left:
        row[:, :-k] = torch.minimum(row[:, :-k], row[:, k:] + cost)


def distance_fields(free: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """(S, H, W) float32 shortest-path lengths to each source mask of `sources` (S, H, W) bool,
    over the walkable tiles `free` (H, W) bool. inf off the walkable region and where cut off.

    A sweep is idempotent (each row is final once processed), so a sweep that changes nothing
    right after a sweep the other way means the field is a fixed point of both: converged."""
    h, _ = free.shape
    vert, diag, jumps = _row_tables(free)
    dist = torch.where(sources & free, 0.0, _INF)
    for y in range(h):
        _close_row(dist[:, y], *jumps[y])

    down, first = True, True
    while True:
        before = dist.clone()
        for y in (range(1, h) if down else range(h - 2, -1, -1)):
            j = y - 1 if down else y                  # the pair of rows (j, j+1) being crossed
            prev = dist[:, y - 1] if down else dist[:, y + 1]
            row = dist[:, y]
            row.copy_(torch.minimum(row, prev + vert[j]))
            row[:, 1:] = torch.minimum(row[:, 1:], prev[:, :-1] + diag[j])    # from column x-1
            row[:, :-1] = torch.minimum(row[:, :-1], prev[:, 1:] + diag[j])   # from column x+1
            _close_row(row, *jumps[y])
        if not first and torch.equal(before, dist):
            return dist
        first, down = False, not down


def step_codes(free: torch.Tensor, dist: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    """(S, H, W) uint8 neighbour codes: from each tile, the legal step whose neighbour lies on a
    shortest path to source s (dist[s] + step cost within _TIE_TOL of the best), the one nearest
    the bearing to `goals[s]` (S, 2) (x, y) on a tie. 0 at a source tile and wherever dist is inf."""
    s_total, h, w = dist.shape
    padded = torch.nn.functional.pad(dist, (1, 1, 1, 1), value=_INF)
    free_pad = torch.nn.functional.pad(free, (1, 1, 1, 1), value=False)
    legal = []
    for dx, dy in OFFSETS[1:]:
        ok = free & free_pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        if dx and dy:
            ok = ok & free_pad[1:1 + h, 1 + dx:1 + dx + w] & free_pad[1 + dy:1 + dy + h, 1:1 + w]
        legal.append(ok)
    cy, cx = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")

    codes = torch.zeros((s_total, h, w), dtype=torch.uint8)
    for s0 in range(0, s_total, _CHUNK):
        s1 = min(s0 + _CHUNK, s_total)
        cand = []
        for k, (dx, dy) in enumerate(OFFSETS[1:]):
            cost = _SQRT2 if (dx and dy) else 1.0
            shifted = padded[s0:s1, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            cand.append(torch.where(legal[k], shifted + cost, _INF))
        best = torch.stack(cand).amin(dim=0)
        bx = goals[s0:s1, 0].view(-1, 1, 1) - cx
        by = goals[s0:s1, 1].view(-1, 1, 1) - cy
        norm = torch.clamp(torch.hypot(bx, by), min=1e-6)
        top = torch.full_like(best, -_INF)
        code = torch.zeros((s1 - s0, h, w), dtype=torch.uint8)
        for k, (dx, dy) in enumerate(OFFSETS[1:]):
            align = (dx * bx + dy * by) / (norm * math.hypot(dx, dy))
            score = torch.where(cand[k] <= best + _TIE_TOL, align, -_INF)
            better = score > top
            top = torch.where(better, score, top)
            code = torch.where(better, k + 1, code).to(torch.uint8)
        arrived = dist[s0:s1] <= 0.0
        codes[s0:s1] = torch.where(arrived | ~torch.isfinite(best), 0, code).to(torch.uint8)
    return codes


def anchor_of(free: torch.Tensor, anchor_xy: np.ndarray, dist: torch.Tensor, cell_tiles: int,
              cfg, body_radius: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """((H, W) int64 goal-tile -> anchor index, (H, W) bool orphans), chosen as the module
    docstring says. `dist` is the anchors' own (A, H, W) fields; the walks are tested over the
    unit-blocking plane with `walk_blocked` at `body_radius`. An orphan is a walkable tile no
    neighbouring anchor's walk reaches, which _one_map makes an anchor of its own; its entry here
    is a placeholder."""
    h, w = free.shape
    n_anchor = len(anchor_xy)
    ax = torch.as_tensor(anchor_xy[:, 0])
    ay = torch.as_tensor(anchor_xy[:, 1])
    cells_y, cells_x = -(-h // cell_tiles), -(-w // cell_tiles)
    cell_anchor = torch.full((cells_y, cells_x), -1, dtype=torch.int64)
    cell_anchor[ay // cell_tiles, ax // cell_tiles] = torch.arange(n_anchor)

    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    tcy, tcx = ys // cell_tiles, xs // cell_tiles
    cand = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ny, nx = tcy + dy, tcx + dx
            inside = (ny >= 0) & (ny < cells_y) & (nx >= 0) & (nx < cells_x)
            idx = cell_anchor[ny.clamp(0, cells_y - 1), nx.clamp(0, cells_x - 1)]
            cand.append(torch.where(inside, idx, -1))
    cand = torch.stack(cand)                                              # (9, H, W)
    valid = cand >= 0
    safe = cand.clamp(min=0)
    path = dist[safe, ys, xs]                                             # (9, H, W)

    start = torch.stack([ax[safe] + 0.5, ay[safe] + 0.5], dim=-1).to(torch.float32)
    goal = torch.stack([xs + 0.5, ys + 0.5], dim=-1).to(torch.float32).expand_as(start)
    reach = math.sqrt(2.0) * (2 * cell_tiles - 1) + 1.0   # bounds every neighbour-cell pair
    hit = walk_blocked((~free).unsqueeze(0), torch.zeros((), dtype=torch.int64), start,
                       goal - start, body_radius, cfg, max_tiles=reach)

    best, pick = torch.where(valid & ~hit, path, _INF).min(dim=0)
    reached = torch.isfinite(best)
    eu = (ax.view(-1, 1, 1) + 0.5 - (xs + 0.5)) ** 2 + (ay.view(-1, 1, 1) + 0.5 - (ys + 0.5)) ** 2
    out = torch.where(reached, torch.gather(cand, 0, pick.unsqueeze(0)).squeeze(0), eu.argmin(dim=0))
    return out, free & ~reached


def path_box(codes: torch.Tensor) -> torch.Tensor:
    """(H, W, 4) int16 (x_min, y_min, x_max, y_max): the bounding box of the tiles one field's
    path (`codes` (H, W)) visits from each tile, that tile included; the tile alone where the
    field gives no step. Pointer doubling: after round k each tile's box covers the next 2**k
    tiles of its path, and no path revisits a tile, since every step shortens the distance."""
    h, w = codes.shape
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    off = torch.as_tensor(OFFSETS, dtype=torch.int64)[codes.long()]
    nxt = ((ys + off[..., 1]) * w + xs + off[..., 0]).reshape(-1)
    lo = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
    hi = lo.clone()
    for _ in range((h * w).bit_length()):
        lo = torch.minimum(lo, lo[nxt])
        hi = torch.maximum(hi, hi[nxt])
        nxt = nxt[nxt]
    return torch.cat([lo, hi], dim=-1).reshape(h, w, 4).to(torch.int16)


def _sources(h: int, w: int, xy: np.ndarray) -> torch.Tensor:
    """(K, H, W) bool, one source tile per row of `xy` (K, 2) (x, y)."""
    out = torch.zeros((len(xy), h, w), dtype=torch.bool)
    out[torch.arange(len(xy)), torch.as_tensor(xy[:, 1]), torch.as_tensor(xy[:, 0])] = True
    return out


def _one_map(free_np: np.ndarray, cell_tiles: int, cfg, body_radius: float):
    """(codes (A+1, H, W) uint8, anchor_of (H, W) int64, A, centre box (H, W, 4) int16, anchor
    tiles (A,) int64 as y * W + x) for one map, the centre field last. The cell anchors come
    first, then the orphans `anchor_of` returns, each an anchor of its own."""
    key = (free_np.shape, free_np.tobytes(), cell_tiles, float(body_radius))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    h, w = free_np.shape
    free = torch.as_tensor(free_np)
    anchor_xy = anchors(free_np, cell_tiles)
    n_cell = len(anchor_xy)
    centre = torch.zeros((1, h, w), dtype=torch.bool)
    centre_xy = centre_tiles(free_np)
    centre[0, torch.as_tensor(centre_xy[:, 1]), torch.as_tensor(centre_xy[:, 0])] = True

    dist = distance_fields(free, torch.cat([_sources(h, w, anchor_xy), centre]))
    aof, orphan = anchor_of(free, anchor_xy, dist[:n_cell], cell_tiles, cfg, body_radius)
    oy, ox = torch.nonzero(orphan, as_tuple=True)
    if len(ox):
        own = np.stack([ox.numpy(), oy.numpy()], axis=1).astype(np.int64)
        dist = torch.cat([dist[:n_cell], distance_fields(free, _sources(h, w, own)), dist[n_cell:]])
        aof[oy, ox] = n_cell + torch.arange(len(ox))
        anchor_xy = np.concatenate([anchor_xy, own])
    n_anchor = len(anchor_xy)

    goals = torch.cat([
        torch.as_tensor(anchor_xy, dtype=torch.float32) + 0.5,
        torch.tensor([[w / 2.0, h / 2.0]], dtype=torch.float32),
    ])
    codes = step_codes(free, dist, goals)
    tiles = torch.as_tensor(anchor_xy[:, 1] * w + anchor_xy[:, 0], dtype=torch.int64)
    out = (codes, aof, n_anchor, path_box(codes[n_anchor]), tiles)
    _CACHE[key] = out
    return out


def build_tables(blocks_unit: torch.Tensor, cfg, device, body_radius: float = 0.0):
    """(nav_next (M, A+1, H, W) uint8, nav_anchor_of (M, H, W) int64, nav_anchor_tile (M, A+1)
    int64, nav_offsets (9, 2) f32, nav_centre_box (M, H, W, 4) int16), all on `device`, from a
    bank's (M, H, W) unit-blocking plane. A is the largest anchor count in the bank; a map with
    fewer leaves its spare slots all-zero (never read: nav_anchor_of names only real anchors).
    Slot A is the centre field on every map, and nav_centre_box its `path_box`. nav_anchor_tile
    is the tile (y * W + x) each slot's field ends on, -1 for the centre field and spare slots.
    `body_radius` is the bots' body (config.body_radius_tiles), which the anchor walks must
    clear; 0 tests the centre line."""
    blocked = blocks_unit.detach().to("cpu").numpy()
    per_map = [_one_map(~blocked[m], cfg.bots_nav_anchor_tiles, cfg, body_radius)
               for m in range(blocked.shape[0])]
    n_slots = max(one[2] for one in per_map) + 1
    m_count, h, w = blocked.shape
    nav_next = torch.zeros((m_count, n_slots, h, w), dtype=torch.uint8)
    nav_anchor_tile = torch.full((m_count, n_slots), -1, dtype=torch.int64)
    for m, (codes, _, n, _, tiles) in enumerate(per_map):
        nav_next[m, :n] = codes[:n]
        nav_next[m, n_slots - 1] = codes[n]
        nav_anchor_tile[m, :n] = tiles
    nav_anchor_of = torch.stack([one[1] for one in per_map])
    nav_offsets = torch.tensor(OFFSETS, dtype=torch.float32)
    nav_centre_box = torch.stack([one[3] for one in per_map])
    return (nav_next.to(device), nav_anchor_of.to(device), nav_anchor_tile.to(device),
            nav_offsets.to(device), nav_centre_box.to(device))


def centre_slot(bank) -> int:
    """The centre field's slot in `bank.nav_next`."""
    return bank.nav_next.shape[1] - 1


# ---------------------------------------------------------------------------------------------
# per-tick reads (bots/policy)
# ---------------------------------------------------------------------------------------------

def _tile(pos: torch.Tensor, cfg):
    ix, iy = terrain.to_tile(pos)
    return ix.clamp(0, cfg.map_w - 1), iy.clamp(0, cfg.map_h - 1)


def goal_slot(bank, map_id: torch.Tensor, goal: torch.Tensor, cfg) -> torch.Tensor:
    """(...) int64: the field slot that leads to `goal` (..., 2), via nav_anchor_of."""
    ix, iy = _tile(goal, cfg)
    return bank.nav_anchor_of[terrain._broadcast_map_id(map_id, ix.shape), iy, ix]


def tile_centre(pos: torch.Tensor, cfg) -> torch.Tensor:
    """(..., 2) the centre of the map tile `pos` (..., 2) lies in, the nearest edge tile's off
    the map."""
    ix, iy = _tile(pos, cfg)
    return torch.stack([ix, iy], dim=-1).to(pos.dtype) + 0.5


def at_anchor(bank, map_id: torch.Tensor, pos: torch.Tensor, slot, cfg) -> torch.Tensor:
    """(...) bool: `pos` (..., 2) is on the tile field `slot` (shaped like pos[..., 0]) ends on,
    its anchor's. Never on the centre field, which ends on several."""
    ix, iy = _tile(pos, cfg)
    end = bank.nav_anchor_tile[terrain._broadcast_map_id(map_id, ix.shape), slot]
    return end == iy * cfg.map_w + ix


def centre_path_inside(bank, map_id: torch.Tensor, pos: torch.Tensor, zone_lo: torch.Tensor,
                       zone_hi: torch.Tensor, cfg) -> torch.Tensor:
    """(...) bool: the centre of every tile the centre field's path from `pos`'s tile visits lies
    inside the rect [zone_lo, zone_hi] (each (..., 2), broadcasting against `pos`)."""
    ix, iy = _tile(pos, cfg)
    box = bank.nav_centre_box[terrain._broadcast_map_id(map_id, ix.shape), iy, ix].to(pos.dtype) + 0.5
    return ((box[..., 0] >= zone_lo[..., 0]) & (box[..., 1] >= zone_lo[..., 1])
            & (box[..., 2] <= zone_hi[..., 0]) & (box[..., 3] <= zone_hi[..., 1]))


def rect_centred(zone_lo: torch.Tensor, zone_hi: torch.Tensor, cfg) -> torch.Tensor:
    """(...) bool: the rect [zone_lo, zone_hi] is centred on the map to within a tile, as every
    rect core/zone.py shrinks is, so the centre field is its way out. Only a test that moves the
    rect by hand makes one that is not."""
    map_centre = geo.vec2(cfg.map_w / 2.0, cfg.map_h / 2.0, zone_lo.device, zone_lo.dtype)
    return geo.safe_norm((zone_lo + zone_hi) / 2.0 - map_centre, dim=-1) <= 1.0


def _room(lo: torch.Tensor, hi: torch.Tensor, zone_lo: torch.Tensor, zone_hi: torch.Tensor):
    """(...) the box [lo, hi]'s least distance to the edge of the rect [zone_lo, zone_hi],
    measured from inside, negative where the box crosses the edge. A point is the box lo == hi."""
    per_axis = torch.minimum(lo - zone_lo, zone_hi - hi)
    return torch.minimum(per_axis[..., 0], per_axis[..., 1])


def centre_path_dip(bank, map_id: torch.Tensor, pos: torch.Tensor, zone_lo: torch.Tensor,
                    zone_hi: torch.Tensor, cfg) -> torch.Tensor:
    """(...) tiles >= 0: how much nearer the edge of the rect [zone_lo, zone_hi] (each (..., 2),
    broadcasting against `pos`) the centre field's path from `pos`'s tile comes than that tile's
    own centre does. A gas rule that subtracts it from a clearance measures the clearance along
    the way out, which is what a pocket's exit decides (the module docstring's centre box).

    Exactly 0 wherever the path comes no nearer the edge than its first tile, so a rule that
    subtracts it is unchanged there, and 0 where the rect is not centred (`rect_centred`): the
    centre field is not the way out of one."""
    ix, iy = _tile(pos, cfg)
    box = bank.nav_centre_box[terrain._broadcast_map_id(map_id, ix.shape), iy, ix].to(pos.dtype) + 0.5
    here = torch.stack([ix, iy], dim=-1).to(pos.dtype) + 0.5
    dip = _room(here, here, zone_lo, zone_hi) - _room(box[..., :2], box[..., 2:], zone_lo, zone_hi)
    return torch.where(rect_centred(zone_lo, zone_hi, cfg), dip, torch.zeros_like(dip))


def _corner_clip(bank, map_id: torch.Tensor, here: torch.Tensor, off: torch.Tensor,
                 pos: torch.Tensor, radius, cfg):
    """((...) bool, (...) bool): a body of `radius` at `pos`, stepping from its tile (ix, iy),
    centre `here`, by `off` = (dx, dy) to the next tile's centre, overhangs its tile toward a
    unit-blocking tile it could clip on the way: (ix+dx, iy-1) or (ix+dx, iy+1), above or below
    where the step's x part leads (`row`), or (ix-1, iy+dy) or (ix+1, iy+dy), left or right of
    where its y part leads (`col`). Those four are the only tiles the walk can bring a 9-probe body
    (terrain.circle_blocked) into that it does not already touch, while radius < 0.5. Off the map
    counts as blocked, as in terrain.sample."""
    dx, dy = off[..., 0], off[..., 1]
    fx, fy = pos[..., 0] - here[..., 0] + 0.5, pos[..., 1] - here[..., 1] + 0.5
    cx, cy = here[..., 0], here[..., 1]
    beside = torch.stack([
        torch.stack([cx + dx, cy - 1.0], dim=-1), torch.stack([cx + dx, cy + 1.0], dim=-1),
        torch.stack([cx - 1.0, cy + dy], dim=-1), torch.stack([cx + 1.0, cy + dy], dim=-1),
    ], dim=-2)                                                            # (..., 4, 2)
    over = torch.stack([(dx != 0) & (fy < radius), (dx != 0) & (fy + radius >= 1.0),
                        (dy != 0) & (fx < radius), (dy != 0) & (fx + radius >= 1.0)], dim=-1)
    hit = over & terrain.sample(bank.blocks_unit, map_id, beside, cfg)
    return hit[..., 0] | hit[..., 1], hit[..., 2] | hit[..., 3]


def step_dir(bank, map_id: torch.Tensor, pos: torch.Tensor, slot, cfg, radius=0.0) -> torch.Tensor:
    """(..., 2) unit direction from `pos` to the centre of the next tile on field `slot`'s path
    (an int, or a tensor shaped like pos[..., 0]); zero where the field gives no step.

    Given a body `radius` (a float, or a tensor broadcasting against pos[..., 0]), a body that
    overhangs its tile toward a blocked tile beside the next one (`_corner_clip`) first moves to
    its own tile's centre line across that edge: along y where the step's x part would clip,
    along x where its y part would, to the tile's centre where both would (the module
    docstring's Following). Each of those moves only re-enters tiles the body already touches."""
    ix, iy = _tile(pos, cfg)
    code = bank.nav_next[terrain._broadcast_map_id(map_id, ix.shape), slot, iy, ix].long()
    off = bank.nav_offsets[code]
    here = torch.stack([ix, iy], dim=-1).to(pos.dtype) + 0.5
    target = here + off
    if torch.is_tensor(radius) or radius > 0:
        row, col = _corner_clip(bank, map_id, here, off, pos, radius, cfg)
        tx = torch.where(col, here[..., 0], torch.where(row, pos[..., 0], target[..., 0]))
        ty = torch.where(row, here[..., 1], torch.where(col, pos[..., 1], target[..., 1]))
        target = torch.stack([tx, ty], dim=-1)
    direction = geo.normalize(target - pos)
    return torch.where((code > 0).unsqueeze(-1), direction, torch.zeros_like(direction))
