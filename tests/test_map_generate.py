"""Pins brawl_sim/maps/generate.py: the grid primitives, every
stamp's size and bounds, the repair pass on hand-built grids, the checks, and `generate` itself --
bands, symmetry, loader validity and byte-for-byte determinism per family."""
import random
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from brawl_sim.maps import generate as g
from brawl_sim.maps import loader


def _connected4(cells) -> bool:
    cells = set(cells)
    start = next(iter(cells))
    seen = {start}
    dq = deque([start])
    while dq:
        r, c = dq.popleft()
        for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if n in cells and n not in seen:
                seen.add(n)
                dq.append(n)
    return seen == cells


# ---- primitives --------------------------------------------------------------------------------


def test_mirror_helpers_are_involutions_and_swap_the_corners():
    assert g.mirror_point((0, 0)) == (59, 59)
    assert g.mirror_point((1, 58)) == (58, 1)
    assert g.mirror_lr((3, 4)) == (3, 55)
    for cell in ((0, 0), (7, 41), (29, 30), (58, 1)):
        assert g.mirror_point(g.mirror_point(cell)) == cell
        assert g.mirror_lr(g.mirror_lr(cell)) == cell
    with pytest.raises(ValueError, match="symmetry"):
        g.mirror_fn("diagonal")


def test_new_grid_is_a_wall_ring_around_floor():
    grid = g.new_grid()
    assert grid.shape == (60, 60)
    assert (grid[0] == "#").all() and (grid[-1] == "#").all()
    assert (grid[:, 0] == "#").all() and (grid[:, -1] == "#").all()
    assert (grid[1:-1, 1:-1] == ".").all()


def test_stamp_writes_the_mirror_too_and_never_touches_the_border():
    grid = g.new_grid()
    g.stamp(grid, {(5, 7): "#", (0, 3): "~", (1, 1): "b"}, g.POINT)
    assert grid[5, 7] == "#" and grid[54, 52] == "#"
    assert grid[1, 1] == "b" and grid[58, 58] == "b"
    assert grid[0, 3] == "#" and grid[59, 56] == "#", "the border cell and its mirror are untouched"
    grid = g.new_grid()
    g.stamp(grid, {(5, 7): "#"}, None)
    assert grid[5, 7] == "#" and grid[54, 52] == "."


def test_moat_is_a_two_wide_point_symmetric_ring_with_four_three_tile_gaps():
    grid = g.new_grid()
    g.add_moat(grid, random.Random(3), inset=4, symmetry=g.POINT)
    ring = np.zeros((60, 60), dtype=bool)
    ring[4:6, 4:56] = True
    ring[54:56, 4:56] = True
    ring[4:56, 4:6] = True
    ring[4:56, 54:56] = True
    assert (grid[~ring] == ".").sum() == (~ring).sum() - 4 * 60 + 4, "nothing outside the ring but the border"
    gaps = (grid[ring] == ".").sum()
    assert gaps == 4 * g.MOAT_GAP * 2, "four gaps, three tiles wide, through both rows of water"
    assert np.array_equal(grid, g.mirror_grid(grid, g.POINT))
    # one gap per side: floor in the top rows, bottom rows, left cols and right cols of the ring
    assert (grid[4:6, 6:54] == ".").any() and (grid[54:56, 6:54] == ".").any()
    assert (grid[6:54, 4:6] == ".").any() and (grid[6:54, 54:56] == ".").any()


def test_moat_under_mirror_symmetry_still_has_a_gap_on_every_side():
    grid = g.new_grid()
    g.add_moat(grid, random.Random(3), inset=4, symmetry=g.MIRROR)
    assert np.array_equal(grid, g.mirror_grid(grid, g.MIRROR))
    assert (grid[4:6, 6:54] == ".").any() and (grid[54:56, 6:54] == ".").any()
    assert (grid[6:54, 4:6] == ".").any() and (grid[6:54, 54:56] == ".").any()


# ---- stamps ------------------------------------------------------------------------------------


def test_wall_cluster_sizes_bounds_and_skirt():
    for seed in range(200):
        cells = g.wall_cluster(random.Random(seed), (20, 20))
        walls = [c for c, ch in cells.items() if ch == "#"]
        bush = [c for c, ch in cells.items() if ch == "b"]
        assert 4 <= len(walls) <= 24 + 6, seed
        assert len(bush) >= 2, "a skirt on at least one side"
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        assert max(rows) - min(rows) + 1 <= 10 and max(cols) - min(cols) + 1 <= 10, seed
        assert _connected4(walls), seed
        assert all(any(abs(r - wr) + abs(c - wc) == 1 for wr, wc in walls) for r, c in bush), \
            "every skirt cell is 4-adjacent to the wall"


def test_pond_is_a_filled_ellipse_with_a_bush_rim_on_part_of_its_edge():
    cells = g.pond(random.Random(0), (30, 30), radius=3)
    water = [c for c, ch in cells.items() if ch == "~"]
    bush = [c for c, ch in cells.items() if ch == "b"]
    assert 25 <= len(water) <= 49
    assert all(max(abs(r - 30), abs(c - 30)) <= 3 for r, c in water)
    assert _connected4(water)
    assert bush and all(any(abs(r - wr) + abs(c - wc) == 1 for wr, wc in water) for r, c in bush)
    edge = {(r + dr, c + dc) for r, c in water for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))} - set(water)
    assert 0.3 * len(edge) <= len(bush) <= 0.7 * len(edge), "about half the perimeter"
    assert not [c for c, ch in g.pond(random.Random(0), (30, 30), radius=3, rim=False).items() if ch == "b"]


def test_channel_is_a_connected_water_strip_of_bounded_size():
    for seed in range(100):
        cells = g.channel(random.Random(seed), (20, 20))
        assert all(ch == "~" for ch in cells.values())
        assert 6 <= len(cells) <= 2 * 12, seed
        assert _connected4(cells), seed


def test_bush_patch_grows_to_the_asked_size_on_open_ground_and_stops_when_boxed_in():
    grid = g.new_grid()
    free = lambda cell: grid[cell] == "."  # noqa: E731
    cells = g.bush_patch(random.Random(1), (30, 30), free, size=20)
    assert len(cells) == 20 and set(cells.values()) == {"b"} and _connected4(cells)
    boxed = g.bush_patch(random.Random(1), (30, 30), lambda cell: cell == (30, 30), size=20)
    assert set(boxed) == {(30, 30)}
    assert g.bush_patch(random.Random(1), (30, 30), lambda cell: False, size=20) == {}
    for seed in range(50):
        n = len(g.bush_patch(random.Random(seed), (30, 30), free))
        assert 6 <= n <= 30, seed


def test_crate_spot_is_one_or_two_adjacent_boxes():
    for seed in range(50):
        cells = g.crate_spot(random.Random(seed), (10, 10), lambda cell: True)
        assert set(cells.values()) == {"X"} and 1 <= len(cells) <= 2
        if len(cells) == 2:
            (a, b) = cells
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    assert g.crate_spot(random.Random(0), (10, 10), lambda cell: False) == {(10, 10): "X"}


def test_centre_features_are_all_point_symmetric():
    seen = set()
    for seed in range(40):
        cells, reserved = g.centre_feature(random.Random(seed))
        chars = set(cells.values())
        if chars == {"#", "b"}:
            seen.add("wall")
            assert sum(ch == "#" for ch in cells.values()) == 4 and sum(ch == "b" for ch in cells.values()) == 4
        elif chars == {"~"}:
            seen.add("pond")
            assert 12 <= len(cells) <= 32
        else:
            seen.add("cross")
            assert not cells and len(reserved) > 40
        for cell in list(cells) + list(reserved):
            assert g.mirror_point(cell) in (cells if cell in cells else reserved)
    assert seen == {"wall", "pond", "cross"}


# ---- repair ------------------------------------------------------------------------------------


def test_cap_wall_runs_punches_gaps_until_no_straight_run_exceeds_eight():
    grid = g.new_grid()
    grid[20, 10:22] = "#"  # a 12-run
    assert g.cap_wall_runs(grid, None) >= 1
    assert max(len(run) for run in g._wall_runs(grid)) <= g.MAX_WALL_RUN
    assert (grid[20, 10:22] == ".").sum() >= 2, "a 2-tile gap"


def test_cap_wall_runs_keeps_point_symmetry():
    grid = g.new_grid()
    g.stamp(grid, {(20, c): "#" for c in range(10, 22)}, g.POINT)
    g.cap_wall_runs(grid, g.POINT)
    assert np.array_equal(grid, g.mirror_grid(grid, g.POINT))
    assert max(len(run) for run in g._wall_runs(grid)) <= g.MAX_WALL_RUN


def test_fill_dead_ends_fills_a_one_wide_corridor_with_what_blocks_it():
    grid = g.new_grid()
    grid[10, 10:17] = "#"
    grid[12, 10:17] = "#"
    grid[11, 16] = "#"     # (11, 10..15) is a 1-wide corridor open only at (11, 9)
    assert g.fill_dead_ends(grid) == 6
    assert (grid[11, 10:16] == "#").all()
    assert grid[11, 9] == "."
    grid = g.new_grid()
    grid[10, 10:17] = "~"
    grid[12, 10:17] = "~"
    grid[11, 16] = "~"
    g.fill_dead_ends(grid)
    assert (grid[11, 10:16] == "~").all(), "a water-lined dead end fills with water"


def test_connect_components_carves_the_cheapest_path_between_the_two_largest():
    grid = g.new_grid()
    grid[1:59, 30] = "~"   # water is never run-capped, so this genuinely splits the map
    passable = np.isin(grid, list(g.PASSABLE))
    assert len(g._components(passable)) == 2
    carved = g.connect_components(grid, None)
    assert carved == 1, "one water cell is all it takes"
    assert len(g._components(np.isin(grid, list(g.PASSABLE)))) == 1


def test_repair_on_a_grid_with_a_long_run_and_two_components_gives_one_component_and_short_runs():
    grid = g.new_grid()
    grid[1:59, 30] = "#"   # a 58-run that also splits the map
    grid[20, 5:17] = "#"   # and a 12-run
    stats = g.repair(grid, None)
    assert stats["gaps_punched"] >= 2
    assert len(g._components(np.isin(grid, list(g.PASSABLE)))) == 1
    assert max(len(run) for run in g._wall_runs(grid)) <= g.MAX_WALL_RUN


# ---- checks ------------------------------------------------------------------------------------


def test_reach_counts_on_an_open_field_and_pocket_fractions_scale_to_the_border():
    grid = g.new_grid()
    passable = np.isin(grid, list(g.PASSABLE))
    counts = g.reach_counts(passable)
    assert counts[30, 30] == g.POCKET_OPEN_FIELD
    assert counts[1, 1] == 28, "a corner cell's quadrant within radius 6"
    assert counts[0, 0] == 0, "the border is not passable"
    frac = g.pocket_fractions(grid)
    assert np.allclose(frac, 1.0), "an open field is nowhere a pocket, corners included"


def test_pocket_fractions_flag_a_dead_end_lane_but_not_a_two_wide_lane_between_features():
    grid = g.new_grid()
    grid[20, 20:32] = "#"
    grid[24, 20:32] = "#"
    grid[21:24, 31] = "#"  # a 3-wide, 11-deep dead end lane, mouth at col 20
    frac = g.pocket_fractions(grid)
    assert frac[22, 30] < g.POCKET_MIN_REACH / g.POCKET_OPEN_FIELD
    grid = g.new_grid()
    grid[20, 20:32] = "#"
    grid[23, 20:32] = "#"  # a 2-wide lane open at both ends, the shape a margin-2 layout makes
    frac = g.pocket_fractions(grid)
    assert frac[21, 25] >= g.POCKET_MIN_REACH / g.POCKET_OPEN_FIELD


def test_bush_spread_counts_hunt_cells_with_any_bush():
    grid = g.new_grid()
    assert g.bush_spread(grid) == 0.0
    for r0 in range(0, 60, 10):
        for c0 in range(0, 30, 10):
            grid[r0 + 5, c0 + 5] = "b"
    assert g.bush_spread(grid) == pytest.approx(0.5)


def test_interior_shares_exclude_the_border():
    grid = g.new_grid()
    grid[5, 1:59] = "#"
    grid[6, 1:59] = "b"
    grid[7, 1:30] = "~"
    shares = g.interior_shares(grid)
    assert shares["wall"] == pytest.approx(58 / 3364)
    assert shares["bush"] == pytest.approx(58 / 3364)
    assert shares["water"] == pytest.approx(29 / 3364)


def test_check_names_every_failure_of_a_bare_grid():
    grid = g.new_grid()
    reasons = g.check(grid, g.FAMILIES["standard"], g.POINT)
    text = " | ".join(reasons)
    assert "loader" in text and "wall share" in text and "bush share" in text
    assert "water share" in text and "boxes" in text and "spawns" in text and "bush spread" in text


def test_csv_round_trips_through_the_loader(tmp_path):
    grid, _ = g.generate(1, "open")
    path = tmp_path / "m.csv"
    path.write_text(g.to_csv(grid))
    tiles = loader.load_map_csv(path, SimpleNamespace(map_h=60, map_w=60))
    assert np.array_equal(tiles, g.to_tiles(grid))
    assert g.render(grid).count("\n") == 59


# ---- generate ----------------------------------------------------------------------------------


@pytest.mark.parametrize("family", list(g.FAMILIES))
def test_generate_passes_every_band_and_check_and_is_deterministic(family):
    fam = g.FAMILIES[family]
    grid, stats = g.generate(1, family)
    assert g.check(grid, fam, fam.symmetry) == []
    assert np.array_equal(grid, g.mirror_grid(grid, fam.symmetry))
    assert fam.wall[0] <= stats["wall"] <= fam.wall[1]
    assert fam.bush[0] <= stats["bush"] <= fam.bush[1]
    assert fam.water[0] <= stats["water"] <= fam.water[1]
    assert fam.boxes[0] <= stats["boxes"] <= fam.boxes[1]
    assert stats["spawns"] == g.N_SPAWNS
    assert stats["bush_spread"] >= g.BUSH_SPREAD_MIN
    assert stats["pocket_min"] >= g.POCKET_MIN_REACH / g.POCKET_OPEN_FIELD
    assert stats["longest_wall_run"] <= g.MAX_WALL_RUN
    assert stats["bush_waypoints"] <= loader.MAX_BUSH_WAYPOINTS
    loader.validate_map(g.to_tiles(grid), SimpleNamespace(map_h=60, map_w=60))

    again, again_stats = g.generate(1, family)
    assert g.to_csv(again) == g.to_csv(grid), "byte-identical across two calls"
    assert again_stats["seed"] == stats["seed"]
    direct, direct_stats = g.generate(stats["seed"], family)
    assert direct_stats["attempts"] == 1 and g.to_csv(direct) == g.to_csv(grid), \
        "the reported seed regenerates the map on its own"
    if fam.moat_inset:
        assert (grid[fam.moat_inset, 6:54] == "~").sum() > 40, "the moat is there"


def test_generate_symmetry_override_makes_a_mirror_standard_map():
    grid, stats = g.generate(1, "standard", symmetry=g.MIRROR)
    assert stats["symmetry"] == g.MIRROR
    assert np.array_equal(grid, g.mirror_grid(grid, g.MIRROR))
    assert g.check(grid, g.FAMILIES["standard"], g.MIRROR) == []


def test_generate_reports_rejected_seeds_and_gives_up_on_an_impossible_family():
    impossible = g.Family("impossible", 0, (0.90, 1.0), (0.0, 1.0), (0.0, 1.0), (0, 64))
    with pytest.raises(RuntimeError, match="impossible: no seed in \\[5, 7\\)"):
        g.generate(5, impossible, max_attempts=2)
    tight = g.Family("tight", 0, (0.07, 0.12), (0.20, 0.30), (0.03, 0.08), (20, 32))
    _, stats = g.generate(1, tight, max_attempts=50)
    assert all(isinstance(seed, int) and reasons for seed, reasons in stats["rejected"])
    assert stats["attempts"] == len(stats["rejected"]) + 1
