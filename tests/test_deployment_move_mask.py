"""`move_mask.py` -- the dead-bin move mask, held against the sim's own collision rule.

The module is a numpy port of `core/terrain.resolve_move` for one hero on the grid the policy
sees, and the port IS the product: a bin it kills that the sim would have walked, or the reverse,
is a deployment disagreeing with training in exactly the place the mask was added to make them
agree. So the first test is the equivalence itself, on random planes, against `resolve_move`
called directly; the pictures pin what the rule means; and the rest cover the two branches
`terrain.py` has no counterpart for and where the constants come from.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from brawl_deployment.move_mask import (MoveMask, circle_blocked, legal_move_bins,
                                        unit_radius_tiles)
from brawl_deployment.perception.shadow import ShadowParams
from brawl_sim.config import load_config
from brawl_sim.core import geometry as geo
from brawl_sim.core import terrain

RADIUS = 0.4          # entities.unit_radius
STEP = 0.1365         # Mortis: move_speed 2.73 * dt 0.05, one sim tick of walking
N_BINS = 16


def _dead(blocks, xy, radius=RADIUS, step=STEP) -> list[int]:
    return np.flatnonzero(~legal_move_bins(blocks, xy, radius=radius, step=step)).tolist()


def _sim_legal(blocks: np.ndarray, hero_xy) -> list[bool]:
    """The sim's answer: `resolve_move` on the same plane as a one-map bank, one row per bin,
    legal where it moved the hero. Idle prepended, as the mask lays it out."""
    h, w = blocks.shape
    cfg = SimpleNamespace(map_h=h, map_w=w)
    bank = torch.as_tensor(np.asarray(blocks, bool))[None]
    map_id = torch.zeros(N_BINS, dtype=torch.int64)
    pos = torch.tensor(hero_xy, dtype=torch.float32).expand(N_BINS, 2)
    delta = STEP * geo.dir_from_bin(torch.arange(N_BINS), N_BINS)
    after = terrain.resolve_move(bank, map_id, pos, delta, torch.full((N_BINS,), RADIUS), cfg)
    moved = (after - pos).abs().max(dim=-1).values > 1e-6
    return [True] + moved.tolist()


# ---- the port ---------------------------------------------------------------------------------

def test_the_port_agrees_with_resolve_move_on_random_planes():
    """1200 seeded (13, 21) planes at 25% wall density -- the grid's own shape -- with the hero
    anywhere its footprint is free, which is the only place the sim ever asks. Every bin, both
    ways. The counts at the end say the comparison had teeth: enough free placements, and enough
    dead bins among them (a free footprint is within half a cell of a wall on roughly one
    placement in five), that a port which never killed anything could not pass."""
    rng = np.random.default_rng(20260923)
    checked = dead = 0
    for _ in range(1200):
        blocks = rng.random((13, 21)) < 0.25
        xy = (float(rng.uniform(0.5, 20.5)), float(rng.uniform(0.5, 12.5)))
        if circle_blocked(blocks, xy, RADIUS):
            continue
        ours = legal_move_bins(blocks, xy, radius=RADIUS, step=STEP).tolist()
        assert ours == _sim_legal(blocks, xy), (xy, np.argwhere(blocks).tolist())
        checked += 1
        dead += ours.count(False)
    assert checked >= 300 and dead >= 60, (checked, dead)


# ---- what the rule means ----------------------------------------------------------------------

def test_a_wall_east_kills_east_alone_because_the_sim_slides():
    """`resolve_move` is axis-separated: a bin with an east component into the wall loses its x
    step and keeps its y step, so ENE and ESE still move the hero -- along the wall. Only the bin
    pointing straight in has nowhere to go. A mask that killed the whole eastern arc would be
    stricter than the sim it claims to copy."""
    p = np.zeros((5, 5), bool)
    p[2, 3] = True
    assert _dead(p, (2.55, 2.5)) == [1]


def test_a_corner_kills_the_arc_between_its_walls():
    """Walls east and south: the five bins from E through S lose both steps, or the only step
    they have; SSW (bin 6) keeps its westward x step and moves."""
    p = np.zeros((5, 5), bool)
    p[2, 3] = True
    p[3, :] = True
    assert _dead(p, (2.55, 2.55)) == [1, 2, 3, 4, 5]


def test_a_wall_further_than_a_step_plus_the_radius_kills_nothing():
    """The rule is sub-cell: at x = 2.1 the eastward step puts the footprint's edge at 2.6365,
    inside the free cell, so east is legal. The hero's fractional position matters, which is why
    the loop hands the mask `hero_pos - origin_tile` and not a cell index."""
    p = np.zeros((5, 5), bool)
    p[2, 3] = True
    assert _dead(p, (2.1, 2.5)) == []


def test_a_one_cell_pocket_kills_the_axis_bins_and_the_diagonals_slide():
    """A hero centred in a one-cell pocket has 0.1 tiles of play on each side of its 0.4
    footprint. The four axis bins want 0.1365 and are dead; every other bin loses its long
    component and keeps the short one, which is under the play, so it moves -- the sim creeps
    along the pocket the same way, and the mask must be exactly that permissive, no stricter.
    Idle is legal, as it is everywhere: index 0 is never written, the invariant
    `DeployedPolicy.act` refuses to be handed otherwise."""
    p = np.ones((3, 3), bool)
    p[1, 1] = False
    legal = legal_move_bins(p, (1.5, 1.5), radius=RADIUS, step=STEP)
    assert legal[0]
    assert np.flatnonzero(~legal).tolist() == [1, 5, 9, 13]
    assert legal.tolist() == _sim_legal(p, (1.5, 1.5))


def test_the_planes_edge_is_a_wall_as_the_maps_edge_is_in_the_sim():
    """`terrain.oob` reads off-map as blocked and so does the port. Moot on the live grid -- the
    hero sits six cells from the crop's nearest edge and a step plus the radius is half a cell --
    but the port copies the rule rather than deciding it cannot matter."""
    p = np.zeros((5, 5), bool)
    assert _dead(p, (0.45, 2.5)) == [9]


# ---- the two branches terrain.py does not have ------------------------------------------------

def test_a_footprint_the_map_already_blocks_shrinks_rather_than_giving_up():
    """The game lets a brawler press nearer to a wall than `unit_radius`, and the map is an
    estimate: at x = 2.7 against a wall face at 3.0 the 0.4 footprint overlaps the wall, which in
    the sim cannot happen. Halving the footprint until it is free (0.2 here) keeps the rule
    running with the clearance the hero actually has: east is still dead, everything else moves."""
    p = np.zeros((5, 5), bool)
    p[2, 3] = True
    assert _dead(p, (2.7, 2.5)) == [1]


def test_a_hero_whose_centre_reads_blocked_keeps_every_bin():
    """A wall drawn under the hero is a map error, and the mask must not pin him to idle on it:
    all sixteen bins stay legal, as they were through training."""
    p = np.zeros((5, 5), bool)
    p[2, 2] = True
    assert _dead(p, (2.7, 2.7)) == []
    assert _dead(p, (2.5, 2.5)) == []


# ---- the constants and the call shape ---------------------------------------------------------

def test_the_constants_are_the_sims_own_and_not_literals():
    """`radius` is `entities.unit_radius` from configs/default.yaml, `step` the shadow's
    `move_speed` times the checkpoint config's `dt`, `n_bins` the config's `n_move_bins`. The
    module holds no number of its own; the literals in THIS file are the pinned values."""
    cfg = load_config("configs/default.yaml")
    params = ShadowParams.load()
    mm = MoveMask.from_configs(params, cfg)
    env = yaml.safe_load(open("configs/default.yaml").read())
    assert mm.radius == float(env["entities"]["unit_radius"]) == RADIUS
    assert mm.step == pytest.approx(float(params.move_speed) * float(cfg.dt))
    assert mm.step == pytest.approx(STEP)
    assert mm.n_bins == int(cfg.n_move_bins) == N_BINS
    assert mm.legal(np.zeros((13, 21), bool), (10.5, 6.5)) == (True,) * 17


def test_a_randomized_unit_radius_is_refused(tmp_path):
    """The sim may draw `unit_radius` from a range for domain randomisation; a deployed hero has
    one radius, and guessing the midpoint would be a wall too near or too far on every decision.
    Named, so the operator finds the yaml key rather than a TypeError."""
    bad = tmp_path / "env.yaml"
    bad.write_text(yaml.safe_dump({"entities": {"unit_radius": [0.3, 0.5]}}))
    with pytest.raises(ValueError, match="randomized"):
        unit_radius_tiles(bad)
    (tmp_path / "none.yaml").write_text(yaml.safe_dump({"entities": {}}))
    with pytest.raises(KeyError, match="unit_radius"):
        unit_radius_tiles(tmp_path / "none.yaml")


def test_a_stacked_grid_or_a_bare_scalar_is_refused():
    """The loop must hand over ONE plane -- `grid[blocks_plane]` -- not the (C, H, W) stack, and
    the hero as (column, row). Either slip would otherwise index something and return a mask."""
    with pytest.raises(ValueError, match="rows, cols"):
        legal_move_bins(np.zeros((21, 13, 21), bool), (10.5, 6.5), radius=RADIUS, step=STEP)
    with pytest.raises(ValueError, match="column, row"):
        legal_move_bins(np.zeros((13, 21), bool), 10.5, radius=RADIUS, step=STEP)
