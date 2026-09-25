"""The one-wide-gap rule's detector (brawl_vision/terrain/gaps.py). Its voting behaviour is
tested with the occupancy map in test_vision_occupancy.py; this file pins what counts as a gap.

Grids are written by hand in class indices so a wrong answer is a wrong picture, not a wrong
constant: `F`/`W`/`B`/`A`/`N` below are FLOOR, WALL, BUSH, WATER (aqua) and FENCE.
"""
import numpy as np
import pytest

from brawl_sim.constants import Tile
from brawl_vision.terrain.gaps import one_wide_gaps
from brawl_vision.terrain.labeling import CLASS_INDEX
from brawl_vision.terrain.occupancy import UNKNOWN

F, W, B, A, N = (CLASS_INDEX[t] for t in (Tile.FLOOR, Tile.WALL, Tile.BUSH, Tile.WATER, Tile.FENCE))


def _mask(rows):
    return np.array(rows, bool)


def test_a_floor_cell_between_two_walls_is_a_horizontal_pinch():
    g = np.array([[F, W, F, W, F]])
    assert (one_wide_gaps(g, flanks=False) == _mask([[0, 0, 1, 0, 0]])).all()
    assert (one_wide_gaps(g) == _mask([[0, 1, 1, 1, 0]])).all(), "flanks are the two walls"


def test_a_floor_row_between_two_wall_rows_is_a_vertical_pinch():
    g = np.array([[W, W, W],
                  [F, F, F],
                  [W, W, W]])
    assert (one_wide_gaps(g, flanks=False) == _mask([[0, 0, 0], [1, 1, 1], [0, 0, 0]])).all()
    assert one_wide_gaps(g).all(), "every cell of a three-row pinch takes part"


def test_a_two_wide_passage_is_allowed():
    g = np.array([[W, W, W],
                  [F, F, F],
                  [F, F, F],
                  [W, W, W]])
    assert not one_wide_gaps(g).any()


def test_a_one_thick_wall_between_floors_is_not_a_gap():
    """The rule is about passages, not obstacles: thin walls are everywhere in real maps."""
    g = np.array([[F, W, F]])
    assert not one_wide_gaps(g).any()


@pytest.mark.parametrize("middle", [F, B])
@pytest.mark.parametrize("flank", [W, A, N])
def test_walkable_is_floor_or_bush_and_blocking_is_whatever_blocks_a_unit(middle, flank):
    """WATER and FENCE stop a brawler exactly as a WALL does (TILE_BLOCKS_UNIT), and a bush is
    ground you can stand in; a one-wide passage is one whichever of them form it."""
    g = np.array([[flank, middle, flank]])
    assert (one_wide_gaps(g, flanks=False) == _mask([[0, 1, 0]])).all()


def test_water_between_walls_is_not_a_passage():
    g = np.array([[W, A, W]])
    assert not one_wide_gaps(g).any()


def test_a_corner_contact_is_not_a_pinch():
    """Only orthogonal neighbours count; two blockers touching at a corner leave the floor
    between them unflagged."""
    g = np.array([[W, F],
                  [F, W]])
    assert not one_wide_gaps(g).any()


def test_unknown_cells_neither_form_nor_flank_a_gap():
    assert not one_wide_gaps(np.array([[W, F, UNKNOWN]])).any()
    assert not one_wide_gaps(np.array([[W, UNKNOWN, W]])).any()


def test_the_known_mask_can_withdraw_a_flank():
    """A blocker under a crate, the HUD or the footprint's edge is not evidence, so the rule
    must not pinch against it -- `OccupancyMap.update` passes exactly the cells it will vote."""
    g = np.array([[W, F, W]])
    assert one_wide_gaps(g).any()
    assert not one_wide_gaps(g, known=_mask([[1, 1, 0]])).any()
    assert not one_wide_gaps(g, known=_mask([[1, 0, 1]])).any(), "an unknown middle is no gap"


def test_the_grid_edge_is_not_a_blocker():
    """Off the edge of the frame is unknown, not wall: a floor cell against the edge with a
    wall on its other side is not pinched."""
    g = np.array([[F, W]])
    assert not one_wide_gaps(g).any()


def test_a_non_grid_input_is_refused():
    with pytest.raises(ValueError, match="rows, cols"):
        one_wide_gaps(np.array([F, W, F]))
