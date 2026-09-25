"""The one-wide-gap rule: a map-design constraint used as evidence against a classification.

Brawl Stars maps are built under a rule that no walkable passage is a single tile wide. Any gap
between obstacles -- a corridor, a doorway through a wall, a notch into one -- is at least two
tiles across. So whenever a classified grid shows a walkable cell with unit-blocking cells on BOTH
sides of it along one axis, that grid is claiming something the map cannot contain: at least one
of the three cells is wrong. Either the middle cell is really blocking (the "gap" is a solid line)
or a flank is really walkable (the gap is two wide). Which one is not knowable from the frame.

`one_wide_gaps` marks the cells that take part in such a claim. `OccupancyMap.update` casts those
cells' votes at `occupancy_gap_rule_weight` instead of 1, so that the views which do NOT make the
claim decide the cell. The rule is deliberately symmetric -- it down-weights the pinched cell and
both flanks alike -- because that gives it a property worth having: it can never flip a cell that
every view agrees on (all votes scale together and the argmax is unchanged). It changes an
outcome only where views disagree, and then it prefers the interpretation the map rules allow.

What counts as a gap. Walkable is FLOOR or BUSH; blocking is what blocks a unit, WALL, WATER or
FENCE (`brawl_sim.constants.TILE_BLOCKS_UNIT`), because the rule is about passages a brawler can
take, not about sightlines. A cell that is not `known` -- unlabelled, outside the footprint, under
the HUD, gassed, occluded by a box, or off the edge of the grid -- is neither walkable nor
blocking, so it neither forms a gap nor flanks one: the rule only ever acts on cells the caller
already trusts. Only orthogonal pinches count. Two blockers touching at a corner are not a
passage in the game's sense and the rule does not try to reason about them.

Hand labels break the rule at about 0.6% of walkable cells (121 of 18,658 across the 92 label
files, 2026-09-22), almost all of them vertical pinches under a wall's overhang: the top face of a
stone cube and the arch of a gravestone are drawn one row north of the tile they stand on and get
labelled `#`, which narrows the corridor above a wall run by a row. Those are consistent across
views, so the rule leaves them alone; it is not a fix for a labelling convention.
"""
import numpy as np

from brawl_sim.constants import TILE_BLOCKS_UNIT

from .labeling import CLASSES

# Indexed by class index (a `CLASSES` position), not by Tile: `cells` grids hold class indices.
_BLOCKING = np.array([bool(TILE_BLOCKS_UNIT[int(t)]) for t in CLASSES])
_WALKABLE = ~_BLOCKING


def one_wide_gaps(cells: np.ndarray, known: np.ndarray | None = None, *,
                  flanks: bool = True) -> np.ndarray:
    """(rows, cols) bool of the cells taking part in a one-wide gap.

    `cells` is a grid of `CLASSES` indices; negative entries (`occupancy.UNKNOWN`, an unlabelled
    cell) are never known. `known`, when given, further restricts which cells may take part.
    With `flanks` the two blockers either side of each pinched cell are marked too -- the form
    the vote weighting wants; without it only the pinched walkable cells are, which is the count
    "how many one-wide gaps does this map show".
    """
    cells = np.asarray(cells)
    if cells.ndim != 2:
        raise ValueError(f"expected a (rows, cols) grid, got shape {cells.shape}")
    valid = cells >= 0
    if known is not None:
        valid &= np.asarray(known, bool)
    idx = np.clip(cells, 0, len(CLASSES) - 1)
    walk = valid & _WALKABLE[idx]
    block = valid & _BLOCKING[idx]

    pinched_h = np.zeros_like(walk)
    pinched_v = np.zeros_like(walk)
    pinched_h[:, 1:-1] = walk[:, 1:-1] & block[:, :-2] & block[:, 2:]
    pinched_v[1:-1, :] = walk[1:-1, :] & block[:-2, :] & block[2:, :]
    out = pinched_h | pinched_v
    if flanks:
        out = out.copy()
        out[:, :-2] |= pinched_h[:, 1:-1]       # the blocker to the left of a horizontal pinch
        out[:, 2:] |= pinched_h[:, 1:-1]        # ...and to the right
        out[:-2, :] |= pinched_v[1:-1, :]       # above a vertical pinch
        out[2:, :] |= pinched_v[1:-1, :]        # ...and below
    return out
