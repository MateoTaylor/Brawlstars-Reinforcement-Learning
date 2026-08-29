"""The colours the vision stack paints a reconstructed map in.

**Deliberately not the simulator's palette, and that is a change.** Phases E, I and L originally
imported `brawl_sim.render.viewer.TILE_COLORS` so a vision map and a simulator render could be put
side by side and compared by eye. That still has value, and this file gives it up on purpose: the
vision output is looked at next to *game footage*, not next to a simulator render, and the palette
that reads best against a screenshot is not the one that reads best against `scripts/watch.py`.

If the two ever need to match again, the fix is to point this module's table at the simulator's --
one import, one place -- rather than to re-scatter the colours through three viewers.

`brawl_sim` is untouched. `scripts/watch.py` and the simulator's own renderer keep their palette.
"""
from brawl_sim.constants import Tile

# A cell nobody has observed. Black rather than a hue: UNKNOWN is the absence of a claim, and
# giving it a colour of its own put it in visual competition with the classes that ARE claims.
UNKNOWN_COLOR = "#000000"

TILE_COLORS = {
    # Floor and wall are both blue-greys taken from the footage rather than invented: sampling the
    # cells the classifier is >90% sure of across showdown_alternate_map2 gives a floor of #25293a
    # and a wall of #233e64. Floor is dark and walls light, the inverse of the simulator's scheme,
    # because floor is ~73% of a typical map and is the one that should recede.
    #
    # FLOOR IS NOT THE SAMPLED VALUE, and the constraint is UNKNOWN. #25293a's brightest channel is
    # 58/255 = 0.227, under the 0.3 that `tests/test_vision_overlay.py` requires between UNKNOWN and
    # every class -- a near-black floor beside a black "never observed" makes the two
    # indistinguishable exactly when the distinction matters most, on a half-explored map where the
    # whole question is which dark region is floor and which is nothing. So the sampled hue is kept
    # and the value lifted 1.5x until it clears (0.341). Blue carries it: at the same lightness a
    # NEUTRAL grey would fail, because its brightest channel is lower.
    Tile.FLOOR: "#383e57",
    # Lighter than the sampled #233e64, which is a whole-cell average including the crates' shadowed
    # sides. This echoes their lit tops, and buys the contrast against the floor that the average
    # does not (141 in RGB against 39).
    Tile.WALL:  "#9aa8cc",
    Tile.BUSH:  "#d23b2e",
    Tile.WATER: "#e08b2a",
    # Not specified when the rest were chosen. Picked to sit clear of both the bush red and the
    # water orange, which are its nearest neighbours on the wheel and the two it would otherwise be
    # confused with -- FENCE is the rarest class and the one that can least afford to be ambiguous.
    Tile.FENCE: "#e8d44a",
    # Never rendered by the vision stack (the classifier has five classes), but present so the
    # table can stand in for the simulator's anywhere that iterates `Tile`.
    Tile.SPAWN: "#383e57",
    Tile.BOX:   "#383e57",
}
