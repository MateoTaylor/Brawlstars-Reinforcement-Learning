"""The colours the vision stack paints a reconstructed map in.

**Deliberately not the simulator's palette, and that is a change.** Phases E, I and L originally
imported `brawl_sim.render.viewer.TILE_COLORS` so a vision map and a simulator render could be put
side by side and compared by eye. That still has value, and this file gives it up on purpose: the
vision output is looked at next to *game footage*, not next to a simulator render, and the palette
that reads best against a screenshot is not the one that reads best against `scripts/watch.py`.

If the two ever need to match again, the fix is to point this module's table at the simulator's --
one import, one place -- rather than to re-scatter the colours through three viewers.

`brawl_sim` is untouched. `scripts/watch.py` and the simulator's own renderer keep their palette.

#### There is no such thing as "the game's colours", and this table has to pick a map

Brawl Stars reskins its environments, and the fixtures are not subtle about it:
`showdown_alternate_map` is magenta ground with red foliage, `showdown_alternate_map2` is
blue-grey with blue crates, `day10_gameplay` is dark purple with teal bushes and purple stone
walls. A palette sampled from one is wrong for the others -- not broken, just no longer echoing
what is on screen beside it.

**This table is sampled from `9_5_brawlstars_eval`, retargeted from `day10_gameplay`.** The two
skins share a purple ground and diverge on everything else, so the move was not a no-op: day10's
teal bushes and its washed lilac walls do not exist here.

#### How these numbers were produced, so the next retarget is not a pipette-and-taste job

`scripts/vision_sample_palette.py`. It rectifies every 40th frame, runs the *terrain classifier*
over it, keeps cells the classifier is >=85% sure of and that are fully inside `plan.valid`, and
reports luminance percentiles of the pixels underneath. Sampling by classifier label rather than
by eye matters for one class in particular: a WALL cell contains a lit top face AND a shadowed
side, and their average is a colour that appears nowhere in the footage.

Measured over 62 s of `9_5_brawlstars_eval` (cells / p50 / p90):

    FLOOR   19568   #4e2b62   #6a3371
    WALL     4124   #793aa9   #965ec6      <- p50 is slab-average, p90 is the lit top
    BUSH     2385   #1d9737   #1ec13d
    WATER      107   #85da6e   #90f07c
    FENCE        4   -                     <- four cells in the whole clip; noise, not a sample

Two constraints bound what can be done with those numbers, and both are enforced by tests rather
than by care:

  * every class at least 60 apart in RGB from every other AND from UNKNOWN
    (`test_every_class_is_visually_separable_from_every_other`), and
  * every class more than 0.3 from UNKNOWN in its brightest channel
    (`test_unknown_is_far_from_every_real_class`), which is what sets how dark FLOOR may be.
"""
from brawl_sim.constants import Tile

# A cell nobody has observed. Black rather than a hue: UNKNOWN is the absence of a claim, and
# giving it a colour of its own put it in visual competition with the classes that ARE claims.
UNKNOWN_COLOR = "#000000"

TILE_COLORS = {
    # SAMPLED p50, used as-is. The dark maroon-purple plank floor this skin actually has, and the
    # one class that needs no adjustment: its brightest channel is 98/255 = 0.384, clear of the
    # 0.3 the UNKNOWN test demands. day10's floor needed a 1.35x lift to clear the same bar; this
    # one is simply lighter to begin with, so nothing is invented.
    #
    # Floor is ~73% of a typical map and is the one that should recede. Left at the sample.
    Tile.FLOOR: "#4e2b62",
    # SAMPLED p90 -- the LIT TOP FACE, not the p50 slab average of #793aa9.
    #
    # This is the one deliberate departure from "median of the class", and it is the same
    # departure the day10 palette made for the same reason: a wall is a raised block, roughly half
    # of what the classifier calls WALL is its shadowed side, and what a viewer reads as "wall" is
    # the top. Taking p50 also costs the contrast that makes the map legible -- #793aa9 is only 82
    # from FLOOR in RGB, barely over the 60 the test floors at, and floor-vs-wall is the pair a
    # map is mostly made of. The p90 top face is 133 away.
    Tile.WALL:  "#965ec6",
    # SAMPLED p50, used as-is. Forest green, and a real change from day10's teal #1b8b9a -- this
    # skin's foliage is straightforwardly green.
    #
    # That green is what makes WATER the interesting problem below, and it is also what un-fixed a
    # fix: the day10 note recorded that teal had cleared an old clash between bush red and the
    # enemy marker's red. Green keeps that clearance (enemy red is 200 away), so the clash stays
    # gone for a different reason than before.
    Tile.BUSH:  "#1d9737",
    # SAMPLED p75/p90 (they agree). Flat, bright, almost fluorescent green.
    #
    # **BUSH and WATER are both green in this skin, and that is the palette's one hard problem.**
    # The footage tells them apart by texture and edge -- water is a flat rectangle with a crisp
    # rim, foliage is spiky and mottled -- and a flat-shaded tile map has neither of those to work
    # with. So the whole distinction has to ride on VALUE, which fortunately is where the two are
    # furthest apart in the source: green channel 151 against 240. That is 161 in RGB, well clear
    # of the 60 floor, and it is why neither is nudged toward the other for the sake of "matching"
    # more closely.
    #
    # Only 107 cells backed this, against 2385 for BUSH -- there is little water in the clip. p75
    # and p90 landing on the same value is the reason to trust it anyway: a noisy sample would not
    # be flat across percentiles.
    Tile.WATER: "#90f07c",
    # NOT SAMPLED -- four cells in 62 seconds, which is the classifier hedging on bush edges
    # rather than fence anywhere on this map. Retained from the previous palette, which is the
    # honest thing to do with a class the footage cannot speak to.
    #
    # Still the only warm hue here (floor/wall purple, bush/water green), so it cannot be confused
    # with any of them, and it clears every pair by 105 or better.
    Tile.FENCE: "#e8d44a",
    # Never rendered by the vision stack (the classifier has five classes), but present so the
    # table can stand in for the simulator's anywhere that iterates `Tile`.
    Tile.SPAWN: "#4e2b62",
    Tile.BOX:   "#4e2b62",
}
