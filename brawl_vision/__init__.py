"""brawl_vision -- the perception engine that lets a policy trained in `brawl_sim` read the
real game. See Terrain_Perception_Build_Plan.md.

Structured as CHUNKS. This package's root holds what every chunk needs (frame sources, camera
geometry, config); each subpackage is one chunk. `terrain/` is the first: calibrated homography,
odometry, per-cell classification, accumulated occupancy. `object_detection/` is the second, and
it is deliberately NOT built on the first -- it runs a third-party YOLO over the RAW frame and
returns boxes in screen pixels, where terrain rectifies and accumulates in tile space. The two
share `sources.Frame` and nothing else; the scripts that show both compose them. `hud.py` is the
third and sits in this root rather than in a subpackage, because a screen-anchored readout needs
neither a homography nor a detection -- only a frame and the fact that it is 16:9.

**CONVENTIONS.md governs `brawl_sim`, not this package, and the boundary is deliberate.** The
leading-(N,) batch rule and the no-host-sync rule scope to `BrawlVecEnv.step()`; there is
exactly one screen and one live match here, so `.item()`, `.cpu()`, and OpenCV's numpy-in /
numpy-out contract are all fine. Do not contort this code to look batched.

What DOES carry over, unchanged, because it is what lets the two halves speak the same language
when they are fused: **units** (tiles, seconds, radians), **coordinates** (`pos = (x, y)`,
x = column, y = row, y increasing DOWNWARD, tile `(i, j)` covering `[i, i+1) x [j, j+1)`), and
the terrain vocabulary itself -- `brawl_sim.constants.Tile`, imported directly rather than
re-declared. See `terrain/occupancy.py` for the one thing that vocabulary is missing (UNKNOWN)
and why it is defined here instead of widening the enum.
"""
