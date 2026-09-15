"""brawl_deployment.perception -- screen pixels to the policy's observation vector.

The order is fixed by what each stage needs from the one before: rectify (camera-relative tiles)
-> odometry (a world frame to put them in) -> detect -> track (identity and velocity) -> assemble.

Two trackers, not one, because they run at different rates: `tracker` identifies entities at the
4 Hz decision rate, `projectiles` tracks projectiles at the perception rate because at 4 Hz they are barely
observable at all (BRAWL_DEPLOYMENT_DESIGN.md 9.5).

`loot` shares the projectile model's boxes and keeps the other two classes: crates and dropped
power cubes, as a sticky map of fixed cells for the grid's `box` and `pickup` planes. It also
supplies the occupancy map's occlusion masks, for crates and for brawler sprites.

`grid` is the last stage before assembly: it crops the accumulated terrain map and scatters the
tracks into the hero-centred planes the observation's `grid` group wants (6.2). Everything
it places was produced upstream, so its whole contract is placement -- matching
`brawl_sim/core/observation.py:_build_grid` cell for cell.

`zone` is the `zone` group's supplier -- `GasMap` plus a hero position to four margins, an
active flag and an area fraction (9.14-9.16). It is NOT `brawl_vision/terrain/zone.py`, which
answers "is this cell gassed" for one frame; this one reads what that has accumulated.

`shadow` is the odd one out and sits here anyway: it reads no pixels at all. It dead-reckons the
hero's own timers from the actions we issue, which is where the whole `self` group comes from
(6.3), and it consumes `hero_bars.read_ammo` as a desync check -- so it is a supplier of
observation fields and a consumer of CV, which is what this subpackage is.
"""
from .grid import GasMap, GridBuilder, GridSpec
from .loot import (Loot, LootMap, LootResult, box_occlusion, crate_occlusion,
                   require_loot_classes)
from .projectiles import Projectile, ProjectileResult, ProjectileTracker
from .shadow import Desync, ShadowHero, ShadowParams
from .tracker import EntityTracker, Track, TrackerResult
from .zone import ZoneEstimator

__all__ = ["EntityTracker", "Track", "TrackerResult",
           "Projectile", "ProjectileResult", "ProjectileTracker",
           "Loot", "LootMap", "LootResult", "box_occlusion", "crate_occlusion",
           "require_loot_classes",
           "ShadowHero", "ShadowParams", "Desync",
           "GridBuilder", "GridSpec", "GasMap",
           "ZoneEstimator"]
