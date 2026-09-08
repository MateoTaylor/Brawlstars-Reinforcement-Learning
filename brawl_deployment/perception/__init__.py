"""brawl_deployment.perception -- screen pixels to the policy's observation vector.

The order is fixed by what each stage needs from the one before: rectify (camera-relative tiles)
-> odometry (a world frame to put them in) -> detect -> track (identity and velocity) -> assemble.

Two trackers, not one, because they run at different rates: `tracker` identifies entities at the
4 Hz decision rate, `projectiles` tracks projectiles at 20 Hz because at 4 Hz they are barely
observable at all (BRAWL_DEPLOYMENT_DESIGN.md 9.5). `assemble` does not exist yet; see 6.

`grid` is the last stage before assembly: it crops the accumulated terrain map and scatters the
tracks into the eight hero-centred planes the observation's `grid` group wants (6.2). Everything
it places was produced upstream, so its whole contract is placement -- matching
`brawl_sim/core/observation.py:_build_grid` cell for cell.

`shadow` is the odd one out and sits here anyway: it reads no pixels at all. It dead-reckons the
hero's own timers from the actions we issue, which is where the whole `self` group comes from
(6.3), and it consumes `hero_bars.read_ammo` as a desync check -- so it is a supplier of
observation fields and a consumer of CV, which is what this subpackage is.
"""
from .grid import GasMap, GridBuilder, GridSpec
from .projectiles import Projectile, ProjectileResult, ProjectileTracker
from .shadow import Desync, ShadowHero, ShadowParams
from .tracker import EntityTracker, Track, TrackerResult

__all__ = ["EntityTracker", "Track", "TrackerResult",
           "Projectile", "ProjectileResult", "ProjectileTracker",
           "ShadowHero", "ShadowParams", "Desync",
           "GridBuilder", "GridSpec", "GasMap"]
