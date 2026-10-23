"""`GasMap` -> the fields deploy6's `zone` group wants. See BRAWL_DEPLOYMENT_DESIGN.md 9.14-9.16.

The group is the honest pair the second deploy spec settled on and every spec since has kept:

| field | supplier here | honesty |
|---|---|---|
| `hero_margin_local` | four ray scans over observed gas | recoverable **to the sim's horizon** |
| `active` | has any gas been observed this match | truthful |

`estimate` returns exactly these two, and `assemble._put_zone` takes what the loaded spec names, so
a spec that wants a zone field this file does not make fails closed on its first decision.

The first deploy spec (`archive/configs/agent_obs_deploy.yaml`) also read three columns this file
supplied until that spec was archived on 2026-10-21: the unclamped `hero_margin`, fed these clamped
values (-1.9 pp, §9.15's "clamped at 10 tiles"); `safe_area_frac`, estimated over the sim's map
extent (-2.8 pp at +/-0.05); and `next_shrink_in`, pinned at 0.0 because zeroing it cost 1.5 pp and
a plausible wrong countdown 4.4 pp (§9.14). The countdown is structurally unobservable: an
egocentric camera cannot tell the gas front advancing from the gas front being revealed as the
camera pans (measured, 7 tiles in 3 s purely from reveal).

**The horizon is read from the run's config, never hardcoded.** `cfg.zone_margin_horizon_tiles`
(10.0 tiles) is the same number `zone.hero_margin_local` clamps to on the sim side. Two copies in
two packages is exactly how the two halves drift apart, and the drift is silent: the spec still
names the right column while the estimator saturates somewhere the training data never did.

**The known bias is one-sided and optimistic.** `GasMap` is sticky -- a cell it has never seen
stays clear forever -- so the ray scans read *safer* than the truth, in exactly the places the
hero walked away from. The clamp bounds that error at the horizon; it does not remove it. §9.16
records this as the residual and as a deployment-side problem: the fix is a better accumulation
rule in `GasMap`, not a different observation.
"""
import numpy as np

# `hero_margin_local`'s four components, in the sim's own order (its `hero_margin`, clamped):
# `(x - lo.x, hi.x - x, y - lo.y, hi.y - y)` -- see `core/obs_schema.py`. Each is a signed distance
# in tiles to one edge of the safe rect, POSITIVE when the hero is inside on that side.
_RAYS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class ZoneEstimator:
    """`GasMap` + hero position -> the `zone` group, in the sim's units.

    One instance per loop, reset by the loop at every match start. The margins are stateless
    between calls: they are a function of the gas map handed in, which is where the accumulation
    actually lives, so a `GasMap.reset` on a segment change needs no corresponding reset here --
    margins of its own would point at a world frame that moved.

    `active` is the one exception, and on purpose. It is a single bit with no position in it, so
    it has no world frame to go stale in, and it must NOT follow the gas map through a segment
    change (see `active`). `reset` clears it, and only a match start calls that.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # Read, not stored as a literal. See the module docstring.
        self.horizon = float(cfg.zone_margin_horizon_tiles)
        self._seen = False          # `active`'s latch

    def reset(self) -> None:
        """A new match. `DeployLoop._begin_match` calls this with its other per-match resets, and
        nothing calls it on a segment change: that is the difference between this latch and the
        gas map it reads."""
        self._seen = False

    def estimate(self, gas, hero_pos) -> dict:
        """`{field: value}` for `assemble`'s `zone` argument.

        `hero_pos` is in WORLD tiles -- `camera_relative + odometry.position_tiles`, the same frame
        `GasMap` deposits in. Passing camera-relative coordinates here would put the scan origin
        wherever the hero happened to start the match, which reads as a plausible margin and is
        wrong by however far they have walked.
        """
        return {"hero_margin_local": self.hero_margin(gas, hero_pos), "active": self.active(gas)}

    # -- the fields -----------------------------------------------------------

    def hero_margin(self, gas, hero_pos) -> tuple[float, float, float, float]:
        """Four signed distances in tiles, clamped to +/- the horizon.

        **The sign flips on where the hero is standing, and that is the whole trick.** On clear
        ground, the distance along -x to the first gassed cell *is* `x - lo.x`, the safe rect's
        left margin. Standing in gas, the hero is outside the rect on every side, so all four go
        negative and their magnitude is the distance to clear ground in that direction.

        **The clamp is symmetric because the blindness is.** Standing deep in gas, the nearest
        clear ground is as far outside the sensor as distant gas is when standing safe -- so a
        one-sided clamp would model a sensor this is not.
        """
        col, row = self._cell(gas, hero_pos)
        if not self._inside(gas, col, row):
            # Off the canvas entirely. Report full margins rather than zeros: an unknown world is
            # an unseen one, and this estimator's documented bias is optimistic in exactly that
            # direction. Zeros would claim the hero is standing exactly on all four edges at once.
            return (self.horizon, self.horizon, self.horizon, self.horizon)

        in_gas = bool(gas.gassed[row, col])
        sign = -1.0 if in_gas else 1.0
        out = []
        for dx, dy in _RAYS:
            out.append(sign * self._ray(gas, col, row, dx, dy, looking_for=not in_gas))
        return tuple(out)

    def active(self, gas) -> bool:
        """Whether gas has been on screen this match. The sim's `zone.active` is the latch
        `state.zone_seen` (`brawl_sim/core/zone.py` `mark_seen`): set once gas is on screen, kept
        for the rest of the episode, cleared only by a reset. This is that latch, on the gas the
        loop has deposited.

        **The latch is kept here because `GasMap` cannot keep it.** The map is sticky only within
        an odometry segment: a new segment (a cut, or a lattice re-lock) clears `gassed`, since its
        cells are in a world frame that no longer exists. Reading `gassed.any()` alone would drop
        the flag to 0 mid-match whenever that happens with no gas in view, a 1 -> 0 flip the sim
        never produces. So each call folds the current map into `_seen`, which only `reset` (a
        match start) clears.

        Called once per decision, from `estimate`, which is also when the sim's `mark_seen` runs.
        """
        self._seen = self._seen or bool(gas.gassed.any())
        return self._seen

    # -- geometry -------------------------------------------------------------

    def _cell(self, gas, hero_pos) -> tuple[int, int]:
        ox, oy = gas.origin
        return (int(np.floor(hero_pos[0])) - ox, int(np.floor(hero_pos[1])) - oy)

    def _inside(self, gas, col: int, row: int) -> bool:
        return 0 <= col < gas.width and 0 <= row < gas.height

    def _ray(self, gas, col: int, row: int, dx: int, dy: int, *, looking_for: bool) -> float:
        """Tiles from `(col, row)` to the first cell matching `looking_for`, capped at the horizon.

        Walks cell by cell rather than vectorising the whole row: the horizon is 10 tiles, there
        are four rays, and this runs four times a second. A slice-and-argmax would be the same
        answer with two more off-by-one opportunities in it.

        Running off the canvas returns the horizon, not the distance travelled -- past the edge is
        unknown, and unknown is optimistic here by the same rule as everywhere else in this file.
        """
        limit = int(np.ceil(self.horizon))
        for step in range(1, limit + 1):
            c, r = col + dx * step, row + dy * step
            if not self._inside(gas, c, r):
                break
            if bool(gas.gassed[r, c]) == looking_for:
                return min(float(step), self.horizon)
        return self.horizon
