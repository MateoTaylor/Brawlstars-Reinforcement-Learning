"""`GasMap` -> the fields a deploy spec's `zone` group wants. See BRAWL_DEPLOYMENT_DESIGN.md 9.14-9.16.

**This is the group with no honest full supplier, and that was priced and decided rather than
worked around.** Two ablation passes put the whole group at 10.4 pp (7 standard errors -- the only
large reproducible effect in either pass) and every realistic degradation of it at 0-4 pp against a
1.5 pp standard error. The operator's call (§9.15) was to deploy on the checkpoint of the day, which
read `configs/agent_obs_deploy.yaml`, and to supply the group like this:

| field | supplier here | honesty |
|---|---|---|
| `hero_margin` | four ray scans over observed gas | recoverable **out to a horizon**; saturates past it |
| `hero_margin_local` | the same four scans | the name and the values agree; the sim clamps at the same horizon |
| `active` | has any gas been observed | truthful |
| `safe_area_frac` | 1 - observed gas area / the sim's map area | the group's weak column, knowingly |
| `next_shrink_in` | **pinned at 0.0** | a lie, measured at <=1 SE, recorded, reversible |

`estimate` returns every one of them, and `assemble._put_zone` takes only what the loaded spec
names: `agent_obs_deploy.yaml` wants the four `hero_margin` / `active` / `safe_area_frac` /
`next_shrink_in`, and `agent_obs_deploy2.yaml` and `agent_obs_deploy3.yaml` want
`hero_margin_local` and `active`. Supplying the union rather than asking which spec is loaded keeps
the estimator from needing to know about specs at all; the spec-driven choice stays in one place.

**On `agent_obs_deploy.yaml` the column is `hero_margin` and the values are `hero_margin_local`.
That is on purpose and it is priced.** The sim's `hero_margin` is unclamped -- a training episode
could show 25 tiles -- while everything below saturates at the horizon, because a bounded sensor is
what deployment has. §9.15's second ablation pass measured exactly this substitution at **-1.9 pp**
("clamped at 10 tiles"), against a 1.5 pp standard error. `agent_obs_deploy2.yaml` fixes it properly
by naming `zone.hero_margin_local`, which the sim emits, and `agent_obs_deploy3.yaml` inherits that
group unchanged. So the mismatch is confined to runs on the first deploy spec, and it is the thing to
look at first if one of those agents misjudges gas at long range.

**Pinning `next_shrink_in` is a deliberate exception to "never feed a constant for a field you
cannot supply", not an oversight of it.** The rule exists because a fabricated column is usually an
*unpriced* lie. This one is priced: §9.14 measured zeroing it at 1.5 pp and feeding it a
plausible-looking wrong number at 4.4 pp -- so inventing a countdown is worse than admitting there
isn't one. It is structurally unobservable, and not for want of a better reader: an egocentric
camera cannot tell the gas front advancing from the gas front being revealed as the camera pans
(measured, 7 tiles in 3 s purely from reveal). `configs/agent_obs_deploy2.yaml` drops the column
outright and is the honest spec; it has no run behind it, and the retrain is a lever, not a
prerequisite.

**The horizon is read from the run's config, never hardcoded.** `cfg.zone_margin_horizon_tiles`
(10.0 tiles) is the same number `zone.hero_margin_local` clamps to on the sim side. Two copies in
two packages is exactly how the two halves drift apart, and the drift is silent: the spec still
names the right column while the estimator saturates somewhere the training data never did.

**The known bias is one-sided and optimistic.** `GasMap` is sticky -- a cell it has never seen
stays clear forever -- so both the ray scans and the area estimate read *safer* than the truth, in
exactly the places the hero walked away from. The clamp bounds that error at the horizon; it does
not remove it. §9.16 records this as the residual and as a deployment-side problem: the fix is a
better accumulation rule in `GasMap`, not a different observation.
"""
import numpy as np

# `hero_margin`'s four components, in the sim's own order:
# `(x - lo.x, hi.x - x, y - lo.y, hi.y - y)` -- see `core/obs_schema.py`. Each is a signed distance
# in tiles to one edge of the safe rect, POSITIVE when the hero is inside on that side.
_RAYS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# What a pinned `next_shrink_in` is set to. Zero, not a large number: the schema bounds the field
# at >= 0 and the sim emits `clamp(zone_next_t - time, min=0)`, so 0 is what an overdue-or-disabled
# schedule already looks like in training. A large constant would be a value the column's training
# distribution does contain, which is worse -- it would read as "a shrink is far away" rather than
# as the degenerate case the policy has seen.
PINNED_NEXT_SHRINK_IN = 0.0


class ZoneEstimator:
    """`GasMap` + hero position -> the `zone` group, in the sim's units.

    One instance per match, but stateless between calls: everything it reports is a function of the
    gas map handed in, which is where the accumulation actually lives. Keeping no state of its own
    means a `GasMap.reset` on a segment change needs no corresponding reset here -- one fewer thing
    to forget, and forgetting it would leave stale margins pointing at a world frame that moved.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # Read, not stored as a literal. See the module docstring.
        self.horizon = float(cfg.zone_margin_horizon_tiles)
        self.map_area = float(cfg.map_w) * float(cfg.map_h)

    def estimate(self, gas, hero_pos) -> dict:
        """`{field: value}` for `assemble`'s `zone` argument.

        `hero_pos` is in WORLD tiles -- `camera_relative + odometry.position_tiles`, the same frame
        `GasMap` deposits in. Passing camera-relative coordinates here would put the scan origin
        wherever the hero happened to start the match, which reads as a plausible margin and is
        wrong by however far they have walked.

        `hero_margin` and `hero_margin_local` are one scan under two names, not two estimates: the
        scan saturates at the horizon, which is what `hero_margin_local` means and what
        `hero_margin` knowingly borrows (module docstring). Computing them separately would only
        make room for them to disagree.
        """
        margin = self.hero_margin(gas, hero_pos)
        return {
            "hero_margin": margin,
            "hero_margin_local": margin,
            "active": self.active(gas),
            "safe_area_frac": self.safe_area_frac(gas),
            "next_shrink_in": PINNED_NEXT_SHRINK_IN,
        }

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
        """Whether the zone has started. `cfg.zone_enabled` on the sim side, so it is a property of
        the match rather than of the moment -- but the deployed loop cannot know the mode's rules,
        only what it has seen, and gas on screen is the one unambiguous signal that it is on.

        Latches by construction: `GasMap` is sticky, so once true this stays true for the match.
        That matches the sim, where the zone never turns back off.
        """
        return bool(gas.gassed.any())

    def safe_area_frac(self, gas) -> float:
        """Safe area / map area, over the sim's map extent.

        **The group's weak column, and knowingly so.** The true value needs the map's total extent,
        which an odometry-anchored 128x128 canvas never learns. What makes an estimate possible at
        all is the assumption §9.17 already accepted for `hero.pos_norm`: the map is
        `cfg.map_w x cfg.map_h` tiles centred on the odometry origin. That assumption buys the
        *scale* exactly and quarantines the uncertainty in the origin -- one assumption, reused
        here rather than a second one invented.

        Optimistic, like everything else that reads `GasMap`: cells never seen count as safe.
        Priced at -2.8 pp for a +/-0.05 error (§9.15), which is the cost being accepted.
        """
        gassed = float(np.count_nonzero(self._map_region(gas)))
        return float(np.clip(1.0 - gassed / self.map_area, 0.0, 1.0))

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

    def _map_region(self, gas) -> np.ndarray:
        """The `cfg.map_w x cfg.map_h` block of the gas canvas, centred on the world origin.

        `GasMap.origin` is `(-(width // 2), -(height // 2))`, so the canvas is already centred on
        world (0, 0) -- which is where `MapFrame` puts the map's centre. The two agree by
        construction rather than by a coincidence worth asserting elsewhere.
        """
        ox, oy = gas.origin
        c0 = max(0, int(-self.cfg.map_w / 2) - ox)
        r0 = max(0, int(-self.cfg.map_h / 2) - oy)
        c1 = min(gas.width, c0 + int(self.cfg.map_w))
        r1 = min(gas.height, r0 + int(self.cfg.map_h))
        return gas.gassed[r0:r1, c0:c1]
