"""Perception outputs -> the observation dict the policy was trained to read.

See BRAWL_DEPLOYMENT_DESIGN.md 6. This is the last perception stage: everything it places was
produced upstream by `tracker`, `projectiles`, `shadow`, `grid` and the `brawl_vision` readers,
and its whole contract is to arrange those into the exact tensor the checkpoint expects.

#### It does not build the observation. `obs_select.build_agent_obs` does.

The tempting shape for this module is to lay out the columns itself -- read the spec, concatenate
in order, divide by the normalization scales. That would be a second implementation of a thing
that already exists and already ran 300M times during training, and every one of its failure
modes is silent: a column in the wrong order, a divisor off by the map size, a projectile group
sorted differently. The policy would not crash. It would just be worse, in a way no test that
does not own both implementations can see.

So this module builds a **minimal `full_obs`** -- the same nested dict of leading-`(N,)` tensors
`core/observation.py` produces, at `N = 1`, holding only the fields the deployed spec names -- and
hands it to `obs_select.build_agent_obs`. Column order, normalization, the projectile group's
top-K-by-`time_to_closest` selection and the fairness masking are then not merely equivalent to
training's, they are training's, by construction.

Two consequences worth stating, because both look like bugs:

* **The `view` grid is built with all 12 sim channels, of which 4 are never filled.** `GridBuilder`
  produces the 8 the spec asks for; this scatters them into their canonical sim indices
  (`obs_select._CHANNEL_INDEX`) and leaves `enemy_any`, `enemy_hidden`, `box` and `pickup` as
  zeros. `_build_grid_group` does an `index_select` over exactly the 8 configured channels, so the
  zeros are **provably unread** rather than fed to anything -- which is the distinction that makes
  this legal under the never-feed-a-constant rule. Building an 8-channel array and splicing it in
  as the finished group would work too, and would quietly stop working the day a spec reorders its
  channels.
* **The entities axis carries a hero slot.** `obs_select` drops index 0 (`_HERO_AXIS_PREFIX`), so
  slot k of `EntityTracker` lands at `entities[:, k + 1]`. Writing enemies at index 0 would shift
  every enemy one slot and silently delete the last one.

#### The `entities.hp` promotion rule

Measured in 6.5: 84% of enemy boxes yield an HP, and every enemy track that NEVER produced one
lived four frames or fewer -- glancing detections, not enemies. Tracks that persist read within 1
frame at p50 and 3 at p90, inside one 250 ms decision.

So a track is promoted to an enemy slot **only once it has a committed HP**, and until then the
slot reads `entities.alive = 0` with its other columns zeroed. That is the state the sim itself
produces for an empty slot, so it is in-distribution rather than invented; the alternative is a
made-up HP in a column the policy was trained to trust. It costs at most one decision of latency.

#### What this module does NOT decide

It never invents a value for a field it was handed nothing for. Every supplier is an argument, and
`assemble` raises rather than defaulting when one is missing -- `_require` exists for exactly that.
The one place absolute position is unavoidable is `MapFrame`, which is quarantined below and
documented there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from brawl_sim.core import obs_select

# The canonical 12-channel `view` ordering, borrowed rather than restated. A private name, taken
# deliberately: the alternative is a copy of the table that goes stale the day a channel is added,
# and there is no public accessor for it. `_channel_indices` asserts the two agree.
_CHANNEL_INDEX = obs_select._CHANNEL_INDEX
_N_VIEW_CHANNELS = 12

# `meta.time_frac`'s denominator is the episode length in SECONDS, and it is clamped -- running
# long is out-of-distribution in a way running short is not, because the sim truncates at 1.0.
# See BRAWL_DEPLOYMENT_DESIGN.md 9.6.


@dataclass(frozen=True)
class MapFrame:
    """Odometry tiles -> the sim's absolute map frame. **The only invented quantity in here.**

    `hero.pos_norm` is `hero.pos / (map_w, map_h)` and the deployed loop has neither half of it.
    Odometry's `position_tiles` is measured from wherever tracking started -- an arbitrary
    per-match origin -- and nothing on screen gives the map's extent. It is the same missing
    quantity that removed `zone.safe_area_frac`, in the one group that was signed off as fully
    suppliable, and it was found while writing this module rather than while auditing that one.

    Everything else in the observation is relative and needs none of this: velocities are
    differences, `entities.rel_*` are hero-relative, the grid is a hero-centred crop, and the zone
    margins are distances to gas. That isolation is why this is a small object and not a
    pervasive assumption -- change it here and nothing else in the package notices.

    **MEASURED, 2026-09-08 -- the origin is free and the scale is exact, so use the sim's own.**
    Priced at inference against `runs/mortis_deploy-20260907-041522/best_model.zip`, 250 episodes
    per tier, same reseeded scenarios each time (overall win rate; per-tier SE ~2.8 pp, overall
    ~1.4 pp):

    | pos_norm                 | overall | elite |
    |--------------------------|---------|-------|
    | true (baseline)          | 0.734   | 0.588 |
    | **+ 0.15 (offset)**      | 0.734   | 0.600 |
    | **+ 0.35 (offset)**      | 0.743   | 0.580 |
    | ~ U(0, 1) per step       | 0.720   | 0.564 |
    | 0                        | 0.687   | 0.468 |

    A constant offset -- right scale, wrong origin, which is precisely odometry anchored at spawn
    -- is **free at two different magnitudes**, including one that pushes the column past 1.0 and
    out of its trained range. Zeroing it is not: -4.7 pp overall and **-12 pp on elite**, 3.3 and
    4.3 standard errors. The asymmetry is the useful part -- the policy is not reading this column
    for "where am I on the map" (it would then care about the offset), it is reading it for
    something the offset preserves and zero destroys, and zero additionally *asserts* the specific
    claim "pinned in the top-left corner" every single tick. That is the never-feed-a-constant
    rule showing up as a number.

    So:

    * `origin = (map_w / 2, map_h / 2)`, putting odometry's arbitrary start at the map centre --
      the point with the most room before the column saturates in any direction.
    * `extent = (map_w, map_h)` -- **the SIM's map size, not a guess at the real one.** Odometry is
      in tiles and `hero.pos` is in tiles, so dividing by the sim's extent makes one tile of real
      movement produce exactly the pos_norm movement it produced in training. Guessing the actual
      Nulls Brawl map's size would introduce a *scale* error, which is a different perturbation
      from the one measured above and was not tested. This choice quarantines the entire
      uncertainty into the origin, which is the half that measured free.

    Both are settings, not measurements, and `explain()` says so to whoever reads a log.
    """

    origin: tuple[float, float]
    extent: tuple[float, float]

    def pos_norm(self, world_xy: tuple[float, float]) -> tuple[float, float]:
        """`(x, y)` in [0, 1]-ish. Not clamped: a value outside the unit square means the assumed
        extent is wrong, and silently squashing it would hide exactly that."""
        return ((world_xy[0] + self.origin[0]) / self.extent[0],
                (world_xy[1] + self.origin[1]) / self.extent[1])

    def explain(self) -> str:
        return (f"MapFrame(origin={self.origin}, extent={self.extent}) -- ASSUMED, not measured; "
                f"hero.pos_norm is the only field that reads it")


def _require(value, name: str):
    """A supplier that was not passed is an error, never a zero.

    This is the whole discipline of `configs/agent_obs_deploy.yaml` expressed as a function: a
    field the loop cannot fill is dropped from the spec, and a field that IS in the spec must be
    filled by something real. A `None` reaching a column here means a supplier failed upstream,
    and the loop's answer to that is to fail closed, not to substitute.
    """
    if value is None:
        raise ValueError(
            f"assemble: no value supplied for {name!r}. Every field in the deployed spec has a "
            f"supplier; a missing one is a perception failure and must fail closed rather than "
            f"be filled with a constant (BRAWL_DEPLOYMENT_DESIGN.md 9.8)."
        )
    return value


class ObservationAssembler:
    """One per match. Holds the spec, the config and the reusable output buffers.

    Stateless across ticks apart from those buffers: everything temporal lives in the trackers and
    the shadow, which is the same split `HealthReader`/`HealthTracker` uses upstream.
    """

    def __init__(self, spec, cfg, *, map_frame: MapFrame | None = None, device: str = "cpu"):
        """Takes the LOADED spec and config, not paths.

        `policy.py` resolves both out of the run's own `train.yaml` and passes them here, so the
        assembler and the checkpoint provably share one spec and one config. Re-loading them from
        paths here would make it possible for the observation to be built to a different spec than
        the network was trained on -- a mismatch that shows up as a shape error at best and a
        silent column shift at worst. `from_paths` exists for tests and one-offs.
        """
        self.cfg = cfg
        self.spec = spec
        self.device = device
        self.buffers = obs_select.make_agent_obs_buffers(self.spec, self.cfg, n_envs=1,
                                                         device=device)
        self.map_frame = map_frame or MapFrame(origin=(self.cfg.map_w / 2.0,
                                                       self.cfg.map_h / 2.0),
                                               extent=(float(self.cfg.map_w),
                                                       float(self.cfg.map_h)))
        self.episode_seconds = float(self.cfg.max_episode_steps) * float(self.cfg.dt)
        self._view = np.zeros((_N_VIEW_CHANNELS, self.cfg.view_h, self.cfg.view_w), np.uint8)
        self._grid_channels = self._channel_indices()

    @classmethod
    def from_paths(cls, spec_path, cfg_path, **kw) -> "ObservationAssembler":
        """Load a spec and config off disk. The deployed loop does NOT use this -- it goes through
        `policy.DeployedPolicy`, which reads both out of the run that trained the checkpoint."""
        from brawl_sim.config import load_config

        cfg = load_config(cfg_path)
        return cls(obs_select.load_agent_spec(spec_path, cfg), cfg, **kw)

    def _channel_indices(self) -> tuple[int, ...]:
        """Where each of `GridBuilder`'s planes belongs in the 12-channel `view`.

        Read off the loaded spec rather than off `GridSpec`, so that if the two ever name
        different channel lists this raises here instead of producing a correctly-shaped grid with
        its planes permuted -- which no shape check catches and no test that mocks one side sees.
        """
        grid = next((g for g in self.spec.groups if g.view_channels is not None), None)
        if grid is None:
            raise ValueError("the deployed spec has no grid group; assemble cannot place a view")
        return tuple(_CHANNEL_INDEX[ch] for ch in grid.view_channels)

    # -- the tick -------------------------------------------------------------

    def assemble(self, *, hero_pos, hero_vel, shadow, hero_hp, n_enemies_alive, elapsed_s,
                 hero_in_bush, hero_in_zone, enemies, enemy_hp, enemy_in_bush,
                 projectiles, zone, grid) -> dict:
        """One decision's observation, as `{group: np.ndarray}` with no leading batch axis.

        Every argument is a supplier's output and none has a default:

        * `hero_pos`, `hero_vel` -- world tiles and tiles/s, from `TrackerResult.hero`.
        * `shadow` -- `ShadowHero.observe()`.
        * `hero_hp` -- the HP numeral, from `HealthTracker` on the `player` box.
        * `n_enemies_alive` -- `BrawlersLeft.enemies_alive(hero_alive)`.
        * `elapsed_s` -- seconds since the match gate went true.
        * `enemies` -- `TrackerResult.enemies`, slot-ordered, `None` in empty slots.
        * `enemy_hp` -- `{slot: hp}`; a slot missing from it is not yet promoted (see the module
          docstring's promotion rule).
        * `enemy_in_bush`, `hero_in_bush`, `hero_in_zone` -- terrain lookups, done by the caller
          because it owns the occupancy map.
        * `projectiles` -- `ProjectileTracker.snapshot(hero_pos)`.
        * `zone` -- `{field: value}` for the spec's zone group, in the sim's own units.
        * `grid` -- `GridBuilder.build(...)`, `(len(spec channels), view_h, view_w)` uint8.
        """
        full = {}
        self._put_self(full, hero_pos, hero_vel, shadow, hero_hp, n_enemies_alive, elapsed_s,
                       hero_in_bush, hero_in_zone)
        self._put_entities(full, hero_pos, hero_vel, enemies, enemy_hp, enemy_in_bush)
        self._put_projectiles(full, projectiles)
        self._put_zone(full, zone)
        self._put_view(full, grid)

        out = obs_select.build_agent_obs(full, self.spec, self.cfg, self.buffers)
        # `.numpy()` on a buffer tensor SHARES its memory, so without the copy every tick's
        # observation silently rewrites the previous one -- anything that holds a reference (a
        # telemetry ring, a frame stack, a debug dump) reads the wrong frame with no error. The
        # buffers exist so `build_agent_obs` allocates nothing in the 300M-step training loop;
        # here it is one 3 KB copy per DECISION, four times a second, which is not the kind of
        # cost this package is watching for.
        return {name: t[0].cpu().numpy().copy() for name, t in out.items()}

    # -- groups ---------------------------------------------------------------

    def _f(self, *values) -> torch.Tensor:
        return torch.tensor([list(values)], dtype=torch.float32, device=self.device)

    def _put_self(self, full, hero_pos, hero_vel, shadow, hero_hp, n_enemies_alive, elapsed_s,
                  in_bush, in_zone) -> None:
        s = _require(shadow, "shadow.observe()")
        px, py = self.map_frame.pos_norm(_require(hero_pos, "hero_pos"))
        vx, vy = _require(hero_vel, "hero_vel")
        fx, fy = s["facing_vec"]
        dx, dy = s["dash_dir"]
        full["hero"] = {
            "pos_norm": self._f(px, py),
            "vel": self._f(vx, vy),
            "facing_vec": self._f(fx, fy),
            "hp": self._f(float(_require(hero_hp, "hero_hp")))[:, 0],
            "ammo_frac": self._f(s["ammo_frac"])[:, 0],
            "ammo_whole": self._f(s["ammo_whole"])[:, 0],
            "attack_cd": self._f(s["attack_cd"])[:, 0],
            "can_attack": self._b(s["can_attack"]),
            "dashing": self._b(s["dashing"]),
            "dash_t": self._f(s["dash_t"])[:, 0],
            "dash_dir": self._f(dx, dy),
            "invuln": self._b(s["invuln"]),
            "in_bush": self._b(_require(in_bush, "hero_in_bush")),
            "in_zone": self._b(_require(in_zone, "hero_in_zone")),
            "long_dash_ready": self._b(s["long_dash_ready"]),
            "long_dash_frac": self._f(s["long_dash_frac"])[:, 0],
            "super_ready": self._b(s["super_ready"]),
            "super_charge_frac": self._f(s["super_charge_frac"])[:, 0],
        }
        # Clamped, and the clamp is the part that matters: the sim truncates at time_frac == 1, so
        # the column's whole training range is bounded by 1 and a long match would feed it 1.33.
        frac = float(_require(elapsed_s, "elapsed_s")) / self.episode_seconds
        full["meta"] = {
            "time_frac": self._f(min(max(frac, 0.0), 1.0))[:, 0],
            "n_enemies_alive": self._f(float(_require(n_enemies_alive, "n_enemies_alive")))[:, 0],
        }

    def _b(self, value) -> torch.Tensor:
        return torch.tensor([bool(value)], dtype=torch.bool, device=self.device)

    def _put_entities(self, full, hero_pos, hero_vel, enemies, enemy_hp, enemy_in_bush) -> None:
        """The `entities` axis, hero at index 0 and enemy slot k at index k + 1.

        `obs_select` slices index 0 off (`_HERO_AXIS_PREFIX`), so the hero row here is never read
        and is left zero -- the one place in this module where zeros are correct, because they are
        discarded before the policy sees anything.
        """
        e = self.cfg.n_entities
        alive = torch.zeros((1, e), dtype=torch.bool, device=self.device)
        revealed = torch.zeros((1, e), dtype=torch.bool, device=self.device)
        in_bush = torch.zeros((1, e), dtype=torch.bool, device=self.device)
        rel_pos = torch.zeros((1, e, 2), dtype=torch.float32, device=self.device)
        rel_vel = torch.zeros((1, e, 2), dtype=torch.float32, device=self.device)
        dist = torch.zeros((1, e), dtype=torch.float32, device=self.device)
        hp = torch.zeros((1, e), dtype=torch.float32, device=self.device)

        hx, hy = _require(hero_pos, "hero_pos")
        hvx, hvy = _require(hero_vel, "hero_vel")
        for slot, track in enumerate(_require(enemies, "enemies")):
            if track is None:
                continue
            value = enemy_hp.get(slot)
            if value is None:
                # Tracked, but no HP has ever committed for it. Not promoted -- see the module
                # docstring. Leaving the slot empty is in-distribution; inventing an HP is not.
                continue
            i = slot + 1
            alive[0, i] = True
            revealed[0, i] = bool(track.seen_now)
            rx, ry = track.pos[0] - hx, track.pos[1] - hy
            rel_pos[0, i] = torch.tensor([rx, ry], device=self.device)
            dist[0, i] = math.hypot(rx, ry)
            rel_vel[0, i] = torch.tensor([track.vel[0] - hvx, track.vel[1] - hvy],
                                         device=self.device)
            hp[0, i] = float(value)
            in_bush[0, i] = bool(enemy_in_bush.get(slot, False))

        full["entities"] = {"alive": alive, "revealed_to_hero": revealed, "rel_pos": rel_pos,
                            "dist": dist, "rel_vel": rel_vel, "hp": hp, "in_bush": in_bush}

    def _put_projectiles(self, full, snapshot) -> None:
        """All `max_projectiles` slots, of which `obs_select` keeps the nearest K by
        `time_to_closest`. The selection has to happen there and not here: `max_slots` is a
        property of the spec, and doing it twice differently is how deployment and training end up
        looking at different projectiles.

        `in_view` is not in the spec but IS read, as the projectile group's fairness mask
        (`_FAIRNESS_MASK_FIELD`). Every tracked projectile is one the detector found on screen, so
        this is a fact about the tracker rather than a value being invented -- a projectile the
        camera never showed has no track.
        """
        p = self.cfg.max_projectiles
        alive = torch.zeros((1, p), dtype=torch.bool, device=self.device)
        in_view = torch.zeros((1, p), dtype=torch.bool, device=self.device)
        rel_pos = torch.zeros((1, p, 2), dtype=torch.float32, device=self.device)
        vel = torch.zeros((1, p, 2), dtype=torch.float32, device=self.device)
        ttc = torch.zeros((1, p), dtype=torch.float32, device=self.device)

        snapshot = _require(snapshot, "projectiles")
        if len(snapshot) > p:
            # More live projectiles than the sim can represent. Keeping the nearest is the same
            # rule `max_slots` applies one stage later, so the discard is consistent with it.
            snapshot = sorted(snapshot, key=lambda s: s[2])[:p]
        for i, (rel, v, t) in enumerate(snapshot):
            alive[0, i] = True
            in_view[0, i] = True
            rel_pos[0, i] = torch.tensor(rel, device=self.device)
            vel[0, i] = torch.tensor(v, device=self.device)
            ttc[0, i] = float(t)

        full["projectiles"] = {"alive": alive, "rel_pos": rel_pos, "vel": vel,
                               "time_to_closest": ttc, "in_view": in_view}

    def _put_zone(self, full, zone) -> None:
        """Whatever zone fields the loaded spec names, and nothing else.

        Driven by the spec rather than by a fixed list because there are two live spec versions:
        `agent_obs_deploy.yaml` wants four fields and `agent_obs_deploy2.yaml` wants
        `hero_margin_local` and `active` (9.16). A hardcoded list here would silently supply the
        wrong one after a retrain.
        """
        zone = _require(zone, "zone")
        wanted = [f.split(".", 1)[1] for g in self.spec.groups if g.name == "zone"
                  for f in g.fields]
        out = {}
        for name in wanted:
            value = _require(zone.get(name), f"zone.{name}")
            if name.startswith("hero_margin"):
                out[name] = self._f(*value)
            elif name == "active":
                out[name] = self._b(value)
            else:
                out[name] = self._f(float(value))[:, 0]
        full["zone"] = out

    def _put_view(self, full, grid) -> None:
        """`GridBuilder`'s planes scattered into the 12-channel sim `view`.

        The four unfilled channels are never read: `_build_grid_group` index-selects exactly the
        spec's own channels. Zeroing the whole array each tick rather than only the filled planes
        costs 3 KB of memset and removes the question of whether a stale plane could survive.
        """
        grid = _require(grid, "grid")
        if grid.shape != (len(self._grid_channels), self.cfg.view_h, self.cfg.view_w):
            raise ValueError(
                f"grid is {grid.shape}, expected "
                f"{(len(self._grid_channels), self.cfg.view_h, self.cfg.view_w)} -- the builder "
                f"and the spec disagree about the grid group"
            )
        self._view[:] = 0
        for plane, channel in enumerate(self._grid_channels):
            self._view[channel] = grid[plane]
        full["view"] = torch.from_numpy(self._view).unsqueeze(0).to(self.device)
