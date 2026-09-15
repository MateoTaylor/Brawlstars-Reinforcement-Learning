"""The egocentric view grid, from the terrain map and the trackers. BRAWL_DEPLOYMENT_DESIGN.md 6.2.

`configs/agent_obs_deploy.yaml`'s `grid` group is eight `(13, 21)` uint8 planes centred on the
hero. Nothing here is new perception: the static planes are a crop of `terrain.OccupancyMap`, the
dynamic ones are a scatter of the tracks `tracker.py` and `projectiles.py` already produce, and
this module's whole job is to place them where `brawl_sim/core/observation.py:_build_grid` places
them. Every choice below is therefore a claim about matching that function.

#### The crop, and the one thing it makes free

`observation._view_origin` is `hero_tile - (view_w // 2, view_h // 2)` where `hero_tile` is the
FLOOR of the hero's position, and the sim can express nothing else (`configs/default.yaml`'s view
block says so in its own comment). Two consequences worth stating because they are easy to get
subtly wrong:

* The hero's cell is exactly the crop centre, always, by construction -- so the `hero` channel is
  `1` at `(6, 10)` and `0` everywhere else in EVERY training sample where the hero is alive, and
  all-zero where it is not. It is not a plane to project into; it is a constant plus an aliveness
  bit, and it is written that way here.
* The crop moves in whole tiles. Sub-tile hero motion changes nothing until it crosses a cell
  boundary, at which point the whole grid shifts by one. That is the sim's behaviour and so it is
  this module's; smoothing it would be a different observation.

#### Counts, not booleans, on the dynamic planes

`_scatter_count` does `+= 1` per occupant, so two enemies sharing a cell give `2`. The terrain
planes are 0/1 because they come from a per-tile lookup, but `enemy_revealed`, `hero`, `projectile`,
`box` and `pickup` are occupancy counts clamped to 255. Reproduced here.

#### Who supplies what

| channel | supplier | note |
|---|---|---|
| `blocks_unit`, `blocks_projectile`, `is_bush`, `is_water` | `OccupancyMap.best()` | one crop, one lookup table |
| `in_zone` | `terrain/zone.py`, accumulated | see "The gas is sticky" |
| `enemy_revealed` | `EntityTracker` | tracks with `seen_now`, and only those |
| `hero` | the shadow's `alive` | the constant above |
| `projectile` | `ProjectileTracker.live()` | `live()` already excludes coasted tracks |
| `box` | `LootMap.crates()` | sticky, one fixed cell per crate (`loot.py`) |
| `pickup` | `LootMap.cubes()` | sticky, one fixed cell per cube on screen |

`box` and `pickup` are what `configs/agent_obs_deploy3.yaml` added back after 9.11 dropped them for
want of a detector. They are sticky where every other dynamic plane here is "seen this tick", and
that is the sim's rule rather than a choice: `_build_grid` scatters every live crate and pickup in
the crop, visible or not, and `fair` gates neither. `loot.py` says what that costs and where the
two lists differ from the sim's. `enemy_any` and `enemy_hidden` are refused: they leak the position
of an enemy the hero cannot see, and `obs_select` already rejects them at spec-load time under
`fair: true`.

**`enemy_revealed` takes `Track.seen_now`, not "the track exists".** Under `fair: true` an
unrevealed enemy appears in NO grid channel at all, so a coasted track -- one carried forward on a
constant-velocity prediction because the detector missed it -- must not be deposited. It is also
exactly the rule `entities.revealed_to_hero` uses in the `enemies` group, so the two halves of the
observation cannot disagree about who is visible.

#### The gas is sticky, because it only ever grows

The zone reader judges the CURRENT frame, and the current frame does not cover the whole crop (see
below). The poison closes inward and never recedes, so a cell seen gassed is gassed for the rest of
the match -- accumulating is not a heuristic here, it is the game's rule. Without it a cell that
went under the joystick after being gassed would read "clear" and the policy would walk into it.

The deposit uses `ZoneMask.cells`, not `at_least(0.05)`. Those are different questions and
`zone.py` says so: `at_least` exists for the occupancy map's *abstain* path, where over-flagging is
free and missing gas casts a wrong vote every frame, while `cells` answers "is this cell in the
zone" for the agent, "where a wrong answer either way costs the same". The agent is the consumer
here. The sim samples each cell's CENTRE against the zone rectangle; `cells` thresholds a per-cell
gassed fraction. They disagree only on the cells the zone boundary cuts through.

The cost of sticky is that a false positive is permanent. `min_cell_fraction` is the guard, and the
failure it does not cover -- a full-screen tint misread as gas -- would flag a large contiguous
region at once and is worth watching for rather than defending against here.

#### MEASURED: what is UNKNOWN in the crop, where it sits, and what goes there

The occupancy map's own docstring instructs its consumers to treat UNKNOWN as its own state
("a policy that believes unexplored ground is walkable will walk into walls"). This observation
has no state to treat it as: the policy was trained on eight planes with fixed meanings, and there
is no ninth channel to add without another from-scratch run -- and no way to train one, since the
sim is fully observed and the channel would be constant `1` in every training sample, which is the
exact lie `agent_obs_deploy.yaml` exists to prevent. So UNKNOWN must map onto a real tile, and the
only question is which.

Measured over two clips at the 4 Hz decision rate, hero-centred crop, occupancy accumulating
normally (`scratchpad/crop_coverage.py`; the hero is approximated by the fixed screen-centre
camera-relative tile, which is what the +0.889 world-position-on-odometry slope licenses):

    clip                      median UNKNOWN   worst   after 10 s   final
    day12_recording1 (BlueStacks)      8.8%    17.2%        1.5%     5.9%
    showdown_alternate_map2            1.5%     ~6%         0.0%     0.0%

and the per-cell rate, which is the part that decides it -- rows are crop rows, `@` is the hero:

    .....................        The unknown region is the HUD. The joystick sits bottom-left and
    .....................        the attack buttons bottom-right, they are opaque, and the camera
    .....................        holds the hero nearly still on screen -- so the hole is roughly
    .....................        HERO-fixed and does not sweep out. It closes when the hero walks
    .....................        into already-seen ground and reopens at the exploration frontier
    .....................        (the curve goes 17% -> 0% by t=12 s -> 11% by t=20 s).
    ..........@..........
    44442................        NOTHING above the hero row is ever unknown, on either clip. The
    66653...........222..        21x13 view was sized to fit inside the visible trapezoid
    66653...........222..        (configs/default.yaml), so the only hole is the UI.
    76664...........222..
    77764.........22222..        Tenths of the ticks that cell was UNKNOWN. day12_recording1;
    77764.........33.....        showdown_alternate_map2 has the same shape, one third the height.

**Filling UNKNOWN with WALL is the tempting choice and it is the wrong one.** It matches the sim's
out-of-map padding (`maps/loader.py` pads `pad_tiles` with `Tile.WALL`) and it is conservative in
the abstract. But the table above says the error would not be spread evenly: it would put a
phantom wall in the bottom-left of the observation, on 40-77% of ticks, **and that phantom would
follow the hero**, because the hole is hero-fixed. A policy whose entire kit is a dash would be
told, permanently, that it cannot retreat down-left. A systematic lie that tracks the agent is a
far worse failure than an unbiased one.

**So UNKNOWN is FLOOR** -- all four static planes zero. Measured, FLOOR is 61% / 73% of observed
crop cells on the two clips, so it is also the modal class. The remaining error splits by channel
in a way worth recording: on `blocks_projectile`, zero is the CONSERVATIVE answer (believe you are
exposed, and that cover you cannot see does not exist); on `blocks_unit` it is the optimistic one
(the hero may dash at a wall it cannot see, which the game clips and costs one dash). That is the
trade, taken deliberately, and `unknown_tile` is a constructor argument so it can be taken back.

Two caveats on the numbers. They are a mild UNDER-estimate: the hero sits +0.80 tiles below the
viewport centre (`configs/default.yaml`), so the real crop is about one row further into the HUD
than the approximation used. And out-of-map ground is a separate case that mostly does not arise --
when the hero stands at the map edge the ground beyond it is on screen and gets classified as
whatever the game draws there, because the hole is only ever BELOW.

#### The interlock is `occupancy.update`'s, not a new one

Only the gas accumulation touches odometry here; the tracks arrive already in the world frame and
already gated. It follows the same three rules: deposit only at `status == "ok"`, drop everything
on a segment change, and hold the segment as `None` until the first tick so that first tick adopts
it instead of reading as a change.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from brawl_sim.constants import (TILE_BLOCKS_PROJ, TILE_BLOCKS_UNIT, TILE_IS_BUSH, TILE_IS_WATER,
                                 Tile)
from brawl_vision.terrain.labeling import CLASSES
from brawl_vision.terrain.occupancy import UNKNOWN

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_DEPLOY_SPEC = _CONFIGS_DIR / "agent_obs_deploy.yaml"
_DEFAULT_CFG = _CONFIGS_DIR / "default.yaml"

# Which of `_build_grid`'s twelve channels this module can fill, and how. The static four are
# columns of the terrain lookup table; the rest are written by hand.
_STATIC_COLUMN = {"blocks_unit": 0, "blocks_projectile": 1, "is_bush": 2, "is_water": 3}
_DYNAMIC = ("in_zone", "enemy_revealed", "hero", "projectile", "box", "pickup")

# Why the other two are refused, verbatim enough to act on.
_REFUSED = {
    "enemy_any": "leaks the position of an enemy the hero cannot see; obs_select refuses it "
                 "under fair:true, and so does this",
    "enemy_hidden": "leaks hidden enemies, same as enemy_any",
}

_COUNT_MAX = 255            # `_build_grid` clamps its occupancy counts before the uint8 cast


def _tile_lut(unknown_tile: Tile) -> np.ndarray:
    """`(len(CLASSES) + 1, 4)` uint8 -- `[blocks_unit, blocks_projectile, is_bush, is_water]` per
    terrain class, with one extra row standing in for UNKNOWN.

    Built by reading `brawl_sim.constants`' own tables rather than restating them, which is the
    one part of `CONVENTIONS.md` that does bind this package: if the simulator ever decides a
    fence stops a bullet, this follows without an edit.
    """
    tiles = [*CLASSES, unknown_tile]
    return np.array(
        [[int(TILE_BLOCKS_UNIT[t]), int(TILE_BLOCKS_PROJ[t]),
          int(TILE_IS_BUSH[t]), int(TILE_IS_WATER[t])] for t in tiles], np.uint8)


@dataclass(frozen=True)
class GridSpec:
    """The shape of the grid group, read from the same files training read."""
    channels: tuple[str, ...]
    view_h: int
    view_w: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.channels), self.view_h, self.view_w)

    @property
    def centre(self) -> tuple[int, int]:
        """`(row, col)` of the hero's own cell -- `_view_origin`'s floor division, restated."""
        return (self.view_h // 2, self.view_w // 2)

    @classmethod
    def load(cls, spec_path=None, cfg_path=None, group: str = "grid") -> "GridSpec":
        """Channels from the agent-obs spec, view dimensions from the sim config.

        Reading both rather than hardcoding `(8, 13, 21)` is the same discipline
        `ShadowParams.load` applies to `configs/brawlers.yaml`: the deployed grid has to be the
        grid the checkpoint was trained on, and the run directory's archived `train.yaml` names
        the spec by path. A constant here would go stale silently the first time either moved.
        """
        raw = yaml.safe_load(Path(spec_path or _DEPLOY_SPEC).read_text())
        for g in raw["groups"]:
            if g["name"] == group:
                channels = tuple(g["view_channels"])
                break
        else:
            raise KeyError(f"{spec_path or _DEPLOY_SPEC} has no group named {group!r}")

        view = yaml.safe_load(Path(cfg_path or _DEFAULT_CFG).read_text())["view"]
        spec = cls(channels=channels, view_h=int(view["height"]), view_w=int(view["width"]))
        spec.check()
        return spec

    @classmethod
    def from_agent_spec(cls, agent_spec, cfg) -> "GridSpec":
        """From an already-loaded `obs_select.AgentObsSpec` and `EnvConfig`: the pair the policy
        was built from, so the grid cannot be built for a different spec than the one assembling
        it. `load` with no arguments reads `agent_obs_deploy.yaml`, which a deploy3 checkpoint
        would get wrong by two channels. The grid group is found the way `assemble.py` finds it,
        as the one with `view_channels`."""
        g = next((g for g in agent_spec.groups if g.view_channels is not None), None)
        if g is None:
            raise KeyError("the agent spec has no grid group (no group with view_channels)")
        spec = cls(channels=tuple(g.view_channels), view_h=int(cfg.view_h), view_w=int(cfg.view_w))
        spec.check()
        return spec

    def check(self) -> None:
        for ch in self.channels:
            if ch in _REFUSED:
                raise ValueError(f"grid channel {ch!r} has no supplier at deploy time: "
                                 f"{_REFUSED[ch]}")
            if ch not in _STATIC_COLUMN and ch not in _DYNAMIC:
                raise ValueError(
                    f"unknown grid channel {ch!r}; this builder fills "
                    f"{sorted([*_STATIC_COLUMN, *_DYNAMIC])}")


class GasMap:
    """Where poison gas has been seen, in the occupancy grid's world frame. Sticky by design.

    Same shape and same `origin` as the `OccupancyMap` it accompanies, so one pair of crop indices
    serves both. It is a separate object rather than a channel of the occupancy grid because the
    two have opposite update rules: occupancy keeps voting and REFUSES to vote on a gassed cell,
    while this one latches on first sight and never reconsiders.
    """

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.gassed = np.zeros((height, width), bool)
        self.seen = np.zeros((height, width), bool)
        # None, not -1 -- the first tick adopts odometry's segment. Same reasoning as
        # `EntityTracker.__init__`: a sentinel makes the first tick of every match a reset.
        self._segment: int | None = None

    @classmethod
    def from_occupancy(cls, occupancy) -> "GasMap":
        return cls(occupancy.height, occupancy.width)

    @property
    def origin(self) -> tuple[int, int]:
        """World tile `(x, y)` at grid index `[0, 0]` -- `OccupancyMap.origin`'s formula."""
        return (-(self.width // 2), -(self.height // 2))

    def reset(self, segment: int) -> None:
        self.gassed[:] = False
        self.seen[:] = False
        self._segment = segment

    def update(self, zone, plan, odometry) -> int:
        """Deposit one frame's `ZoneMask`. Returns the number of cells newly marked gassed.

        Gated exactly as `occupancy.update` is: a new segment drops the map (the world frame has
        no defined offset to the old one), and anything but `status == "ok"` deposits nothing.
        """
        if self._segment is None:
            self._segment = odometry.segment
        elif odometry.segment != self._segment:
            self.reset(odometry.segment)
            return 0
        if odometry.status != "ok":
            return 0

        cols, rows = plan.size_tiles
        px, py = odometry.position_tiles
        ox, oy = self.origin
        # Round only here, for the same reason occupancy does: `position_tiles` stays continuous
        # so a run of sub-tile steps accumulates instead of being discarded a fraction at a time.
        col0 = int(round(plan.origin_tile[0] + px)) - ox
        row0 = int(round(plan.origin_tile[1] + py)) - oy

        r_lo, r_hi = max(row0, 0), min(row0 + rows, self.height)
        c_lo, c_hi = max(col0, 0), min(col0 + cols, self.width)
        if r_hi <= r_lo or c_hi <= c_lo:
            return 0
        sub = (slice(r_lo - row0, r_hi - row0), slice(c_lo - col0, c_hi - col0))
        dst = (slice(r_lo, r_hi), slice(c_lo, c_hi))

        fresh = zone.cells[sub] & ~self.gassed[dst]
        self.gassed[dst] |= zone.cells[sub]
        self.seen[dst] |= zone.observed[sub]
        return int(fresh.sum())


class GridBuilder:
    """Terrain map + tracks -> the `(C, view_h, view_w)` uint8 array the `grid` group wants.

    One instance per match, holding the gas accumulation. The `OccupancyMap` is passed in rather
    than owned: the terrain stage builds and feeds it, and two owners of one map is how a reset
    gets applied to half of the pipeline.
    """

    def __init__(self, occupancy, spec: GridSpec | None = None, *,
                 unknown_tile: Tile = Tile.FLOOR):
        self.occupancy = occupancy
        self.spec = spec or GridSpec.load()
        self.spec.check()
        self.unknown_tile = unknown_tile
        self.gas = GasMap.from_occupancy(occupancy)
        self._lut = _tile_lut(unknown_tile)

    def reset(self, segment: int) -> None:
        """The gas half only. `OccupancyMap` resets itself from inside its own `update`, and
        calling both from here would double-reset it on the tick it already handled."""
        self.gas.reset(segment)

    def observe_zone(self, zone, plan, odometry) -> int:
        return self.gas.update(zone, plan, odometry)

    # -- the crop ------------------------------------------------------------

    def origin_tile(self, hero_pos) -> tuple[int, int]:
        """`observation._view_origin`: the FLOOR of the hero's tile, minus half the view."""
        hx = int(np.floor(hero_pos[0])) - self.spec.view_w // 2
        hy = int(np.floor(hero_pos[1])) - self.spec.view_h // 2
        return (hx, hy)

    def _crop(self, source: np.ndarray, origin: tuple[int, int], fill) -> np.ndarray:
        """`(view_h, view_w)` window of a world-frame array, `fill` outside it.

        Falling off a 128x128 grid takes 64 tiles of travel from wherever tracking started, which
        no Showdown map allows -- but a clipped read here would silently wrap or raise, so the
        window is composed rather than sliced.
        """
        h, w = self.spec.view_h, self.spec.view_w
        ox, oy = self.gas.origin
        out = np.full((h, w), fill, source.dtype)
        c0, r0 = origin[0] - ox, origin[1] - oy
        c_lo, c_hi = max(c0, 0), min(c0 + w, source.shape[1])
        r_lo, r_hi = max(r0, 0), min(r0 + h, source.shape[0])
        if r_hi > r_lo and c_hi > c_lo:
            out[r_lo - r0:r_hi - r0, c_lo - c0:c_hi - c0] = source[r_lo:r_hi, c_lo:c_hi]
        return out

    def build(self, hero_pos, *, alive: bool = True, enemies=(), projectiles=(), crates=(),
              cubes=(), out: np.ndarray | None = None) -> np.ndarray:
        """One decision's grid.

        `hero_pos` is world tiles. `enemies` is `TrackerResult.enemies` -- slot-ordered, with
        `None` in empty slots -- and only tracks with `seen_now` are deposited. `projectiles` is
        `ProjectileTracker.live()`. `crates` and `cubes` are `LootMap.crates()`/`cubes()`, world
        `(x, y)` tuples. `alive` is the shadow's, and is the only thing that empties the `hero`
        channel.
        """
        h, w = self.spec.view_h, self.spec.view_w
        if out is None:
            out = np.zeros(self.spec.shape, np.uint8)
        else:
            out[:] = 0
        origin = self.origin_tile(hero_pos)

        best = self._crop(self.occupancy.best(), origin, UNKNOWN)
        # UNKNOWN is -1; the lookup table's last row is what it stands for. Mapping it to an index
        # rather than indexing with -1 matters: numpy would happily read the LAST row anyway and
        # then break the day a class is appended.
        codes = np.where(best < 0, len(CLASSES), best).astype(np.intp)
        static = self._lut[codes]                                  # (h, w, 4)

        gassed = self._crop(self.gas.gassed, origin, False)

        for i, ch in enumerate(self.spec.channels):
            col = _STATIC_COLUMN.get(ch)
            if col is not None:
                out[i] = static[:, :, col]
            elif ch == "in_zone":
                out[i] = gassed
            elif ch == "enemy_revealed":
                self._scatter(out[i], origin,
                              [t.pos for t in enemies if t is not None and t.seen_now])
            elif ch == "projectile":
                self._scatter(out[i], origin, [p.pos for p in projectiles])
            elif ch == "box":
                self._scatter(out[i], origin, crates)
            elif ch == "pickup":
                self._scatter(out[i], origin, cubes)
            elif ch == "hero":
                if alive:
                    r, c = self.spec.centre
                    out[i, r, c] = 1
        return out

    def _scatter(self, plane: np.ndarray, origin: tuple[int, int], positions) -> None:
        """`_scatter_count`'s `+= 1` per occupant, clamped, out-of-window entries dropped.

        A plain loop: there are at most nine enemies and a couple of dozen projectiles, and
        `CONVENTIONS.md`'s vectorization rules do not govern this package (see the boundary note
        in `brawl_deployment/__init__.py`).
        """
        h, w = self.spec.view_h, self.spec.view_w
        for x, y in positions:
            cx = int(np.floor(x)) - origin[0]
            cy = int(np.floor(y)) - origin[1]
            if 0 <= cx < w and 0 <= cy < h and plane[cy, cx] < _COUNT_MAX:
                plane[cy, cx] += 1
