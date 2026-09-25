"""The dead-bin move mask -- the one place the deployment narrows the policy's MOVE choice.

WHY (2026-09-23). `2026-09-23 20-20-47.mp4`, 22-32 s: the hero walks into the wall east of him
for ten seconds. Replaying the clip through the live perception path put the terrain map in the
clear -- the cell east of the hero held 161-163 WALL votes with no dissent and the policy's own
`blocks_unit` plane showed it -- and the deploy4 checkpoint still chose bin 1 (east) at 40 of 45
decisions, at 0.77 probability on the exact observation it was handed. The sim has the same
habit: at the elite tier the checkpoint spends 2.2% of its decisions pushing into a wall on its
training maps and 4.4% on the two held-out ones, stalls for 2 s or more in a third of its
episodes, and has single stalls of 17 s. `core/hero.action_mask` builds the move half all-True,
so nothing in training ever told the policy a bin was pointless, and at inference the argmax
holds a pointless bin for as long as the observation stands still.

WHAT. A move bin is DEAD when the sim's own collision rule would leave the hero where it is.
`core/terrain.resolve_move` tries the x step alone, then the y step from wherever x landed, each
against `circle_blocked` -- the centre and eight probes at `unit_radius` -- on `blocks_unit`.
`legal_move_bins` is that rule ported to numpy for one hero on the grid the policy sees, False
exactly where `resolve_move` would hand back its input; tests/test_deployment_move_mask.py holds
the port against `resolve_move` itself on random planes. A dead bin is masked the way an
uncharged super is, and MaskablePPO takes the argmax of what is left.

MEASURED in the sim before shipping: 96 elite episodes per cell, the deploy4 checkpoint, the mask
computed from the agent's own grid plane, deterministic as deployed.

    | maps     | mask | wall-push | stalls >= 2 s / episode | longest stall | win  |
    |----------|------|-----------|-------------------------|---------------|------|
    | training | off  | 0.022     | 0.30                    | 69 decisions  | 0.26 |
    | training | on   | 0.000     | 0.00                    | 1             | 0.25 |
    | holdout  | off  | 0.044     | 0.38                    | 67            | 0.12 |
    | holdout  | on   | 0.001     | 0.00                    | 1             | 0.10 |

Win rates move within noise (SE ~4.5 pp), the stalls are gone, and about 3% of decisions change.

WHAT IT DOES NOT DO. It never invents a wall: the grid reads UNKNOWN as FLOOR (`GridBuilder`),
so a cell the map has not seen cannot kill a bin. And it does not trust the map over the hero:
the sim's hero is never closer to a wall than `unit_radius`, but the deployed one is an estimate
standing on an estimate, and the game lets a brawler press nearer to a wall than that. When the
hero's own footprint already overlaps a blocked cell, the footprint is halved -- up to four
times -- until it is free, and the rule runs with the clearance the hero actually has; a bin is
dead when it cannot move even that smaller circle. If the hero's CENTRE reads blocked, every bin
stays legal: a wall drawn under the hero is a map error, and pinning him to idle on a wrong map is
worse than letting the policy push. Neither branch has a counterpart in `terrain.py`.

Nor is it the training-side fix, and the habit wants one. The same replay found two things the
deployed observation cannot supply the way training did: `entities.revealed_to_hero` has no range
limit in the sim (`bots/perception.visibility`), so the policy saw at least one enemy on 95% of
its sim decisions and none at all for the whole stuck window live; and its east preference on
that frame was not a reaction to the wall -- mirroring the terrain sent it north-east and a lone
wall slab sent it away from the wall -- which, on 14 fixed training maps with a 0.53 -> 0.37
training-to-holdout win-rate gap in the run's own log, reads as memorised routes. Both are in
BRAWL_DEPLOYMENT_DESIGN.md 6.16; neither is this module's to fix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .perception.shadow import ShadowParams

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

# `terrain._N_PROBES`: the fixed probe ring `circle_blocked` samples, probe 0 at angle 0.
_N_PROBES = 8
_PROBES = np.array([(math.cos(k * 2 * math.pi / _N_PROBES), math.sin(k * 2 * math.pi / _N_PROBES))
                    for k in range(_N_PROBES)], dtype=np.float32)
# Below this a tick's displacement is float noise, not a move. The sim's directions are float32
# (`geo.dir_from_bin`), so a bin pointing straight into a wall still carries a ~1e-8 sideways
# component, which never moves a float32 position at map scale but would count as movement here.
_MOVED_TILES = 1e-6
# How many times the footprint may halve to clear the hero's own cell (module docstring): four
# takes 0.4 to 0.025 tiles, a fortieth of a cell, past which the centre is as good as inside.
_SHRINKS = 4


def circle_blocked(blocks: np.ndarray, pos, radius: float) -> np.ndarray:
    """`terrain.circle_blocked` on one `(rows, cols)` bool plane, for `(..., 2)` positions in its
    units -- `(column, row)`, floats. The centre and `_N_PROBES` probes at `radius`, blocked if
    any lands on a True cell; off the plane reads as blocked, as `terrain.oob` does off the map."""
    pos = np.asarray(pos, dtype=np.float32)
    pts = np.concatenate([pos[..., None, :], pos[..., None, :] + np.float32(radius) * _PROBES],
                         axis=-2)
    ix = np.floor(pts[..., 0]).astype(np.intp)
    iy = np.floor(pts[..., 1]).astype(np.intp)
    h, w = blocks.shape
    oob = (ix < 0) | (ix >= w) | (iy < 0) | (iy >= h)
    hit = blocks[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)] | oob
    return hit.any(axis=-1)


def legal_move_bins(blocks: np.ndarray, hero_xy, *, radius: float, step: float,
                    n_bins: int = 16) -> np.ndarray:
    """`(n_bins + 1,)` bool, index 0 the idle bin: True where the bin would move the hero.

    `blocks` is the `blocks_unit` plane, `(rows, cols)`; `hero_xy` the hero's position in that
    plane's units, `(column, row)` floats -- `hero_pos - GridBuilder.origin_tile(hero_pos)`;
    `step` one sim tick of walking, `move_speed * dt`; `radius` is `entities.unit_radius`. Idle
    is always legal; a footprint the map already blocks shrinks until free, and everything is
    legal when the hero's centre itself reads blocked (module docstring).
    """
    blocks = np.asarray(blocks).astype(bool)
    if blocks.ndim != 2:
        raise ValueError(f"blocks must be a (rows, cols) plane, got shape {blocks.shape}")
    pos = np.asarray(hero_xy, dtype=np.float32)
    if pos.shape != (2,):
        raise ValueError(f"hero_xy must be (column, row), got {hero_xy!r}")
    legal = np.ones(n_bins + 1, dtype=bool)
    radius = float(radius)
    for _ in range(_SHRINKS + 1):
        if not circle_blocked(blocks, pos, radius):
            break
        radius /= 2
    else:
        return legal
    theta = np.arange(n_bins, dtype=np.float32) * np.float32(2 * math.pi / n_bins)
    delta = np.float32(step) * np.stack([np.cos(theta), np.sin(theta)], axis=-1)   # (n_bins, 2)
    pos_x = np.stack([pos[0] + delta[:, 0], np.full(n_bins, pos[1], np.float32)], axis=-1)
    after_x = np.where(circle_blocked(blocks, pos_x, radius)[:, None], pos, pos_x)
    pos_y = np.stack([after_x[:, 0], after_x[:, 1] + delta[:, 1]], axis=-1)
    after_y = np.where(circle_blocked(blocks, pos_y, radius)[:, None], after_x, pos_y)
    legal[1:] = np.abs(after_y - pos).max(axis=-1) > _MOVED_TILES
    return legal


def unit_radius_tiles(env_config_path=None) -> float:
    """`entities.unit_radius` from `configs/default.yaml`, read the way `loop.dash_reach_tiles`
    reads it: it is a SimParams column rather than an `EnvConfig` field, and a literal here would
    drift the moment the sim's changed."""
    import yaml

    env = yaml.safe_load(Path(env_config_path or _CONFIGS_DIR / "default.yaml").read_text())
    try:
        raw = env["entities"]["unit_radius"]
    except KeyError as exc:
        raise KeyError(f"the move mask needs entities.unit_radius: {exc}") from exc
    if isinstance(raw, (list, tuple, dict)):
        raise ValueError(f"unit_radius is a randomized range {raw!r}; deployment needs one value")
    return float(raw)


@dataclass(frozen=True)
class MoveMask:
    """The rule with its constants bound: `radius` in tiles, `step` in tiles per sim tick."""

    radius: float
    step: float
    n_bins: int

    @classmethod
    def from_configs(cls, params: ShadowParams, sim_cfg, *, env_config_path=None) -> "MoveMask":
        """`step = move_speed * dt` from the shadow's brawler block and the checkpoint's own
        config, `radius` from `configs/default.yaml` -- the three sources the sim walks with."""
        return cls(radius=unit_radius_tiles(env_config_path),
                   step=float(params.move_speed) * float(sim_cfg.dt),
                   n_bins=int(sim_cfg.n_move_bins))

    def legal(self, blocks: np.ndarray, hero_xy) -> tuple[bool, ...]:
        return tuple(bool(v) for v in legal_move_bins(blocks, hero_xy, radius=self.radius,
                                                      step=self.step, n_bins=self.n_bins))
