"""`scripts/probes/replay_clip.py` with the odometry overlay rule off, or on only in a match.

    python scripts/probes/odometry_overlay/replay_overlay.py off CLIP [replay_clip.py args]
    python scripts/probes/odometry_overlay/replay_overlay.py match CLIP [replay_clip.py args]

`off` runs the odometry at `overlay_share: 1.0`, the estimator from before 2026-11-05. `match`
runs the rule only while the loop is PLAYING. Its votes outside a match (lobby, loading, results)
then cannot move the world origin the localizer fixes on at the gate. Both leave
configs/vision.yaml alone. For the rule as shipped, run replay_clip.py itself.

MAP_ONBOARDING.md's Truth section has what these showed: the crate spread's rise with the rule on
is the origin, not the rule's work in a match.
"""
import dataclasses
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import brawl_deployment.loop as loop  # noqa: E402
import brawl_vision.terrain.odometry as odometry  # noqa: E402

if len(sys.argv) < 3 or sys.argv[1] not in ("off", "match"):
    sys.exit(__doc__)
MODE = sys.argv.pop(1)

_loops = []
_loop_init = loop.DeployLoop.__init__


def _remember(self, *args, **kwargs):
    _loop_init(self, *args, **kwargs)
    _loops.append(self)


_update = odometry.Odometry.update


def _update_with_mode(self, *args, **kwargs):
    share = self.__dict__.setdefault("_shipped_share", self.cfg.odometry_overlay_share)
    playing = bool(_loops) and _loops[-1].phase is loop.Phase.PLAYING
    if MODE == "off" or not playing:
        share = 1.0
    if self.cfg.odometry_overlay_share != share:
        self.cfg = dataclasses.replace(self.cfg, odometry_overlay_share=share)
    return _update(self, *args, **kwargs)


loop.DeployLoop.__init__ = _remember
odometry.Odometry.update = _update_with_mode

target = REPO / "scripts" / "probes" / "replay_clip.py"
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
