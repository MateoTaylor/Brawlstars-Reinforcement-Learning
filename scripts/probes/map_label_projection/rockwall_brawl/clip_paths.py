"""Which recording the Rockwall Brawl probes read: `$CLIP`, by default the one recording so far.

    CLIP=bluestacks-rockwall-brawl-<spawn> .venv/Scripts/python.exe coverage.py

coverage.py writes the clip's coverage and the other probes read it. The first clip's goes to
runs/map_checks/rockwall_brawl/coverage.json, any other clip's to coverage_<CLIP>.json beside it.
S is the scale the clip's own registration used, read from its reg_at_scale.npz. A copy of Doom
Shroom's (../doom_shroom/clip_paths.py).
"""
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
FIRST_CLIP = "bluestacks-rockwall-brawl-center-east"
CLIP = os.environ.get("CLIP", FIRST_CLIP)
CHECKS = REPO / "runs" / "map_checks" / "rockwall_brawl"
WORK = CHECKS / "projection" / CLIP
COVERAGE = CHECKS / ("coverage.json" if CLIP == FIRST_CLIP else f"coverage_{CLIP}.json")


def scale() -> tuple[float, float]:
    meta = json.loads(str(np.load(WORK / "reg_at_scale.npz", allow_pickle=True)["meta"]))
    return tuple(float(v) for v in meta["S"])
