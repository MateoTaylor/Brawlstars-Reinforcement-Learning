"""Which recording the Doom Shroom probes read: `$CLIP`, by default the first one.

    CLIP=bluestacks-doom-shroom-north-center .venv/Scripts/python.exe coverage.py

coverage.py writes the clip's coverage and the other probes read it. The first clip's stays at
runs/map_checks/doom_shroom/coverage.json, where MAP_ONBOARDING.md's numbers came from. Any other
clip's goes to coverage_<CLIP>.json beside it. S is the scale the clip's own registration used,
read from its reg_at_scale.npz: the first clip's is (1.021, 1.011), set by `--scale`.
"""
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
FIRST_CLIP = "bluestacks-doom-shroom-south-center"
CLIP = os.environ.get("CLIP", FIRST_CLIP)
CHECKS = REPO / "runs" / "map_checks" / "doom_shroom"
WORK = CHECKS / "projection" / CLIP
COVERAGE = CHECKS / ("coverage.json" if CLIP == FIRST_CLIP else f"coverage_{CLIP}.json")


def scale() -> tuple[float, float]:
    meta = json.loads(str(np.load(WORK / "reg_at_scale.npz", allow_pickle=True)["meta"]))
    return tuple(float(v) for v in meta["S"])
