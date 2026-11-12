"""Shared names and paths for the Starfruit Supernova label projection (see README.md).

Importing this puts the repo root on sys.path, so the sibling scripts import it before any
brawl_* package. Intermediate results go to WORK (under runs/, gitignored): reg.npz,
reg2_<sx>_<sy>.npz, lab/aux_<frame>.npz and lab/rect_<frame>.png.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MAP = "starfruit_supernova"
CLIP = "bluestacks-starfruit-supernova-west-upper"
MAP_DIR = REPO / "brawl_deployment" / "data" / "maps"
WORK = REPO / "runs" / "map_checks" / MAP / "projection"
