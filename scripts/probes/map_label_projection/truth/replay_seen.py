"""replay_clip.py, unchanged, plus: the label check's seen map saved to $SEEN_OUT (npz: seen chars,
view-agreement confidence), and any extra terrain weights registered as $EXTRA_TERRAIN=name=path."""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "probes"))
sys.path.insert(0, str(REPO))
from brawl_vision import config as vconfig     # noqa: E402
import replay_clip as rc                        # noqa: E402

for spec in filter(None, os.environ.get("EXTRA_TERRAIN", "").split(";")):
    name, path = spec.split("=", 1)
    vconfig.TERRAIN_MODELS[name] = Path(path)

original = rc.LabelCheck.report


def report(self, known):
    original(self, known)
    rows, cols = known.tiles.shape
    r0, c0 = self.map.height // 2 - rows // 2, self.map.width // 2 - cols // 2
    np.savez(os.environ["SEEN_OUT"], seen=self.map.to_chars()[r0:r0 + rows, c0:c0 + cols],
             conf=self.map.confidence()[r0:r0 + rows, c0:c0 + cols])
    print(f"    seen map -> {os.environ['SEEN_OUT']}")


rc.LabelCheck.report = report
sys.exit(rc.main(sys.argv[1:]))
