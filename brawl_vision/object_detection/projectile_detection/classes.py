"""The label strings this model is trained on, spelled once.

They are CVAT's labels verbatim, and they travel unchanged from the export's `data.yaml` into the
ONNX metadata and out of `Detection.label` -- so they are the key every consumer matches on:
`object_detection/draw.py`'s colour table, and whatever decides which detections are projectiles.
A label renamed in CVAT would otherwise go missing silently from each of those at once.

**The class ids are NOT here, deliberately.** The export decides the order and it has already
changed once: the one-class export had `Projectile` at 0, the three-class one has it at 2. The ids
are read from the export's `data.yaml` when training and from the ONNX metadata at inference, and
`prepare.py` only checks that the names are ones listed below.
"""
PROJECTILE = "Projectile"
CUBE_BOX = "Power Cube Box"          # the crate, intact -- the HP number above it is in the box
CUBE_DROPPED = "Power Cube Dropped"  # the cube left behind when a crate or a brawler dies

KNOWN = frozenset({PROJECTILE, CUBE_BOX, CUBE_DROPPED})
