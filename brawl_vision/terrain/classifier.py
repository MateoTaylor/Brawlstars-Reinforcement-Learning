"""Per-cell terrain classification. See Terrain_Perception_Build_Plan.md Phase H.

**One small fully-convolutional network over the whole rectified patch**, with a total stride of
`pixels_per_tile`, emitting one set of class logits per cell. Not a U-Net -- there is no boundary to
recover, only a label per cell -- and not a sliding-window per-tile classifier, which would redo
almost all of the same convolutions once per cell for the same answer.

**The receptive field must span at least two tiles, and that is a requirement rather than a
preference.** Phase C measured the wall parallax at ~0.88 tiles: a wall's visible top face sits
nearly a whole cell away from the ground footprint that actually blocks movement. A head that sees
only its own cell literally cannot see the wall it is being asked to report. `TerrainNet` ends with
a 3x3 layer at full stride, giving 161 px -- 3.4 tiles -- and `receptive_field_px` exists so a test
can fail if someone trims the architecture below that.

**Handing it the wall-top plane as well was measured, and bought nothing (2026-09-15).** The same
frame rectified through `S @ inv(CameraModel.H_top)` puts each wall's top face IN its footprint
cell, so stacking that with the ground rectification as 6 input channels gives the head its wall
locally. Both inputs were retrained from scratch on the shipped recipe: the same 54 labelled frames,
and the map built from views 1-10 s either side of each label (`occupancy.py`'s protocol, on which
the shipped weights score 0.942). Blocking F1:

                           all windows        BlueStacks windows   BlueStacks labels held out
                           seed 0   seed 1    seed 0   seed 1      seed 0
    ground only            0.944    0.939     0.927    0.923       0.747
    ground + wall-top      0.939    0.944     0.922    0.913       0.723

The bar for shipping it was to beat ground-only on both seeds and on the held-out clips. It won
seed 1 and lost seed 0 and the held-out clips. The seeds disagree by more than the inputs do.
Presumably the 3.4-tile field already reaches the top a cell north. The second plane would also
cost a second warp per frame and double the augmentation work, so it is not shipped. `H_top` is
used elsewhere: `brawl_deployment/perception/lattice.py` is checked against it.

The held-out column is the bigger finding. Trained without their own three labels, the three
BlueStacks recordings -- the deployment setup -- score 0.75, against 0.92 with them (146 blocking
cells, so a small sample). The live map rests on three labelled frames, and more BlueStacks labels
look like the likeliest next gain.

Those were all under the phone's HUD mask. Under each label's own (2026-09-15), holding out one
BlueStacks recording at a time (`scripts/vision_score_map.py --hold-out-each`), so each model still
trains on the other two:

                           example-new   example3   example-zone   pooled
    blocking cells              49           66            9          124
    shipped weights           0.938        0.946        0.762        0.927
    held out                  0.918        0.846        0.571        0.851

One blocking cell moves recall by 0.008 on 124 and by 0.11 on nine, so example-zone's column is
noise. A labelling round is judged on the enlarged label set, the new frames on the held-out
recording included (the script's docstring has the procedure).

**Augmentation is the reskin strategy, not a regularisation detail.** Brawl Stars reskins maps, so a
classifier that memorises one palette is worthless on the next event. Hue rotation attacks exactly
that: it destroys absolute colour while leaving structure and texture, which is the signal a human
player is using too. Geometric augmentation is limited to LEFT-RIGHT flips: the camera tilts toward
the viewer, so a wall's top face always sits north of its footprint, and a vertical flip or a
90-degree rotation shows the net walls whose parallax points the wrong way. See `flip_pair`.

**Two exclusions, both from Section 2**: cells under Phase G's gas mask and cells occluded by a
loot box are never trained on and never voted on. `-1` marks them, and the loss ignores it. So does
every cell the label's own HUD mask covers (`labeling.LabelGrid.hud`); the net still sees those
pixels, since rectification blanks nothing, but is never asked what they are.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..camera import RectifyPlan
from ..config import TERRAIN_WEIGHTS_PATH, VisionConfig
from brawl_sim.constants import Tile

from .labeling import CLASS_INDEX, CLASSES, observed_cells

IGNORE_INDEX = -1


class TerrainNet(nn.Module):
    """Rectified patch -> per-cell logits. Total stride 48, matching `pixels_per_tile`.

    The strides multiply to 48 as 2*2*2*2*3 rather than a power of two, because the output grid has
    to land exactly on the tile grid: a stride-32 tower would emit 45x28.5 cells for a 1440x912
    patch, and resampling that onto 30x19 would smear every cell into its neighbours.
    """

    def __init__(self, n_classes: int = len(CLASSES), width: int = 16):
        super().__init__()
        w = width
        def block(cin, cout, k, s, p):
            # GroupNorm, NOT BatchNorm. Training feeds one frame at a time, and at batch size 1
            # BatchNorm normalises each frame by its own statistics while inference switches to
            # running averages that match no individual frame. Measured on this exact model: 1.000
            # accuracy in train() mode against 0.870 in eval() mode -- on its OWN training data.
            # Inference is the only mode that ships, so that gap was pure loss. GroupNorm computes
            # the same statistics in both modes and does not care about batch size.
            return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False),
                                 nn.GroupNorm(8, cout), nn.ReLU(inplace=True))
        self.features = nn.Sequential(
            block(3, w, 5, 2, 2),            # stride 2
            block(w, w * 2, 3, 2, 1),        # stride 4
            block(w * 2, w * 3, 3, 2, 1),    # stride 8
            block(w * 3, w * 4, 3, 2, 1),    # stride 16
            block(w * 4, w * 6, 3, 3, 1),    # stride 48 -- the tile grid
            block(w * 6, w * 6, 3, 1, 1),    # full-stride context: this is what spans >2 tiles
        )
        self.head = nn.Conv2d(w * 6, n_classes, 1)

    def forward(self, x):
        return self.head(self.features(x))

    @staticmethod
    def receptive_field_px() -> int:
        """Analytic receptive field of one output cell, in input pixels.

        Computed rather than hard-coded so that trimming a layer changes the number a test checks,
        instead of quietly invalidating the comment above it.
        """
        rf, stride = 1, 1
        for k, s in ((5, 2), (3, 2), (3, 2), (3, 2), (3, 3), (3, 1)):
            rf += (k - 1) * stride
            stride *= s
        return rf


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def to_tensor(rect: np.ndarray) -> torch.Tensor:
    """BGR uint8 patch -> (1, 3, H, W) float in [0, 1], RGB order."""
    rgb = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def augment(rect: np.ndarray, rng: np.random.Generator, hue_deg: float = 40.0,
            sat: float = 0.35, val: float = 0.35) -> np.ndarray:
    """Colour-jitter one patch. Hue is rotated on OpenCV's 0-179 wheel, so `hue_deg` is in real
    degrees and halved on the way in -- getting that wrong is a silent 2x and the single easiest
    mistake in this file."""
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(rng.uniform(-hue_deg, hue_deg) / 2.0)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + rng.uniform(-sat, sat)), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + rng.uniform(-val, val)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def flip_pair(rect: np.ndarray, target: np.ndarray, rng: np.random.Generator):
    """A random LEFT-RIGHT flip applied consistently to patch and label grid. Nothing else.

    This used to draw from all eight dihedral transforms on the grounds that terrain has no
    canonical orientation. The terrain doesn't, but the camera does: it is tilted, so every wall's
    visible top face sits ~0.88 tiles NORTH of the footprint the label marks (Phase C). A left-right
    flip preserves that; a vertical flip or a 90-degree rotation turns it into parallax the game
    never shows, and the net spends capacity on it. Measured 2026-09-14 on the 54 label frames, with
    the accumulated map scored on each label's own lattice: 200 epochs with the dihedral set reached
    blocking F1 0.921; with left-right only, 0.930 (precision 0.965).
    """
    if rng.random() < 0.5:
        rect = rect[:, ::-1].copy()
        target = target[:, ::-1].copy()
    return rect, target


@dataclass
class Example:
    rect: np.ndarray            # rectified BGR patch
    target: np.ndarray          # (rows, cols) int64, IGNORE_INDEX where not supervised
    clip: str


def build_examples(pairs, plans, exclude: dict | None = None) -> list[Example]:
    """`pairs` is an iterable of `(LabelGrid, rect)`.

    `plans` is `{hud: plan}` (`labeling.hud_plans`), and each label is taken against the HUD mask
    it records; one `RectifyPlan` instead serves every label alike. A cell its plan does not observe
    is never a target, whatever the file says. `exclude[key]` optionally supplies a boolean
    (rows, cols) of cells to drop -- gas, boxes -- keyed by `(clip, frame)`."""
    out = []
    for grid, rect in pairs:
        plan = plans[grid.hud] if isinstance(plans, dict) else plans
        grid.check_plan(plan)
        target = grid.as_class_index().astype(np.int64)
        target[target < 0] = IGNORE_INDEX
        target[~observed_cells(plan)] = IGNORE_INDEX
        drop = (exclude or {}).get((grid.clip, grid.frame))
        if drop is not None:
            target[drop] = IGNORE_INDEX
        out.append(Example(rect=rect, target=target, clip=grid.clip))
    return out


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------

@dataclass
class TerrainClassifier:
    net: TerrainNet
    device: str = "cpu"

    def predict(self, rect: np.ndarray, plan: RectifyPlan | None = None):
        """(cells, confidence), both (rows, cols). Cells are indices into `labeling.CLASSES`."""
        self.net.eval()
        with torch.no_grad():
            logits = self.net(to_tensor(rect).to(self.device))
            prob = F.softmax(logits, dim=1)[0]
        conf, cls = prob.max(dim=0)
        cells = cls.cpu().numpy().astype(np.int8)
        if plan is not None:
            cols, rows = plan.size_tiles
            if cells.shape != (rows, cols):
                raise ValueError(
                    f"the network emitted {cells.shape} cells but the rectification grid is "
                    f"{(rows, cols)}; the stride no longer matches pixels_per_tile"
                )
        return cells, conf.cpu().numpy()

    def save(self, path) -> None:
        torch.save({"state_dict": self.net.state_dict(),
                    "classes": [t.name for t in CLASSES]}, str(path))

    @classmethod
    def from_config(cls, cfg: VisionConfig | None = None, path=None) -> "TerrainClassifier":
        """The shipped weights on the configured device, so a caller needs neither.

        Every other stage in this package has one of these (`ObjectDetector`, `HudReader`,
        `HealthTracker`, `OccupancyMap`), and this was the only one that made its caller carry a
        path -- which meant `brawl_deployment` would have had to hardcode `data/terrain.pt` and
        pick a device the config already names.
        """
        cfg = cfg or VisionConfig()
        return cls.load(path or TERRAIN_WEIGHTS_PATH, device=cfg.classifier_device)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "TerrainClassifier":
        blob = torch.load(str(path), map_location=device, weights_only=False)
        names = [t.name for t in CLASSES]
        if blob.get("classes") != names:
            raise ValueError(
                f"checkpoint was trained on classes {blob.get('classes')}, but this build uses "
                f"{names}. The head's channel order has changed and every prediction would be "
                f"silently permuted."
            )
        net = TerrainNet()
        net.load_state_dict(blob["state_dict"])
        net.to(device)
        return cls(net=net, device=device)


def train(examples: list[Example], epochs: int = 200, lr: float = 3e-3, seed: int = 0,
          device: str = "cpu", augment_data: bool = True,
          cfg: VisionConfig | None = None, log_every: int = 0) -> TerrainClassifier:
    """Fit on whatever labelled cells exist. Deliberately plain: a few hundred cells does not
    justify a schedule, and anything fancier would be tuned against a validation set too small to
    mean much."""
    cfg = cfg or VisionConfig()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = TerrainNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for epoch in range(epochs):
        total = 0.0
        for ex in rng.permutation(len(examples)):
            e = examples[ex]
            rect, target = e.rect, e.target
            if augment_data:
                rect, target = flip_pair(rect, target, rng)
                rect = augment(rect, rng)
            x = to_tensor(rect).to(device)
            y = torch.from_numpy(target).unsqueeze(0).to(device)
            logits = net(x)
            loss = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:3d}  loss {total / max(len(examples), 1):.4f}")
    return TerrainClassifier(net=net, device=device)


def evaluate(model: TerrainClassifier, examples: list[Example]) -> dict:
    """Per-class recall plus the confusion matrix.

    WALL-vs-FENCE is reported on its own because it is the one pair where a mistake changes what
    the agent believes about cover: both block movement, only WALL blocks shots, so a fence read as
    a wall tells the policy it is safe from a shot that is about to hit it.
    """
    n = len(CLASSES)
    confusion = np.zeros((n, n), np.int64)
    for e in examples:
        pred, _ = model.predict(e.rect)
        sel = e.target != IGNORE_INDEX
        for t, p in zip(e.target[sel].ravel(), pred[sel].ravel()):
            confusion[int(t), int(p)] += 1
    support = confusion.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support > 0, confusion.diagonal() / np.maximum(support, 1), np.nan)
    wall, fence = CLASS_INDEX[Tile.WALL], CLASS_INDEX[Tile.FENCE]
    pair = confusion[[wall, fence]][:, [wall, fence]]
    return {
        "confusion": confusion,
        "support": {t.name: int(support[i]) for i, t in enumerate(CLASSES)},
        "recall": {t.name: float(recall[i]) for i, t in enumerate(CLASSES)},
        "overall": float(confusion.diagonal().sum() / max(confusion.sum(), 1)),
        "wall_vs_fence": (float(pair.diagonal().sum() / pair.sum())
                          if pair.sum() else float("nan")),
    }


def flicker_rate(model: TerrainClassifier, rects: list[np.ndarray]) -> float:
    """Share of cells whose predicted class changes between consecutive frames of a static scene.

    Phase I's voting absorbs some flicker, but it should not be happening on every tile every
    frame -- that would mean the classifier is reading noise, and voting would only launder it.
    """
    preds = [model.predict(r)[0] for r in rects]
    if len(preds) < 2:
        return 0.0
    changes = [(a != b).mean() for a, b in zip(preds, preds[1:])]
    return float(np.mean(changes))
