"""Run a YOLO ONNX detector over a raw frame and get boxes back.

**Adapted from PylaAI's `detect.py`** (github.com/PylaAI/PylaAI), which is the reference
implementation for these weights and gets the awkward parts right: the output layout is
`(1, 4 + n_classes, 8400)` and needs transposing, boxes come out as `cxcywh` in letterboxed
input space, and NMS has to be done per class because the export carries none. What is kept is
that arithmetic. What is not: their global `cv2.setNumThreads`/`torch.set_num_threads` side
effects at construction, their TOML config coupling, and their dict-of-lists return type, which
drops the confidence and makes the per-detection class implicit in the key.

**One correctness fix over the original, and it is not cosmetic.** PylaAI feeds channel 0 of its
capture straight into channel 0 of the tensor. Their capture is `bettercam`, which hands back
RGB, so their pipeline is correct by coincidence of that library's default. Every frame in
`brawl_vision` is BGR (OpenCV's convention, documented on `capture.Frame`), so copying their
indexing would feed red-for-blue to a network trained in RGB.

That is worth measuring rather than asserting, because the failure is not an error and not even
obviously a degradation -- swapped channels produce MORE boxes, not fewer. Over 30 frames of
`showdown_alternate_map.mp4` at conf 0.35:

    fed RGB (this code):  30 player, 37 enemy,  0 teammate   mean conf 0.860, 1.00 players/frame
    fed BGR (their code): 31 player, 24 enemy, 44 teammate   mean conf 0.722, up to 2 per frame

44 teammates in a Solo Showdown clip that has none, a third of the enemies re-labelled, two
simultaneous "player"s, and every surviving box less confident. Count the detections and the
broken pipeline looks busier; that is exactly why the conversion is here at the boundary, and
why `predict` takes BGR like every other function in this package.

**Two output layouts, told apart by the file rather than by the caller.** The PylaAI weights are
YOLOv11 and emit an anchor grid, `(1, 4 + n_classes, 8400)` of cxcywh logits with no NMS applied.
Our own projectile model is YOLO26, which is NMS-FREE end to end and emits `(1, 300, 6)` --
`x0, y0, x1, y1, conf, class`, already decoded, already deduplicated, already sorted. Running the
anchor-grid postprocessing on that second shape is not an error, it is silently wrong: with 6
columns `pred[:, 4:]` reads the confidence and the class INDEX as if they were two class scores,
so `argmax` returns "class 1" for any box whose class id exceeds its confidence and the boxes come
out as cxcywh that were never cxcywh.

Ultralytics stamps `end2end` into the ONNX metadata, so the file says which it is. See
`_read_end2end` for what happens when the metadata is missing.

**Class names come from the file, not from a constant.** Ultralytics stamps `names` into the ONNX
metadata at export. PylaEntityDetectorV1 and V2 have the SAME three classes in DIFFERENT orders
(V1 `enemy, player, teammate`; V2 `enemy, teammate, player`), so a hardcoded list is a silent
label swap the first time somebody points this at the other file -- player boxes drawn as
teammates, which in Solo Showdown is exactly the pair you cannot afford to confuse.
"""
import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import weights as _weights

# YOLO's letterbox pad. Not black: a black border reads as a dark object edge, and 114/128-grey is
# what ultralytics pads with during training, so it is what the model has learned to ignore.
_PAD = 114


@dataclass(frozen=True)
class Detection:
    """One box, in RAW FRAME pixels -- the same coordinate space as `capture.Frame.image`.

    `xyxy` is inclusive-exclusive in the numpy sense and may run slightly outside the frame; the
    model is under no obligation to keep a box for a half-visible brawler on screen, and clamping
    it here would move the sprite's apparent centre. Clamp at draw time if you need to.
    """
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.xyxy
        return (x0 + x1) / 2, (y0 + y1) / 2

    @property
    def ground_point(self) -> tuple[float, float]:
        """Bottom-centre of the box. **This is NOT where the brawler's feet are** -- see `anchor`,
        which is what callers should use. Kept because it is the raw box edge and the thing every
        naive implementation reaches for, including this one before it was measured."""
        x0, _, x1, y1 = self.xyxy
        return (x0 + x1) / 2, y1

    def anchor(self, frac: float = 0.30) -> tuple[float, float]:
        """The point to project: `frac` of the box height UP from its bottom edge, centred in x.

        **The bottom edge is not the feet, and this was measured rather than assumed.** A YOLO box
        here encloses the nameplate, the health bar, the ammo pips, the sprite, and any active
        aura -- so its bottom is the bottom of the *aura*, and its top is above the nameplate.
        On `standstill` f300 the box bottom sits ~68 px (about a tile) below Mortis's actual feet.

        `frac` was chosen against the one accuracy test available without hand labels: a brawler
        cannot stand in a WALL or in WATER, and the occupancy map knows where those are. Rate of
        impossible cells, player only, scored against each clip's fully-accumulated map:

            anchor              0%     20%     30%     40%     50%    100%    base rate
            showdown_..._map   23.7%  11.3%  11.9%  14.9%  23.8%  56.5%     26.5%
            showdown_..._map2  15.6%   6.7%   5.4%   1.1%   0.8%   4.9%     19.1%

        Two things to read off it. **The bottom edge is barely better than chance** -- 23.7% against
        a 26.5% base rate on the first clip, i.e. projecting the raw box bottom says almost nothing
        about which cell a brawler is on. And **the optimum is a band, not a point**: the wall-heavy
        clip prefers 20% and degrades above 40%, the water-heavy one keeps improving to 50%. They
        disagree because the reference is itself imperfect: walls are the terrain classifier's
        known failure, and the top edge of a water block (the shoreline) is frequently mislabelled
        -- so BOTH scoring classes are the map's weak spots, and an "impossible" cell is sometimes
        the map being wrong rather than the anchor. Excluding water entirely does not resolve it;
        wall-only still gives 20% on the first clip and 40-50% on the second.

        0.30 is the middle of that band and near-best on both. **It is not a calibration** -- it is
        fitted against a reference that is very likely less accurate than the detector being
        calibrated. What survives that objection is the effect SIZE: the bottom edge scores at
        chance and anything in 0.20-0.40 roughly halves the impossible rate on both clips.
        Settling it properly needs frames labelled with a brawler's true tile.
        """
        x0, y0, x1, y1 = self.xyxy
        return (x0 + x1) / 2, y1 - frac * (y1 - y0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Indices of `boxes` to keep, highest score first. Plain numpy, per PylaAI's version."""
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x1 - x0) * (y1 - y0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx0 = np.maximum(x0[i], x0[order[1:]])
        yy0 = np.maximum(y0[i], y0[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        inter = np.maximum(0.0, xx1 - xx0) * np.maximum(0.0, yy1 - yy0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return np.asarray(keep, np.int32)


_dlls_preloaded = False


def preload_cuda_dlls() -> None:
    """Put the `nvidia-*` wheels' DLLs on the search path, once, before any session is created.

    **`onnxruntime-gpu` ships no CUDA libraries and does not go looking for them at import.**
    `pip install "onnxruntime-gpu[cuda,cudnn]"` supplies them as `nvidia/*` wheels and
    `onnxruntime.preload_dlls()` is the function that finds them; nothing calls it for you. Its
    own docstring says the call can be skipped when torch is imported first -- but only when
    torch's CUDA MAJOR matches, and here it does not: torch is `+cu128` (CUDA 12) while
    onnxruntime-gpu 1.29 is built against CUDA 13, so `cublasLt64_12.dll` is on the path and
    `cublasLt64_13.dll` is not.

    Without this call, session creation logs `Error loading onnxruntime_providers_cuda.dll which
    depends on cublasLt64_13.dll which is missing`, falls back to the CPU, and **raises nothing**
    -- while `get_available_providers()` goes on listing CUDA, because that describes the BUILD,
    not what can run. MEASURED: this one call moves the entity detector from
    `CPUExecutionProvider` to `CUDAExecutionProvider` with nothing else changed
    (BRAWL_DEPLOYMENT_DESIGN.md 9.12).

    Failure is not fatal on purpose. Every way this can fail -- an older wheel with no such
    function, a CPU-only build, the nvidia wheels absent -- leaves exactly the CPU session that
    would have been created anyway, and `ObjectDetector.provider` is what reports the outcome.

    **cuDNN is the exception, and the two rules below are a pair.** There is exactly one
    `cudnn64_9.dll` per process, and its engine sublibraries must come from the same install:

    1. **Import torch first.** The preload would otherwise put CUDA 13's cuDNN on the search path
       and torch would resolve its own `torch/lib/cudnn_cnn64_9.dll` against it, dying with
       `OSError: [WinError 127] The specified procedure could not be found` at IMPORT, a thousand
       lines from anything to do with detection.
    2. **Then preload everything EXCEPT cuDNN.** With torch loaded, `cudnn64_9.dll` is already
       torch's CUDA 12 copy and cannot be replaced; `cudnn=True` loads CUDA 13's *sublibraries*
       beside it, and that mixed set is worse than either whole one. MEASURED: it fails the
       projectile model outright (`CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE` on
       `/model.0/conv/Conv`) and costs the entity model 6 ms it did not have to spend.

    Everything else -- cuBLAS, cuFFT, cuRAND, the CUDA runtime -- is version-suffixed
    (`cublasLt64_13.dll` vs `_12`), so both majors coexist and the preload is pure gain there.
    Leaving cuDNN to torch means ORT gets a coherent cuDNN 9 built for CUDA 12 while being built
    for CUDA 13, which sounds worse than it is: MEASURED, both detectors run on CUDA and agree
    with their own CPU sessions to 1.2 px and 3e-4 confidence over real frames.

    | preload | entity | projectile | torch |
    |---|---|---|---|
    | none | CPU 26.1 ms | CPU 47.2 ms | fine |
    | `preload_dlls()`, no torch | 7.4 ms | 14.2 ms | **cannot import** |
    | torch, `preload_dlls()` | 13.3 ms | **fails** | fine |
    | torch, `preload_dlls(cudnn=False)` | **7.2 ms** | **11.0 ms** | fine |

    The live loop needs both halves in one process -- the detectors are ONNX, the policy is torch
    -- so the last row is the only one that is actually a configuration.
    """
    global _dlls_preloaded
    if _dlls_preloaded:
        return
    _dlls_preloaded = True
    have_torch = True
    try:
        import torch                                                # noqa: F401, PLC0415
    except ImportError:                                             # pragma: no cover
        have_torch = False
    import onnxruntime as ort
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        # cuDNN comes from torch when torch is here, and from the nvidia wheel when it is not.
        preload(cudnn=not have_torch)
    except TypeError:                                               # pragma: no cover
        try:                                                        # older wheels: no kwargs
            preload()
        except Exception:
            pass
    except Exception:                                               # pragma: no cover
        pass


def providers_for(device: str, key: str = "detector.device") -> list[str] | None:
    """ONNX Runtime providers for a config `device` word, or None to let the runtime choose.

    `auto` returns None -- the caller's `providers=None` path already means "CUDA if the GPU build
    is installed, else CPU", and duplicating that decision here would be a second place for the
    two to drift apart. `cuda` INSISTS, and raises naming `key` so the message points at the
    setting the reader actually has to change; `cpu` pins CPU.

    `key` is a parameter because two config sections spell this same setting -- `detector.device`
    for the third-party entity model and `projectile.device` for ours -- and an error naming the
    wrong one sends you to edit a line that was never the problem.
    """
    if device == "auto":
        return None
    if device == "cuda":
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                f"{key} is 'cuda' but this onnxruntime build has no CUDA provider. Install "
                f"onnxruntime-gpu (uninstall plain onnxruntime first -- the two collide), or set "
                f"{key} to 'auto'."
            )
        return ["CUDAExecutionProvider"]
    return ["CPUExecutionProvider"]


class ObjectDetector:
    """A YOLO ONNX model over raw frames. Construct once, call `predict` per frame.

    Stateless between frames on purpose. There is no tracking here and no temporal smoothing:
    this reports what one frame contains, and anything that wants identity across frames is a
    layer above that does not exist yet.

    `ignore` drops classes by name after NMS rather than before, so a suppressed class still
    competes for boxes -- an enemy the model half-thinks is a teammate should not survive as a
    duplicate enemy box just because teammates are being hidden.
    """

    def __init__(self, path: str | Path | None = None, *, conf: float = 0.5,
                 iou: float = 0.6, ignore=(), providers=None):
        try:
            import onnxruntime as ort
        except ImportError as exc:                                  # pragma: no cover
            raise ImportError(
                "the object detector needs onnxruntime: pip install -e '.[detect]' "
                "(or `pip install onnxruntime-gpu` instead, if you want it off the CPU)"
            ) from exc

        self.path = Path(path) if path is not None else _weights.require()
        self.conf = float(conf)
        self.iou = float(iou)
        self.ignore = frozenset(ignore)

        # ORT_ENABLE_ALL and default thread counts. PylaAI pins intra/inter op threads from its
        # own config; here the detector runs offline over a clip alongside nothing else, so the
        # runtime's own choice is the right one and a pinned number would just be wrong on a
        # different machine.
        preload_cuda_dlls()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if providers is None:
            # CUDA if the GPU build is installed, else CPU. Deliberately not DirectML or Azure,
            # both of which PylaAI accepts: untested here, and an untested provider that silently
            # produces different numbers is worse than a slower one that does not.
            available = ort.get_available_providers()
            providers = (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available
                         else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(self.path), sess_options=so, providers=providers)
        self.provider = self.session.get_providers()[0]
        # A provider list is a REQUEST. ORT appends a CPU fallback, takes it silently when the
        # provider's DLLs will not load, and returns a perfectly working detector at 26 ms a
        # frame. Someone who wrote `device: cuda` in the config asked a question this answers.
        if providers == ["CUDAExecutionProvider"] and self.provider != "CUDAExecutionProvider":
            raise RuntimeError(
                f"asked for CUDA and got {self.provider}: onnxruntime fell back silently. The "
                f"usual cause is the CUDA libraries not being on the DLL path -- install them "
                f"with `pip install \"onnxruntime-gpu[cuda,cudnn]\"` (see preload_cuda_dlls). "
                f"Set the device to 'auto' to accept the CPU instead."
            )

        spec = self.session.get_inputs()[0]
        self.input_name = spec.name
        # (1, 3, H, W), static for these exports. A dynamic axis comes back as a string, which
        # would make `int()` raise -- fall back to YOLO's 640 rather than crashing on a model
        # this class could otherwise run.
        self.input_size = tuple(int(d) if isinstance(d, int) else 640 for d in spec.shape[2:4])
        self.names = self._read_names()
        self.end2end = self._read_end2end()

    @classmethod
    def from_config(cls, cfg=None, **overrides) -> "ObjectDetector":
        """Build from a `VisionConfig`, so every script gets the same thresholds by default.

        Resolves `detector.device` into ONNX Runtime providers here rather than in `__init__`,
        because 'auto' vs 'cuda' is a *config* question -- one means "prefer it", the other means
        "fail if it is missing" -- and `__init__`'s `providers` stays the literal escape hatch.
        """
        from ..config import VisionConfig
        cfg = cfg or VisionConfig()
        device = overrides.pop("device", cfg.detector_device)
        providers = overrides.pop("providers", None)
        if providers is None:
            providers = providers_for(device, "detector.device")
        kwargs = dict(path=_weights.weights_path(cfg.detector_model), conf=cfg.detector_conf,
                      iou=cfg.detector_iou, ignore=cfg.detector_ignore, providers=providers)
        kwargs.update(overrides)
        if not Path(kwargs["path"]).exists():
            _weights.require(cfg.detector_model)   # raises with the command that fixes it
        return cls(**kwargs)

    def _read_names(self) -> dict[int, str]:
        """Ultralytics writes `names` as the repr of a `{index: name}` dict. Missing metadata is
        not fatal -- boxes with numeric labels still tell you whether the model fires at all."""
        raw = self.session.get_modelmeta().custom_metadata_map.get("names")
        if not raw:
            return {}
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
        return {int(k): str(v) for k, v in parsed.items()} if isinstance(parsed, dict) else {}

    def _read_end2end(self) -> bool:
        """Whether this export is NMS-free (YOLO26 and friends) rather than an anchor grid.

        The metadata is the answer: Ultralytics writes `end2end: 'True'` for these exports, next
        to the `names` `_read_names` already reads, and it is written at export time by the code
        that chose the head.

        The shape fallback is for a file that lost its metadata, and it is a heuristic, not a
        second source of truth. It keys on the LAST axis being 6, because that is the one thing
        the two layouts do not share: an anchor grid's trailing axis is the anchor count (8400 at
        640, 21504 at 1024) and its 4+nc axis is the other one. A 2-class v11 export transposed to
        `(1, 8400, 6)` would also end in 6, so the row count is bounded too -- an NMS-free head
        emits its top-k (300), never thousands. If both checks are wrong the boxes will be
        obviously garbage rather than subtly off, which is the failure worth having.
        """
        raw = self.session.get_modelmeta().custom_metadata_map.get("end2end")
        if raw is not None:
            return str(raw).strip().lower() == "true"
        shape = self.session.get_outputs()[0].shape
        return (len(shape) == 3 and shape[-1] == 6
                and isinstance(shape[1], int) and shape[1] <= 1000)

    @property
    def classes(self) -> list[str]:
        return [self.names[i] for i in sorted(self.names)]

    def _letterbox(self, bgr: np.ndarray):
        """BGR frame -> (1, 3, H, W) float32 RGB in [0, 1], plus the scale that undoes it.

        Aspect-preserving resize into the TOP-LEFT of a grey canvas, which is what PylaAI does and
        is the reason the inverse is a single scalar rather than a scale and two offsets. Centring
        it (ultralytics' own default) would detect the same things; this is the cheaper bookkeeping,
        and the padding is uniform grey either way.
        """
        h, w = bgr.shape[:2]
        net_h, net_w = self.input_size
        scale = min(net_h / h, net_w / w)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        # INTER_AREA, not PylaAI's INTER_LINEAR: this is a 3.1x DOWNSCALE of a 2002x1126 frame,
        # where linear point-samples and aliases thin features (a health bar, a sprite outline)
        # that area-averaging keeps. Their input is a phone-sized capture nearer to 1:1.
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        canvas = np.full((net_h, net_w, 3), _PAD, np.uint8)
        canvas[:new_h, :new_w] = rgb
        tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor), scale

    def predict(self, image: np.ndarray, conf: float | None = None) -> list[Detection]:
        """Detections in `image`, a (H, W, 3) uint8 BGR frame, sorted most confident first."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"expected an (H, W, 3) uint8 BGR frame, got "
                             f"{image.shape} {image.dtype}")
        threshold = self.conf if conf is None else float(conf)
        tensor, scale = self._letterbox(image)
        raw = self.session.run(None, {self.input_name: tensor})[0]

        # (1, 4 + n_classes, 8400) -> (8400, 4 + n_classes). The transpose is conditional on which
        # axis is shorter rather than unconditional, because the same postprocessing has to serve
        # an export that already emits (1, 8400, C) -- and with 3 classes the two shapes are 7 and
        # 8400, never ambiguous.
        pred = np.asarray(raw)
        if pred.ndim == 3:
            pred = pred[0]
        if self.end2end:
            return self._decode_end2end(pred, scale, threshold)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        if pred.shape[1] <= 4:
            return []

        scores = pred[:, 4:]
        class_ids = scores.argmax(1)
        confidences = scores[np.arange(len(scores)), class_ids]
        mask = confidences >= threshold
        if not mask.any():
            return []
        cxcywh = pred[mask, :4]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        half = cxcywh[:, 2:] / 2
        boxes = np.concatenate([cxcywh[:, :2] - half, cxcywh[:, :2] + half], axis=1)

        out: list[Detection] = []
        for cls in np.unique(class_ids):
            at = class_ids == cls
            cls_boxes, cls_scores = boxes[at], confidences[at]
            keep = _nms(cls_boxes, cls_scores, self.iou)
            label = self.names.get(int(cls), str(int(cls)))
            if label in self.ignore or int(cls) in self.ignore:
                continue
            for box, score in zip(cls_boxes[keep] / scale, cls_scores[keep]):
                out.append(Detection(label, float(score), tuple(float(v) for v in box)))
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out

    def _decode_end2end(self, pred: np.ndarray, scale: float,
                        threshold: float) -> list[Detection]:
        """`(300, 6)` of `x0, y0, x1, y1, conf, class` in letterboxed input pixels -> Detections.

        Almost everything the anchor-grid path does by hand is already done here. No transpose:
        the top-k axis is first and is 300, not 8400. No cxcywh conversion: the head emits corners.
        **No NMS at all** -- YOLO26 trains its head to emit one box per object, and running NMS
        over the result would only merge genuinely distinct objects that happen to overlap, which
        for projectiles (a volley of three shots along one line) is exactly the wrong outcome.
        `self.iou` is therefore unused on this path, and that is the honest state of it rather
        than a value quietly having no effect somewhere less visible.

        The 300 rows are fixed-length and padded: a frame with two projectiles still returns 300,
        the other 298 at ~0.003 confidence. The threshold is what empties them, so `conf` is doing
        more work here than it does for an anchor grid, where a per-anchor objectness has already
        thinned the field.
        """
        rows = pred[pred[:, 4] >= threshold]
        if not len(rows):
            return []
        out: list[Detection] = []
        for row in rows:
            label = self.names.get(int(row[5]), str(int(row[5])))
            if label in self.ignore or int(row[5]) in self.ignore:
                continue
            out.append(Detection(label, float(row[4]),
                                 tuple(float(v) for v in row[:4] / scale)))
        # Already descending -- sorted anyway, because `predict` PROMISES it and a future export
        # that stops sorting would break that promise silently rather than loudly.
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out
