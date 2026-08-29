"""VisionConfig: the tunable thresholds of the vision pipeline, loaded from configs/vision.yaml.
See Terrain_Perception_Build_Plan.md Phase 0.

**What belongs here vs. in `data/`.** This file holds SETTINGS -- numbers you tune by hand,
where a different value is a different judgment call (how many votes lock a cell, how weak a
phase-correlation response is too weak). MEASUREMENTS live in `brawl_vision/data/*.json`: the
homography matrix and the HUD rectangles are outputs of a calibration script, regenerated rather
than edited, and meaningless to hand-tweak. Mixing the two in one file would invite someone to
"tune" a homography.

**Most defaults below are PLACEHOLDERS, and say so.** A threshold that has not yet been measured
against real footage is a guess; shipping it silently as a default would make it look like a
finding. Each one names the phase that replaces it. `configs/randomization.yaml` sets the same
precedent -- it ships fully commented rather than pretending to a tuning nobody did.

Mirrors `brawl_sim/config.py`'s shape on purpose: a frozen dataclass of static Python values, a
`(dotted YAML path, field name, coercion)` table, a loader where an absent key keeps the
dataclass default, and a separate `validate`. That file is the reference implementation; when
in doubt about a convention here, go read it.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "vision.yaml"

# Measured calibration artifacts, resolved by convention relative to this package rather than
# through config -- same treatment `maps/loader.CSV_DIR` gives the map CSVs. A config key for
# these would only ever be used to point at the wrong ones.
DATA_DIR = Path(__file__).resolve().parent / "data"
HOMOGRAPHY_PATH = DATA_DIR / "homography.json"
HUD_MASK_PATH = DATA_DIR / "hud_mask.json"

_MISSING = object()
_ABSENT = object()


@dataclass(frozen=True)
class VisionConfig:
    # --- capture (Phase A) ---------------------------------------------------------------
    capture_monitor: int = 1          # mss monitor index; 0 is the virtual "all screens"
    capture_target_fps: float = 20.0  # matches the sim's 20 Hz tick; see Phase 1's budget note
    # Brightness above which a pixel counts as rendered content rather than a letterbox bar.
    # 25, not 0 and not 12: compression does not preserve pure black, and the ramp at a hard
    # bar/content boundary measured 11-16 on real footage -- so 12 admits ringing as 'content'
    # and made the detected left edge wander over 194..216 frame to frame. At 25 the same clip's
    # right edge is rock steady at 2217 and the left clusters at 206.
    capture_black_level: int = 25
    # MEASURED, not a tuning knob -- the exception to this file's settings-vs-measurements split,
    # kept here because it is a single float that must survive a resolution change and would be a
    # silly JSON of its own. Solo Showdown constricts the visible WIDTH relative to other modes,
    # at the same zoom, to keep matches fair: 2002x1126 (exactly 16:9) against the training
    # cave's full-screen 2436x1126, both at ~77 px per tile on screen. Every frame is
    # symmetrically trimmed to this aspect so one homography serves every mode.
    # See Terrain_Perception_Build_Plan.md Section 3.
    capture_normalized_aspect: float = 16 / 9

    # --- odometry (Phase F) --------------------------------------------------------------
    # MEASURED (Phase F). A PER-WINDOW floor: a window scoring below it is dropped, and a frame
    # left with fewer than `min_windows` is called a cut. Across all seven fixtures the 1st
    # percentile of good frames is 0.20-0.95, while the one real camera cut in the footage
    # (training_gadget f508) leaves a single window at 0.132 -- so 0.10 separates them with
    # room to spare. Note this floor alone is NOT sufficient: a window locked onto a moving
    # sprite scores 0.92-0.98. Window agreement is what catches that; see min_agreement_ratio.
    odometry_min_response: float = 0.10
    # MEASURED (Phase F), and it assumes the configured 20 Hz capture rate. The fastest thing
    # in the game is not a walk: Mortis's charged dash covers 5.34 tiles in 0.30 s (17.8
    # tiles/s, configs/brawlers.yaml), which is 0.89 tiles per frame at 20 Hz -- against a top
    # walking speed of 2.73 tiles/s, or 0.137. 2.0 leaves 2.2x headroom over the dash and still
    # catches the one real cut in the fixtures, a 3.3-tile single-frame jump. Raise it if the
    # capture rate ever drops, since the bound is per FRAME, not per second.
    odometry_max_shift_tiles: float = 2.0
    # Correlation window geometry. 192/192 puts 11 windows in the rectified patch at 4.5 ms/frame;
    # measured against 128/128 (33 windows) and 256/256 (4) every geometry agrees on a real camera
    # leg to within 0.2 tiles, so this is a cost choice, not an accuracy one. Below ~8 windows the
    # median stops being robust to a couple of them locking onto a moving sprite.
    odometry_window_px: int = 192
    odometry_stride_px: int = 192
    odometry_min_windows: int = 4
    # How far a per-window estimate may sit from the median and still COUNT AS AGREEING, in tiles.
    # This is the tolerance `min_agreement_ratio` is measured against, not a threshold of its own.
    # Measured on real footage: 0.002 tiles of median disagreement in steady tracking, three orders
    # below this. Correlation RESPONSE cannot do this job -- a window reporting 2.8 tiles of pure
    # fiction still scored 0.92-0.98 -- so window AGREEMENT is what the gate is built on.
    odometry_max_disagreement_tiles: float = 0.25
    # Share of windows that must land within max_disagreement_tiles of the median for a frame to
    # be TRUSTED (status 'ok'). Not a cut threshold: below it the frame is still integrated as
    # 'uncertain', because measured across every fixture the ratio is 1.0 at the median and
    # >=0.90 at p01, and the one-or-two-per-thousand frames that dip to ~0.64 are large effects
    # covering part of the patch, not discontinuities. Treating those as cuts would reset the
    # world frame mid-match, which costs Phase I far more than a noisy step. 0.75 sits between
    # the p01 of good frames and the 0.0 a real cut produces.
    odometry_min_agreement_ratio: float = 0.75

    # --- zone / gas (Phase G) ------------------------------------------------------------
    # PLACEHOLDER, and the most obviously fake numbers in this file -- nobody has looked at a
    # gas frame yet. HSV, not RGB, for lighting robustness. OpenCV's H channel is 0-179, S and
    # V are 0-255. Set in Phase G from real frames spanning a shrink.
    zone_hsv_low: tuple[int, int, int] = (42, 60, 185)
    zone_hsv_high: tuple[int, int, int] = (75, 165, 255)
    # Opening kernel, in rectified pixels. Gas clouds are tens of pixels across; the false
    # positives that survive the colour band -- damage numbers, pickup glints, a nameplate -- are
    # not. Measured to cost ~0.4 points of true coverage while removing rather more than that.
    zone_open_px: int = 9
    # A cell needs this share of its area to be real world before it is judged at all. A cell
    # mostly outside the viewport or under the HUD is UNKNOWN, not clear, and collapsing those
    # together is how a stage quietly starts trusting the frame edge.
    zone_min_cell_pixels: float = 0.5
    # Fraction of a cell's pixels that must fall inside the gas threshold for the CELL to count
    # as gassed. Above 0.5 so a cell straddling the boundary resolves to whichever side owns
    # most of it, rather than flickering.
    zone_min_cell_fraction: float = 0.6

    # --- occupancy (Phase I) -------------------------------------------------------------
    # The accumulated per-match grid. Sized generously rather than to a real map: this system
    # has NO pre-built reference to align to, so the origin is wherever tracking started and
    # the camera can walk off in any direction from there. 128 gives ~2x the sim's 60x60 map in
    # every direction before the grid needs to grow.
    occupancy_grid_h: int = 128
    occupancy_grid_w: int = 128
    # PLACEHOLDER. Minimum observations before a cell is eligible to lock, and the share of
    # those votes the winning class must hold. Both trade map-fill speed against the risk of
    # locking a misclassification permanently -- and locking is forever, so err high. Set in
    # Phase I against the hand-verified ground-truth grid.
    occupancy_min_votes: int = 5
    occupancy_lock_ratio: float = 0.8

    # --- classifier (Phase H) ------------------------------------------------------------
    # CPU by default, and this is a real choice rather than a placeholder: the GPU will be busy
    # running the policy at deploy time, and the Phase H model is small enough that CPU
    # inference is very likely to fit inside the 250 ms decision budget. Measure before moving
    # it -- see the plan's Section 1.
    classifier_device: str = "cpu"


# (dotted YAML path, VisionConfig field name, type coercion)
_VISION_CONFIG_FIELDS = (
    ("capture.monitor", "capture_monitor", int),
    ("capture.target_fps", "capture_target_fps", float),
    ("capture.black_level", "capture_black_level", int),
    ("capture.normalized_aspect", "capture_normalized_aspect", float),
    ("odometry.min_response", "odometry_min_response", float),
    ("odometry.max_shift_tiles", "odometry_max_shift_tiles", float),
    ("odometry.window_px", "odometry_window_px", int),
    ("odometry.stride_px", "odometry_stride_px", int),
    ("odometry.min_windows", "odometry_min_windows", int),
    ("odometry.max_disagreement_tiles", "odometry_max_disagreement_tiles", float),
    ("odometry.min_agreement_ratio", "odometry_min_agreement_ratio", float),
    ("zone.hsv_low", "zone_hsv_low", lambda v: tuple(int(x) for x in v)),
    ("zone.open_px", "zone_open_px", int),
    ("zone.min_cell_pixels", "zone_min_cell_pixels", float),
    ("zone.hsv_high", "zone_hsv_high", lambda v: tuple(int(x) for x in v)),
    ("zone.min_cell_fraction", "zone_min_cell_fraction", float),
    ("occupancy.grid_h", "occupancy_grid_h", int),
    ("occupancy.grid_w", "occupancy_grid_w", int),
    ("occupancy.min_votes", "occupancy_min_votes", int),
    ("occupancy.lock_ratio", "occupancy_lock_ratio", float),
    ("classifier.device", "classifier_device", str),
)


def _dget(d: dict, dotted: str, default=_MISSING):
    """Duplicated from `brawl_sim/config.py` rather than imported, and deliberately so. It is
    ~10 lines, and the alternative is reaching across a package boundary for a PRIVATE name --
    which would make `brawl_vision` depend on the simulator's config machinery (and, through
    it, on torch and `core.projectiles`) to read a YAML file. The intended coupling between
    these two packages is exactly one thing: `constants.Tile`, the shared vocabulary. Same
    reasoning `core/observation._in_bush` gives for duplicating `perception.in_bush`."""
    node = d
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(dotted)
            return default
        node = node[part]
    return node


def load_vision_config(path=DEFAULT_CONFIG_PATH, overrides: dict | None = None) -> VisionConfig:
    """A field absent from the YAML keeps its VisionConfig default rather than raising -- the
    dataclass defaults exist to be the fallback. (`brawl_sim.config.load_config` documents what
    happens when that is gotten wrong: it shipped with an unreachable fallback branch, so every
    config file was silently required to spell out all ~40 fields, and adding one broke every
    previously-valid YAML.)"""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)

    kwargs = {}
    for dotted, field_name, coerce in _VISION_CONFIG_FIELDS:
        value = _dget(raw, dotted, default=_ABSENT)
        if value is _ABSENT:
            continue  # keep the VisionConfig default
        kwargs[field_name] = coerce(value)
    return VisionConfig(**kwargs)


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def validate(cfg: VisionConfig) -> None:
    """One-time sanity check on the tunables. Catches the transcription errors that would
    otherwise surface as a silently empty gas mask or a grid that never locks anything."""
    if not 0 <= cfg.capture_black_level <= 255:
        raise ValueError(f"capture.black_level must be in [0, 255], got {cfg.capture_black_level}")
    if cfg.capture_normalized_aspect <= 0:
        raise ValueError(
            f"capture.normalized_aspect must be > 0, got {cfg.capture_normalized_aspect}"
        )
    if not 0.0 <= cfg.odometry_min_response <= 1.0:
        raise ValueError(
            f"odometry.min_response is a correlation response in [0, 1], got "
            f"{cfg.odometry_min_response}"
        )
    for name in ("odometry_window_px", "odometry_stride_px"):
        if getattr(cfg, name) < 32:
            dotted = name.replace("_", ".", 1)
            raise ValueError(f"{dotted} must be >= 32 px, got {getattr(cfg, name)}")
    if cfg.odometry_min_windows < 3:
        raise ValueError(
            "odometry.min_windows must be >= 3: a median over two samples is their mean, which "
            f"is exactly what a moving sprite defeats. Got {cfg.odometry_min_windows}"
        )
    if cfg.odometry_max_disagreement_tiles <= 0:
        raise ValueError(
            "odometry.max_disagreement_tiles must be > 0, got "
            f"{cfg.odometry_max_disagreement_tiles}"
        )
    if not 0.0 < cfg.odometry_min_agreement_ratio <= 1.0:
        raise ValueError(
            "odometry.min_agreement_ratio is a fraction in (0, 1], got "
            f"{cfg.odometry_min_agreement_ratio}"
        )
    if cfg.odometry_max_shift_tiles <= 0:
        raise ValueError(
            f"odometry.max_shift_tiles must be > 0, got {cfg.odometry_max_shift_tiles}"
        )

    for name, value in (("hsv_low", cfg.zone_hsv_low), ("hsv_high", cfg.zone_hsv_high)):
        if len(value) != 3:
            raise ValueError(f"zone.{name} must be 3 channels (H, S, V), got {value!r}")
    # OpenCV's 8-bit HSV: H is 0-179 (a half-degree hue wheel, NOT 0-359), S and V are 0-255.
    # Writing a 0-359 hue here is the single easiest mistake to make in Phase G and produces an
    # all-empty mask rather than an error, so it is checked rather than commented.
    for channel, (name, hi) in enumerate((("H", 179), ("S", 255), ("V", 255))):
        lo_v, hi_v = cfg.zone_hsv_low[channel], cfg.zone_hsv_high[channel]
        if not 0 <= lo_v <= hi:
            raise ValueError(f"zone.hsv_low[{name}] must be in [0, {hi}], got {lo_v}")
        if not 0 <= hi_v <= hi:
            raise ValueError(f"zone.hsv_high[{name}] must be in [0, {hi}], got {hi_v}")
    # Deliberately NOT checked: hsv_low[H] <= hsv_high[H]. Hue is a wheel, so a range that
    # wraps past 179 back through 0 (red, most obviously) is legitimate and is expressed
    # exactly this way. S and V do not wrap, so they are checked.
    for channel, name in ((1, "S"), (2, "V")):
        if cfg.zone_hsv_low[channel] > cfg.zone_hsv_high[channel]:
            raise ValueError(
                f"zone.hsv_low[{name}] > zone.hsv_high[{name}] "
                f"({cfg.zone_hsv_low[channel]} > {cfg.zone_hsv_high[channel]}); "
                f"S and V do not wrap, only H does"
            )

    if cfg.zone_open_px < 0:
        raise ValueError(f"zone.open_px must be >= 0, got {cfg.zone_open_px}")
    if not 0.0 < cfg.zone_min_cell_pixels <= 1.0:
        raise ValueError(
            f"zone.min_cell_pixels is a fraction in (0, 1], got {cfg.zone_min_cell_pixels}"
        )
    if not 0.0 < cfg.zone_min_cell_fraction <= 1.0:
        raise ValueError(
            f"zone.min_cell_fraction must be in (0, 1], got {cfg.zone_min_cell_fraction}"
        )
    if cfg.occupancy_grid_h < 1 or cfg.occupancy_grid_w < 1:
        raise ValueError(
            f"occupancy grid must be at least 1x1, got "
            f"{cfg.occupancy_grid_h}x{cfg.occupancy_grid_w}"
        )
    if cfg.occupancy_min_votes < 1:
        raise ValueError(f"occupancy.min_votes must be >= 1, got {cfg.occupancy_min_votes}")
    # A ratio at or below 1/5 is unanimous-free: with five terrain classes, an even split
    # already clears it and the first cell to reach min_votes locks whatever it happened to see.
    if not 0.2 < cfg.occupancy_lock_ratio <= 1.0:
        raise ValueError(
            f"occupancy.lock_ratio must be in (0.2, 1.0] -- at or below 1/5 (five terrain "
            f"classes) an even vote split would lock a cell. Got {cfg.occupancy_lock_ratio}"
        )
    if cfg.classifier_device not in ("cpu", "cuda"):
        raise ValueError(f"classifier.device must be 'cpu' or 'cuda', got {cfg.classifier_device!r}")
