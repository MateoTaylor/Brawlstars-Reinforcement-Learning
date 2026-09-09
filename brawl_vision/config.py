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
TERRAIN_WEIGHTS_PATH = DATA_DIR / "terrain.pt"

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

    # --- object detector (third-party YOLO, brawl_vision/object_detection) ----------------
    # Which fetched model to run, by MODELS key rather than by path -- the weights are
    # downloaded rather than committed, so a path would point at nothing on a fresh clone.
    detector_model: str = "entity_v2"
    # MEASURED on three Solo Showdown fixtures, 101 frames. Entirely a statement about the
    # ENEMY class: the player box sits at median 0.91 and is flat from 0.35 to 0.70, while
    # enemies run median 0.86 with a long occlusion tail (p25 0.67, p05 0.28). 0.5 keeps 95 of
    # 112 enemy boxes, PylaAI's own 0.6 drops 6 more, 0.35 adds 6 back. See configs/vision.yaml.
    detector_conf: float = 0.5
    # Class-wise NMS overlap. PylaAI's value, unmeasured here: brawlers do not stack, so this
    # barely gets exercised.
    detector_iou: float = 0.6
    # Classes dropped after NMS. 'teammate' cannot exist in Solo Showdown, and at conf 0.5 it
    # fires once in 101 frames anyway -- but it is the canary for a corrupted input (fed BGR
    # instead of RGB the same footage yields 44 of them), so emptying this is how you debug.
    detector_ignore: tuple[str, ...] = ("teammate",)
    # auto | cpu | cuda. Unlike classifier_device this is not a straight device string: ONNX
    # Runtime picks by execution PROVIDER, and CUDA is only available at all if the
    # onnxruntime-gpu build is installed. 'auto' takes it when present, 'cuda' insists.
    detector_device: str = "auto"
    # Where inside a box the brawler's FEET are, as a fraction of box height up from the bottom
    # edge. Not 0: a YOLO box here encloses the nameplate, health bar, ammo pips, sprite and any
    # active aura, so the bottom edge is the bottom of the aura -- ~68 px (about a tile) below
    # the feet on standstill f300. Chosen by wall-and-water impossibility rate; the raw bottom
    # edge scores 23.7% impossible against a 26.5% base rate, i.e. barely better than chance.
    # See Detection.anchor for the table and for why 0.20-0.40 is a band rather than a point.
    detector_anchor_frac: float = 0.30

    # --- projectile detector (OURS, object_detection/projectile_detection) ----------------
    # A PATH, unlike detector.model's registry key, and the difference is provenance: that model
    # is fetched from a URL by a name, this one comes out of a training run on this machine and
    # there is nothing to look it up in. Empty means the promoted default,
    # projectile_detection/weights/projectiles.onnx -- see that package's weights.py for why the
    # default is a stable name rather than a runs/ directory.
    projectile_model: str = ""
    # NOT MEASURED, and lower than detector_conf on purpose rather than by accident. Two reasons
    # to expect a different number here. The model is NMS-free, so its 300 output rows have no
    # objectness gate ahead of them and this threshold is the only thing thinning them. And the
    # cost asymmetry is inverted: a projectile is on screen for a handful of frames and there may
    # be six at once, so a miss loses the event entirely while a duplicate box costs nothing --
    # the opposite of a brawler, which the next frame recovers. 0.30 is a starting point to sweep
    # with --projectile-conf, not a finding. Replace this comment with numbers once the model has
    # been scored on held-out footage.
    projectile_conf: float = 0.30
    # auto | cpu | cuda, resolved exactly as detector.device is. No `iou` key: YOLO26 runs no NMS
    # at all, so an overlap threshold here would be a setting nothing reads.
    projectile_device: str = "auto"

    # --- HP readout (object_detection/hp_detection) ---------------------------------------
    # The slice of each detection box searched for a health readout. Height first: the readout
    # (power pip, nameplate, HP number, bar) sits at the top of the box, and across the sampled
    # crops the digit row's top edge never exceeded 0.35 of box height -- 0.55 is that with room
    # for the box "breathing" as auras come and go (measured 264-380 px within one clip).
    hp_crop_height_frac: float = 0.55
    # Horizontal PADDING, as a fraction of box width, because the health bar is drawn to a width
    # set by MAX HP and overhangs the box -- measured up to 160 px of bar against a 135 px box.
    # The cost is pulling a neighbour's readout into the crop during a fight, which is what the
    # centring test in read.py exists to reject.
    hp_crop_pad_frac: float = 0.12
    # MEASURED. The HP number is white; the nameplate directly above it is TEAM-COLOURED (green
    # for you, salmon for an enemy) and the bars are saturated. So the saturation CEILING is what
    # separates the number from the one other run of glyphs in the same font a few pixels away --
    # this is not a generic "bright" threshold and loosening it merges the two.
    hp_white_max_sat: int = 70
    hp_white_min_val: int = 175
    # MEASURED across 195 harvested glyphs: heights ran 15:47 16:103 17:16, with a thin tail to
    # 12 and 24 that is damage popups caught mid-animation rather than HP digits. 13-19 keeps the
    # real population and excludes the popups, which is the single most important filter here --
    # popups are the same font and the same white, drawn several times larger, and they animate
    # straight across the readout. Depends on capture.normalized_aspect: re-measure at a new
    # resolution, because these are SCREEN pixels and the HUD does not scale with the camera.
    hp_glyph_min_height: int = 13
    hp_glyph_max_height: int = 19
    # Confidence-ramp floors -- the value at which a glyph's score / margin contributes 0. Intact
    # reads sit at a median NCC of 0.949 with margins of 0.15-0.20; the two bad reads that beat
    # every structural gate scored 0.418/0.054 and 0.693/0.367. These sit between the two
    # populations. The upper ends of the ramps are in read.py, tied to those same measurements.
    hp_min_glyph_score: float = 0.70
    hp_min_glyph_margin: float = 0.03
    # MEASURED. Hue band of the FILLED health bar per class, OpenCV 8-bit (H is 0-179). Player
    # green medians 56 and enemy red medians 173 over 161 measured bars; the enemy band wraps
    # past 179, which is legitimate and handled. Only ever searched in a window under an
    # already-found digit row -- a hue search over the whole box locks onto the map instead (the
    # day10 skin's bush is hue 93 and its water is hue ~66, a hair from player green).
    hp_bar_hue_player: tuple[int, int] = (46, 66)
    # (166, 5), not (166, 180): OpenCV hue tops out at 179, so 180 is not a colour -- and written
    # that way the band ALSO fails to wrap, quietly matching nothing at the red end it exists for.
    hp_bar_hue_enemy: tuple[int, int] = (166, 5)
    # --- ammo pips (hp_detection/hero_bars). The ROW below the HP bar, for your own brawler only.
    # A taller crop than the HP readout needs: the ammo track sits ~15-25 px below the HP bar's
    # row, which 0.55 of a box clears only barely.
    hp_ammo_crop_height_frac: float = 0.75
    # Pip orange. MEASURED over 353 runs on bluestacks-example-new.mp4: hue 10-11, sat 155-223,
    # val 213-232. The band is wider than that at the dark end because a pip mid-reload is drawn
    # dimmer than a full one, and refusing to see it would quantise ammo to whole shots.
    hp_ammo_hue: tuple[int, int] = (2, 18)
    hp_ammo_sat_min: int = 110
    hp_ammo_val_min: int = 130
    # One full pip's width at the calibrated 2002x1126 viewport -- the modal run length, with 33
    # and 35 as its shoulders. `AmmoReading.pip_px` reports the frame's own longest run against
    # it, so drift shows up rather than silently scaling every reading.
    hp_ammo_pip_px: int = 34
    # --- super charge (hp_detection/hero_bars). One track below the ammo pips, same stack, and
    # unlike ammo it is SELF-NORMALIZING: the empty remainder is painted in its own colour, so the
    # denominator is the track's own extent rather than a calibrated constant. MEASURED on
    # bluestacks-example-new.mp4 over 181 located frames -- 3890 filled px, 6494 ready px, 1720
    # empty px, each sampled from a run identified by structure rather than by these thresholds.
    #
    # Filled: hue 26-29, sat 114-197, val 219-255. Widened at the dark/dull end for the same
    # reason the ammo band is: the leading edge of a growing bar is drawn dimmer.
    hp_super_hue: tuple[int, int] = (20, 35)
    hp_super_sat_min: int = 100
    hp_super_val_min: int = 180
    # Ready: hue 162-166 at saturation EXACTLY 255 in every percentile, val 189-234. This is a
    # third state, not "full yellow" -- see hero_bars.read_super for why a reader without it
    # reports 0.0 charge at the exact moment the super is available.
    hp_super_ready_hue: tuple[int, int] = (155, 172)
    hp_super_ready_sat_min: int = 230
    hp_super_ready_val_min: int = 160
    # Empty: hue 124-133, sat 102-143, val 74-99. Narrow on purpose, and the narrowness is the
    # whole point -- a first pass used sat >= 80, val 60-140 and the mask ate the map behind the
    # widget, returning "tracks" of 110-225 px for a 108 px widget. Dark blue-purple is what this
    # map is made of; only the tight val band separates the widget from it.
    hp_super_empty_hue: tuple[int, int] = (117, 137)
    hp_super_empty_sat: tuple[int, int] = (90, 165)
    hp_super_empty_val: tuple[int, int] = (62, 118)
    # The track's full width at the calibrated 2002x1126 viewport, and how far a frame's own extent
    # may differ. A SANITY GATE, not a denominator: the reading divides by the extent this frame
    # actually found, and this constant only decides whether that extent is credible. MEASURED as
    # the median of 171 successful reads -- p10 113, p50 116, p90 118, sd 3.3. A first pass put it
    # at 108 by measuring only the magenta ready-state run, which stops at the widget's rounded end
    # where the bridged filled+empty track does not.
    hp_super_track_px: int = 116
    hp_super_track_tol_px: int = 22
    # --- temporal (smooth.HealthTracker). All in FRAMES, at whatever rate the caller drives it;
    # at the 4 Hz decision rate confirm_frames 2 is 500 ms of latency on a real change.
    # A changed HP must be seen this many times before it replaces the committed value, unless it
    # arrives above instant_confidence. 2 rejects essentially every transient at one frame's cost.
    hp_confirm_frames: int = 2
    # Frames a track coasts on its last committed HP through unusable reads before giving up.
    # Not 0: a readout briefly covered by its own damage popup belongs to a brawler whose HP is
    # still, to a good approximation, what it was 750 ms ago, and reporting nothing there punches
    # a hole in the observation exactly while the agent is being shot at.
    hp_max_misses: int = 3
    # A single read this confident commits immediately rather than waiting for confirmation --
    # EXCEPT when it is a digit-truncation of the value it would replace, which is the signature
    # of a partly-covered readout and is never trusted on confidence alone. See smooth.py.
    hp_instant_confidence: float = 0.90
    # Per-frame confidence decay while a track is coasting or being contradicted.
    hp_decay: float = 0.75
    # Association gate for carrying a track between frames, as a fraction of box WIDTH. This is
    # not a tracker: a Mortis dash covers ~340 px between decisions at 4 Hz against a ~180 px
    # box, and breaking there is intended.
    hp_match_distance_frac: float = 1.5
    # The confidence a reading must reach for `trusted()`. A POLICY choice, not a measurement:
    # it trades wrong numbers against absent ones, and which is worse is the caller's business.
    hp_min_confidence: float = 0.50

    # --- screen-anchored HUD (hud.py). "Brawlers left: N", the one field that is neither
    # box-relative nor a bar. Note it reuses `hp.white_max_sat` / `hp.white_min_val` rather than
    # getting its own pair: it is the same font in the same white, measured here at saturation
    # 2-5 and value 203-255, and two knobs that must agree are two knobs that can disagree.
    # How many brawlers the mode fields, which is also the largest readable count. Solo Showdown
    # is 10; a read outside 1..10 is a mis-segmentation rather than a surprising match.
    hud_max_brawlers: int = 10
    # Glyph gates for the count, and they are a BACKSTOP -- the structural gates (a colon at the
    # measured x, a bounded gap to the first digit, a shared row) are what reject the white map
    # pixels a screen-space band admits. Before those existed, an ungated pass over one clip had a
    # score p05 of 0.308: map blobs classifying as digits. MEASURED after them, over 4538 accepted
    # reads across ten clips, 40-42 px HUD digits against the 15-17 px HP templates run score p05
    # 0.893-0.930 per clip with a global minimum of 0.758, and margin p05 0.160-0.229 with a
    # minimum of 0.152. These floors sit under that population the way read.py's do under theirs.
    #
    # What the score floor does NOT catch, measured: a solid 19x41 white rectangle classifies as
    # a `1` at 0.753 / 0.076 and would be read. A solid 28x41 one scores 0.626 and is refused. So
    # this is worth having and is not the defence -- raising it to cover the first case would put
    # it above real reads.
    hud_min_glyph_score: float = 0.70
    hud_min_glyph_margin: float = 0.05


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
    ("detector.model", "detector_model", str),
    ("detector.conf", "detector_conf", float),
    ("detector.iou", "detector_iou", float),
    ("detector.ignore", "detector_ignore", lambda v: tuple(str(x) for x in v)),
    ("detector.device", "detector_device", str),
    ("detector.anchor_frac", "detector_anchor_frac", float),
    ("projectile.model", "projectile_model", str),
    ("projectile.conf", "projectile_conf", float),
    ("projectile.device", "projectile_device", str),
    ("hp.crop_height_frac", "hp_crop_height_frac", float),
    ("hp.crop_pad_frac", "hp_crop_pad_frac", float),
    ("hp.white_max_sat", "hp_white_max_sat", int),
    ("hp.white_min_val", "hp_white_min_val", int),
    ("hp.glyph_min_height", "hp_glyph_min_height", int),
    ("hp.glyph_max_height", "hp_glyph_max_height", int),
    ("hp.min_glyph_score", "hp_min_glyph_score", float),
    ("hp.min_glyph_margin", "hp_min_glyph_margin", float),
    ("hp.bar_hue_player", "hp_bar_hue_player", lambda v: tuple(int(x) for x in v)),
    ("hp.bar_hue_enemy", "hp_bar_hue_enemy", lambda v: tuple(int(x) for x in v)),
    ("hp.ammo_crop_height_frac", "hp_ammo_crop_height_frac", float),
    ("hp.ammo_hue", "hp_ammo_hue", lambda v: tuple(int(x) for x in v)),
    ("hp.ammo_sat_min", "hp_ammo_sat_min", int),
    ("hp.ammo_val_min", "hp_ammo_val_min", int),
    ("hp.ammo_pip_px", "hp_ammo_pip_px", int),
    ("hp.super_hue", "hp_super_hue", lambda v: tuple(int(x) for x in v)),
    ("hp.super_sat_min", "hp_super_sat_min", int),
    ("hp.super_val_min", "hp_super_val_min", int),
    ("hp.super_ready_hue", "hp_super_ready_hue", lambda v: tuple(int(x) for x in v)),
    ("hp.super_ready_sat_min", "hp_super_ready_sat_min", int),
    ("hp.super_ready_val_min", "hp_super_ready_val_min", int),
    ("hp.super_empty_hue", "hp_super_empty_hue", lambda v: tuple(int(x) for x in v)),
    ("hp.super_empty_sat", "hp_super_empty_sat", lambda v: tuple(int(x) for x in v)),
    ("hp.super_empty_val", "hp_super_empty_val", lambda v: tuple(int(x) for x in v)),
    ("hp.super_track_px", "hp_super_track_px", int),
    ("hp.super_track_tol_px", "hp_super_track_tol_px", int),
    ("hp.confirm_frames", "hp_confirm_frames", int),
    ("hp.max_misses", "hp_max_misses", int),
    ("hp.instant_confidence", "hp_instant_confidence", float),
    ("hp.decay", "hp_decay", float),
    ("hp.match_distance_frac", "hp_match_distance_frac", float),
    ("hp.min_confidence", "hp_min_confidence", float),
    ("hud.max_brawlers", "hud_max_brawlers", int),
    ("hud.min_glyph_score", "hud_min_glyph_score", float),
    ("hud.min_glyph_margin", "hud_min_glyph_margin", float),
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

    for name in ("detector_conf", "detector_iou"):
        value = getattr(cfg, name)
        if not 0.0 < value <= 1.0:
            dotted = name.replace("_", ".", 1)
            raise ValueError(f"{dotted} is a fraction in (0, 1], got {value}")
    # Not checked against the model's own class list: validate() runs on a config, and reading
    # the names out of an ONNX file to check them would make config validation depend on the
    # weights being fetched. A typo here silently ignores nothing, which is the harmless
    # direction -- and `scripts/vision_detect.py --list-classes` is the thing that spells them.
    if isinstance(cfg.detector_ignore, str):
        raise ValueError(
            f"detector.ignore must be a LIST of class names, got the string "
            f"{cfg.detector_ignore!r} -- which would be read character by character"
        )
    # 0 is legal -- it is the raw box bottom, and the baseline the 0.30 was measured against.
    # 1.0 is the box TOP, above the nameplate, and is never a brawler's feet.
    if not 0.0 <= cfg.detector_anchor_frac < 1.0:
        raise ValueError(
            f"detector.anchor_frac is a fraction of box height in [0, 1), got "
            f"{cfg.detector_anchor_frac}"
        )
    if cfg.detector_device not in ("auto", "cpu", "cuda"):
        raise ValueError(
            f"detector.device must be 'auto', 'cpu' or 'cuda', got {cfg.detector_device!r}"
        )

    if not 0.0 < cfg.projectile_conf <= 1.0:
        raise ValueError(f"projectile.conf is a fraction in (0, 1], got {cfg.projectile_conf}")
    if cfg.projectile_device not in ("auto", "cpu", "cuda"):
        raise ValueError(
            f"projectile.device must be 'auto', 'cpu' or 'cuda', got {cfg.projectile_device!r}"
        )
    # projectile.model is NOT checked for existence here, for the same reason detector.ignore is
    # not checked against the model's class list: validate() runs on a config, and touching the
    # filesystem would make a config valid or invalid depending on whether a training run had
    # finished. projectile_detection/weights.require() is what reports a missing file, at the
    # point something actually tries to load it, and it lists the runs that would satisfy it.

    for name in ("hp_crop_height_frac", "hp_crop_pad_frac", "hp_ammo_crop_height_frac"):
        value = getattr(cfg, name)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name.replace('_', '.', 1)} is a fraction in (0, 1], got {value}")
    # The ammo track is BELOW the HP bar, so its crop must reach further down the box than the
    # digit search does. A value under hp_crop_height_frac cannot contain the thing it is for.
    if cfg.hp_ammo_crop_height_frac < cfg.hp_crop_height_frac:
        raise ValueError(
            f"hp.ammo_crop_height_frac ({cfg.hp_ammo_crop_height_frac}) must be at least "
            f"hp.crop_height_frac ({cfg.hp_crop_height_frac}) -- the ammo track sits below the "
            f"HP bar, so a shorter crop cannot reach it"
        )
    if cfg.hp_ammo_pip_px <= 0:
        raise ValueError(f"hp.ammo_pip_px is a pixel width, got {cfg.hp_ammo_pip_px}")
    if cfg.hp_super_track_px <= 0:
        raise ValueError(f"hp.super_track_px is a pixel width, got {cfg.hp_super_track_px}")
    if not 0 < cfg.hp_super_track_tol_px < cfg.hp_super_track_px:
        raise ValueError(
            f"hp.super_track_tol_px ({cfg.hp_super_track_tol_px}) must be positive and below "
            f"hp.super_track_px ({cfg.hp_super_track_px}) -- a tolerance as wide as the track "
            f"accepts any run at all, which is the gate's only job to refuse"
        )
    # The empty remainder's mask is the one that decides the DENOMINATOR, so a band wide enough to
    # admit the map behind the widget silently inflates the track and deflates every reading. A
    # first pass at these numbers did exactly that; see the comment on hp_super_empty_val.
    for name in ("hp_super_empty_hue", "hp_super_empty_sat", "hp_super_empty_val"):
        lo, hi = getattr(cfg, name)
        if lo >= hi:
            raise ValueError(f"{name.replace('_', '.', 1)} must be (low, high) with low < high, "
                             f"got {(lo, hi)!r}")
    if not 0 <= cfg.hp_white_max_sat <= 255:
        raise ValueError(f"hp.white_max_sat must be in [0, 255], got {cfg.hp_white_max_sat}")
    if not 0 <= cfg.hp_white_min_val <= 255:
        raise ValueError(f"hp.white_min_val must be in [0, 255], got {cfg.hp_white_min_val}")
    if cfg.hp_glyph_min_height >= cfg.hp_glyph_max_height:
        raise ValueError(
            f"hp.glyph_min_height must be below hp.glyph_max_height, got "
            f"{cfg.hp_glyph_min_height} >= {cfg.hp_glyph_max_height}"
        )
    # Popup glyphs -- the damage numbers the game floats over a brawler -- are the same font and
    # the same white as an HP digit, and are told apart ONLY by being taller than locate.POPUP_MIN_H.
    # A glyph band that reaches up into them stops distinguishing the two, which is this chunk's
    # worst failure mode rather than a slightly looser filter.
    from .object_detection.hp_detection.locate import POPUP_MIN_H
    if cfg.hp_glyph_max_height >= POPUP_MIN_H:
        raise ValueError(
            f"hp.glyph_max_height ({cfg.hp_glyph_max_height}) must stay below "
            f"locate.POPUP_MIN_H ({POPUP_MIN_H}) -- above it a floating damage number is "
            f"indistinguishable from an HP digit, and a covered readout reads as a confident "
            f"wrong number rather than as an error"
        )
    for name in ("hp_min_glyph_score", "hp_min_glyph_margin", "hp_instant_confidence",
                 "hp_decay", "hp_min_confidence"):
        value = getattr(cfg, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name.replace('_', '.', 1)} must be in [0, 1], got {value}")
    for name in ("hp_bar_hue_player", "hp_bar_hue_enemy", "hp_ammo_hue",
                 "hp_super_hue", "hp_super_ready_hue"):
        value = getattr(cfg, name)
        if len(value) != 2:
            raise ValueError(f"{name.replace('_', '.', 1)} must be (low, high), got {value!r}")
        # Not checked: low <= high. Enemy red WRAPS past 179 back through 0 and is spelled
        # exactly that way, the same asymmetry zone.hsv_low documents for its own hue.
        for edge in value:
            if not 0 <= edge <= 179:
                raise ValueError(
                    f"{name.replace('_', '.', 1)} is an OpenCV hue in [0, 179] (a HALF-degree "
                    f"wheel, not 0-359), got {edge}"
                )
    if cfg.hp_confirm_frames < 1:
        raise ValueError(
            f"hp.confirm_frames must be >= 1, got {cfg.hp_confirm_frames} -- at 0 every "
            f"single-frame mis-segmentation commits immediately, which is the failure the "
            f"tracker exists to prevent"
        )
    if cfg.hp_max_misses < 0:
        raise ValueError(f"hp.max_misses must be >= 0, got {cfg.hp_max_misses}")
    if cfg.hp_match_distance_frac <= 0:
        raise ValueError(
            f"hp.match_distance_frac must be > 0, got {cfg.hp_match_distance_frac}"
        )

    if cfg.hud_max_brawlers < 1:
        raise ValueError(f"hud.max_brawlers must be >= 1, got {cfg.hud_max_brawlers}")
    if cfg.hud_max_brawlers >= 100:
        raise ValueError(
            f"hud.max_brawlers is {cfg.hud_max_brawlers}, which needs three digits -- hud.py "
            f"reads at most two (MAX_DIGITS), so every count above 99 would be rejected as a "
            f"mis-segmentation rather than read"
        )
    for name in ("hud_min_glyph_score", "hud_min_glyph_margin"):
        value = getattr(cfg, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name.replace('_', '.', 1)} must be in [0, 1], got {value}")
