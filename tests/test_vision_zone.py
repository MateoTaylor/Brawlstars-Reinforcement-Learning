"""Poison-gas detection. See Terrain_Perception_Build_Plan.md Phase G.

The mechanics run on synthetic patches; the threshold itself is checked against real footage from
three visually different maps, because "does this generalize across skins" is the only property
that matters here and a single-map test cannot see it.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.capture import normalize_viewport
from brawl_vision.clips import load_bounds
from brawl_vision.config import DEFAULT_CONFIG_PATH, VisionConfig, load_vision_config
from brawl_vision.terrain.zone import ZoneMask, detect_zone

CLIPS = Path(__file__).resolve().parent / "fixtures" / "vision"


@pytest.fixture(scope="module")
def plan():
    return build_rectify_plan(load_camera_model(), load_hud_mask())


def _frame(plan, colour=(0, 0, 0)):
    img = np.zeros((plan.size_px[1], plan.size_px[0], 3), np.uint8)
    img[:] = colour
    return img


def _bgr_for(h, s, v):
    """A BGR pixel with a known HSV, so tests can address the threshold in its own units."""
    return cv2.cvtColor(np.array([[[h, s, v]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0].tolist()


def _seek(name, index):
    """One normalized frame by SEEK. Reaching frame 4200 of a clip otherwise costs 4200 decodes,
    which would be most of this file's runtime.

    **`index` is approximate.** Seeking lands on a keyframe and decodes forward, so the frame
    returned is NEAR the requested index, not equal to the one `ClipReader` yields at that count --
    measured, they differ visibly. That is fine here: these tests ask "does the band find the gas
    on this map around this point", never "what is at exactly frame N". Do not use this helper for
    anything frame-indexed against a ground-truth annotation.
    """
    path = CLIPS / f"{name}.mp4"
    bounds = load_bounds(path)
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = cap.read()
    finally:
        cap.release()
    assert ok, f"{name}: could not read frame {index}"
    return np.ascontiguousarray(
        normalize_viewport(img, tuple(bounds["content_box"]),
                           VisionConfig().capture_normalized_aspect))


# ---------------------------------------------------------------------------
# mechanics
# ---------------------------------------------------------------------------

def test_a_frame_entirely_inside_the_band_is_all_gas(plan):
    cfg = VisionConfig()
    mid = [(a + b) // 2 for a, b in zip(cfg.zone_hsv_low, cfg.zone_hsv_high)]
    z = detect_zone(_frame(plan, _bgr_for(*mid)), plan)
    assert z.coverage > 0.99
    assert z.cells[z.observed].all()


def test_a_frame_outside_the_band_is_all_clear(plan):
    z = detect_zone(_frame(plan, (10, 10, 10)), plan)
    assert z.coverage == 0.0 and z.gassed_cells == 0


def test_the_mask_never_extends_outside_the_valid_region(plan):
    """`pixels` feeds the overlay and `cells` feeds Phase I; letting either claim the black border
    outside the viewport would put permanent phantom gas around the frame edge."""
    cfg = VisionConfig()
    mid = [(a + b) // 2 for a, b in zip(cfg.zone_hsv_low, cfg.zone_hsv_high)]
    z = detect_zone(_frame(plan, _bgr_for(*mid)), plan)
    assert not z.pixels[~plan.valid].any()


def test_cells_are_measured_against_valid_pixels_not_the_whole_cell(plan):
    """A cell half-hidden by the HUD must be judged on the half that is visible, not diluted to
    below threshold by counting masked pixels as clear."""
    cfg = VisionConfig()
    mid = [(a + b) // 2 for a, b in zip(cfg.zone_hsv_low, cfg.zone_hsv_high)]
    z = detect_zone(_frame(plan, _bgr_for(*mid)), plan)
    partly = z.observed & (
        np.add.reduceat(np.add.reduceat(plan.valid.astype(int), np.arange(
            0, plan.size_px[1], plan.pixels_per_tile), axis=0), np.arange(
            0, plan.size_px[0], plan.pixels_per_tile), axis=1) < plan.pixels_per_tile ** 2)
    assert partly.any(), "the fixture has no partly-masked cells; the test proves nothing"
    assert z.cells[partly].all()


def test_mostly_masked_cells_are_unobserved_rather_than_clear(plan):
    z = detect_zone(_frame(plan, (10, 10, 10)), plan)
    assert not z.observed.all(), "no cell is mostly outside the viewport; check the fixture"
    assert (z.cells | ~z.observed).any()
    # `cells` must never be true where nothing was observed.
    assert not (z.cells & ~z.observed).any()


def test_opening_removes_speckle_but_keeps_a_cloud(plan):
    cfg = VisionConfig()
    mid = _bgr_for(*[(a + b) // 2 for a, b in zip(cfg.zone_hsv_low, cfg.zone_hsv_high)])
    img = _frame(plan, (10, 10, 10))
    cv2.circle(img, (600, 400), 60, mid, -1)          # a cloud
    for x in range(200, 400, 12):                      # speckle
        img[300:302, x:x + 2] = mid
    opened = detect_zone(img, plan, cfg)
    raw = detect_zone(img, plan, VisionConfig(zone_open_px=0))
    assert opened.coverage < raw.coverage
    assert opened.pixels[400, 600], "the opening ate the cloud"
    assert not opened.pixels[300, 250], "the opening left the speckle"


def test_a_wrapping_hue_range_is_a_union_of_two_bands(plan):
    """Hue is a wheel, so a red-spanning range is expressed as low > high. Implementing it as a
    plain inRange would yield an empty mask -- silently, since it is not an error."""
    cfg = VisionConfig(zone_hsv_low=(170, 60, 185), zone_hsv_high=(10, 165, 255), zone_open_px=0)
    for hue in (175, 5):
        z = detect_zone(_frame(plan, _bgr_for(hue, 110, 220)), plan, cfg)
        assert z.coverage > 0.9, f"hue {hue} inside the wrapped band was not detected"
    z = detect_zone(_frame(plan, _bgr_for(90, 110, 220)), plan, cfg)
    assert z.coverage == 0.0, "a hue outside the wrapped band was detected"


def test_at_least_is_monotonic_and_bounded_by_observed(plan):
    cfg = VisionConfig()
    mid = _bgr_for(*[(a + b) // 2 for a, b in zip(cfg.zone_hsv_low, cfg.zone_hsv_high)])
    img = _frame(plan, (10, 10, 10))
    cv2.circle(img, (700, 450), 200, mid, -1)
    z = detect_zone(img, plan)
    assert z.at_least(0.05).sum() >= z.at_least(0.5).sum() >= z.at_least(0.95).sum()
    assert not (z.at_least(0.01) & ~z.observed).any()


def test_a_non_rectified_frame_is_refused(plan):
    with pytest.raises(ValueError, match="rectified"):
        detect_zone(np.zeros((1126, 2002, 3), np.uint8), plan)


# ---------------------------------------------------------------------------
# the shipped threshold, on real footage from three maps
# ---------------------------------------------------------------------------

def test_the_shipped_threshold_is_no_longer_the_placeholder():
    """The config used to ship the full HSV cube on purpose, and a test failed the moment anyone
    replaced it -- as the prompt to record what it was measured from. This is that record."""
    cfg = load_vision_config(DEFAULT_CONFIG_PATH)
    assert cfg.zone_hsv_low == (42, 60, 185) and cfg.zone_hsv_high == (75, 165, 255)
    text = DEFAULT_CONFIG_PATH.read_text()
    assert "PLACEHOLDER" not in text.split("zone:")[1].split("occupancy:")[0]
    assert "showdown_alternate_map" in text, "the measurement's provenance was dropped"


@pytest.mark.vision
@pytest.mark.parametrize("clip,index,least", [
    ("zone_grows_from_east", 1180, 12),
    ("showdown_alternate_map", 4230, 8),      # the map whose BUSHES share the gas hue
    ("showdown_has_gadget", 810, 3),
])
def test_gas_is_found_on_every_map_that_has_it(plan, clip, index, least):
    z = detect_zone(plan.rectify(_seek(clip, index)), plan)
    assert z.gassed_cells >= least, f"{clip} f{index}: only {z.gassed_cells} cells"


@pytest.mark.vision
@pytest.mark.parametrize("index", [0, 400, 900, 1400, 1900])
def test_no_false_gas_on_a_clip_that_never_gasses(plan, index):
    """`showdown_alternate_map2` is the specificity fixture: the player dies before the shrink
    arrives, so ANY gassed cell here is a false positive. It is a red map with orange furniture,
    i.e. nothing the band should touch."""
    z = detect_zone(plan.rectify(_seek("showdown_alternate_map2", index)), plan)
    assert z.gassed_cells == 0, f"f{index}: {z.gassed_cells} phantom cells"
    assert z.coverage < 0.001


@pytest.mark.vision
def test_green_bushes_are_not_gas(plan):
    """The generalization case, and the reason this phase could not ship on one map's footage.
    `showdown_alternate_map`'s bushes are yellow-green at the SAME median hue as the gas; a
    hue-only band flags 11.5% of this frame. The shipped band must leave the bush strip alone
    while the gas is still on the far side of the map."""
    z = detect_zone(plan.rectify(_seek("showdown_alternate_map", 300)), plan)
    assert z.gassed_cells == 0
    assert z.coverage < 0.01, f"foliage leaked into the mask at {z.coverage:.2%}"


@pytest.mark.vision
def test_coverage_grows_across_a_shrink(plan):
    """Phase G's acceptance, made numeric: the zone only ever grows, so a detector that tracks it
    must too. Checked on the trend rather than frame to frame, since clouds animate."""
    counts = [detect_zone(plan.rectify(_seek("zone_grows_from_east", i)), plan).gassed_cells
              for i in (80, 400, 700, 1000, 1180)]
    assert counts[0] < counts[-1], f"no growth detected: {counts}"
    assert np.mean(counts[3:]) > np.mean(counts[:2]), f"trend is not upward: {counts}"


@pytest.mark.vision
def test_zone_detection_is_cheap_enough_to_run_every_frame(plan):
    """Unlike terrain this is never cached, so its per-frame cost is paid at the decision rate on
    every parallel game."""
    import time
    rect = plan.rectify(_seek("zone_grows_from_east", 1180))
    detect_zone(rect, plan)
    t0 = time.perf_counter()
    for _ in range(20):
        detect_zone(rect, plan)
    ms = (time.perf_counter() - t0) / 20 * 1e3
    assert ms < 25.0, f"{ms:.1f} ms/frame leaves too little of the 250 ms budget for four games"
