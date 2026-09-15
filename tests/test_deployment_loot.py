"""`brawl_deployment.perception.loot`: crates and dropped cubes, for the grid's `box` and `pickup`.

Runs against the REAL rectify plan (the shipped camera model and HUD mask), because two of the
rules under test are about where on screen something is: the edge cut, and "is this spot in view,
so should missing it count". Each box is built backwards from a world position through that same
plan, so a test says where the crate IS and checks where the map puts it.

The anchor fractions are pinned against the footage numbers, not against themselves. Building the
test boxes with `ANCHOR_FRAC` means the map-logic tests cannot catch a wrong fraction, and are
not meant to; `test_the_anchors_are_the_measured_ones` and
`test_a_cube_is_placed_below_its_box_where_its_shadow_is` are the ones that can.
"""
import dataclasses
import math
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from brawl_deployment.perception.loot import (ANCHOR_FRAC, CRATE_ANCHOR_FRAC, CUBE_ANCHOR_FRAC,
                                              DUPLICATE_TILES, EDGE_MARGIN_PX, GATE_TILES,
                                              GONE_S, MIN_HITS, LootMap, box_occlusion,
                                              crate_occlusion, require_loot_classes)
from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.object_detection.detector import Detection
from brawl_vision.object_detection.project import (PROJECTILE_ANCHOR_FRAC, projectile_model_anchor,
                                                   to_tiles)
from brawl_vision.object_detection.projectile_detection.classes import (CUBE_BOX, CUBE_DROPPED,
                                                                        PROJECTILE)

PLAN = build_rectify_plan(load_camera_model(), load_hud_mask())
W, H = PLAN.viewport
DT = 1 / 12                          # the shipped perception rate

# Median box heights (px) and width/height ratios on the 27-clip replay.
CRATE_H, CUBE_H = 167.0, 65.0
ASPECT = {CUBE_BOX: 0.77, CUBE_DROPPED: 1.03}


def _screen_to_tile(px, py):
    """Camera-relative tile under a viewport pixel, through the plan's ground homography."""
    rect = cv2.perspectiveTransform(np.array([[[px, py]]], np.float64), PLAN.M)[0]
    return tuple(float(v) for v in PLAN.rect_to_tile(rect)[0])


def _cell_centre(tile):
    return (math.floor(tile[0]) + 0.5, math.floor(tile[1]) + 0.5)


# A crate spot a little above the screen centre, snapped to a cell centre, the way the sim keeps
# `box_pos`. The camera starts at world (0, 0), so this is also its world position.
SPOT = _cell_centre(_screen_to_tile(1000, 420))
CELL = (math.floor(SPOT[0]), math.floor(SPOT[1]))


def _box(label, world, cam=(0.0, 0.0), *, h=None, conf=0.9):
    """A box whose anchor projects to `world` with the camera at `cam`."""
    h = h or (CRATE_H if label == CUBE_BOX else CUBE_H)
    w = ASPECT.get(label, 1.0) * h
    rect = PLAN.tile_to_rect([[world[0] - cam[0], world[1] - cam[1]]])
    ax, ay = cv2.perspectiveTransform(rect.reshape(1, 1, 2), PLAN.M_inv)[0, 0]
    y1 = ay + ANCHOR_FRAC.get(label, 0.5) * h
    return Detection(label, conf, (ax - w / 2, y1 - h, ax + w / 2, y1))


@dataclass
class _Odo:
    position_tiles: tuple = (0.0, 0.0)
    segment: int = 0
    status: str = "ok"


class _Run:
    """A LootMap on a 12 Hz clock."""

    def __init__(self, **kw):
        self.loot = LootMap(**kw)
        self.odo = _Odo()
        self.t = 0.0

    def tick(self, dets=(), n=1):
        res = None
        for _ in range(n):
            res = self.loot.update(list(dets), PLAN, self.odo, self.t)
            self.t += DT
        return res


# -- the measured constants ----------------------------------------------------------------------

def test_the_anchors_are_the_measured_ones():
    """Crate: 0.30 lands on the tile centre to 0.03 tiles on the calibration clip. Cube: its shadow
    is 5-17 px below a median 65 px box, so the fraction is negative and in that range."""
    assert CRATE_ANCHOR_FRAC == pytest.approx(0.30)
    assert -17 / 65 <= CUBE_ANCHOR_FRAC <= -5 / 65


def test_the_render_projects_crates_and_cubes_with_the_loot_maps_anchors():
    """`scripts/vision_evaluate.py --projectiles-on-map` draws through `projectile_model_anchor`.
    If it and the loot map ever disagreed, a demo would put a crate somewhere the bot does not."""
    assert projectile_model_anchor(CUBE_BOX) == ANCHOR_FRAC[CUBE_BOX] == CRATE_ANCHOR_FRAC
    assert projectile_model_anchor(CUBE_DROPPED) == ANCHOR_FRAC[CUBE_DROPPED] == CUBE_ANCHOR_FRAC
    # A projectile has no ground point, so it keeps the centre, and so does a label nobody knows.
    assert projectile_model_anchor(PROJECTILE) == PROJECTILE_ANCHOR_FRAC == 0.5
    assert projectile_model_anchor("enemy") == PROJECTILE_ANCHOR_FRAC


def test_the_merge_radius_sits_between_the_measured_duplicates_and_distinct_pairs():
    """Widest duplicate pair 0.205 tiles (cubes), closest distinct pair 0.39 (cubes in a pile)."""
    assert 0.205 < DUPLICATE_TILES < 0.39


def test_the_gate_holds_a_crates_own_scatter_and_not_its_neighbour():
    assert DUPLICATE_TILES < GATE_TILES < 0.663       # adjacent crates, closest seen


def test_the_edge_margin_drops_the_whole_biased_band_at_the_bottom():
    """Boxes ending in rows 1100-1123 of 1126 read -0.23 tiles median; above 1100, within 0.1."""
    assert H == 1126
    assert H - EDGE_MARGIN_PX < 1100


def test_a_cube_is_forgotten_sooner_than_a_crate():
    """Cube life is a median 0.6 s; a collected cube left on the map is a walk to nothing."""
    assert GONE_S[CUBE_DROPPED] < GONE_S[CUBE_BOX]


# -- where things land --------------------------------------------------------------------------

def test_a_crate_reaches_the_map_only_after_min_hits_sightings():
    run = _Run()
    crate = _box(CUBE_BOX, SPOT)
    run.tick([crate], MIN_HITS - 1)
    assert run.loot.crates() == []
    run.tick([crate])
    assert run.loot.crates() == [SPOT]
    assert run.loot.cubes() == []


def test_a_cube_is_placed_below_its_box_where_its_shadow_is():
    """The same pixels mean different ground points for the two classes. The crate sits on the
    ground inside its box; the cube floats, and its ground point is under the box's bottom edge."""
    xyxy = (960.0, 400.0, 1025.0, 465.0)
    crate_run, cube_run = _Run(), _Run()
    crate_run.tick([Detection(CUBE_BOX, 0.9, xyxy)])
    cube_run.tick([Detection(CUBE_DROPPED, 0.9, xyxy)])
    crate_y = crate_run.loot.objects[0].pos[1]
    cube_y = cube_run.loot.objects[0].pos[1]
    bottom_y = _screen_to_tile((xyxy[0] + xyxy[2]) / 2, xyxy[3])[1]
    assert crate_y < bottom_y < cube_y


def test_the_cell_is_fixed_at_confirmation_and_does_not_follow_the_median():
    """Odometry drift and the lattice phase move a crate's readings; the grid must not flicker
    between two cells because of it."""
    cx, cy = CELL
    run = _Run()
    run.tick([_box(CUBE_BOX, (cx + 0.9, cy + 0.5))], MIN_HITS)
    assert run.loot.crates() == [(cx + 0.5, cy + 0.5)]
    run.tick([_box(CUBE_BOX, (cx + 1.2, cy + 0.5))], 30)
    assert run.loot.objects[0].pos[0] > cx + 1           # the median did cross the boundary
    assert run.loot.crates() == [(cx + 0.5, cy + 0.5)]


def test_the_map_is_in_world_tiles_so_a_moving_camera_keeps_a_crate_in_place():
    run = _Run()
    for i in range(MIN_HITS + 3):
        run.odo = _Odo(position_tiles=(0.3 * i, -0.2 * i))
        run.tick([_box(CUBE_BOX, SPOT, cam=run.odo.position_tiles)])
    assert run.loot.crates() == [SPOT] and len(run.loot.objects) == 1


# -- merging and the edge cut -------------------------------------------------------------------

def test_two_boxes_on_one_crate_are_one_crate():
    run = _Run()
    a = _box(CUBE_BOX, SPOT, conf=0.9)
    b = _box(CUBE_BOX, (SPOT[0] + 0.1, SPOT[1] + 0.05), conf=0.6)
    res = run.tick([a, b], MIN_HITS)
    assert res.n_merged == 1
    assert len(run.loot.objects) == 1 and run.loot.crates() == [SPOT]


def test_the_closest_distinct_crates_measured_stay_two():
    run = _Run()
    run.tick([_box(CUBE_BOX, SPOT), _box(CUBE_BOX, (SPOT[0] + 0.663, SPOT[1]))], MIN_HITS)
    assert len(run.loot.crates()) == 2


def test_a_crate_and_a_cube_on_one_spot_are_not_merged():
    """Merging is per class: a cube lying beside its crate is two things."""
    run = _Run()
    run.tick([_box(CUBE_BOX, SPOT), _box(CUBE_DROPPED, SPOT)], MIN_HITS)
    assert run.loot.crates() == [SPOT] and run.loot.cubes() == [SPOT]


@pytest.mark.parametrize("edge,xyxy", [
    ("left", (EDGE_MARGIN_PX - 5, 400, EDGE_MARGIN_PX + 125, 567)),
    ("top", (900, EDGE_MARGIN_PX - 5, 1030, EDGE_MARGIN_PX + 162)),
    ("right", (W - EDGE_MARGIN_PX - 125, 400, W - EDGE_MARGIN_PX + 5, 567)),
    ("bottom", (900, H - EDGE_MARGIN_PX - 162, 1030, H - EDGE_MARGIN_PX + 5)),
])
def test_a_box_near_any_edge_is_dropped(edge, xyxy):
    run = _Run()
    res = run.tick([Detection(CUBE_BOX, 0.9, tuple(map(float, xyxy)))], MIN_HITS)
    assert res.n_edge == 1 and run.loot.objects == [], edge


def test_a_box_just_inside_the_margin_is_kept():
    run = _Run()
    m = EDGE_MARGIN_PX
    res = run.tick([Detection(CUBE_BOX, 0.9, (900.0, H - m - 167.0, 1030.0, float(H - m)))])
    assert res.n_edge == 0 and len(run.loot.objects) == 1


def test_projectiles_are_not_loot():
    run = _Run()
    run.tick([_box(PROJECTILE, SPOT)], MIN_HITS + 2)
    assert run.loot.objects == []


# -- when something is gone ---------------------------------------------------------------------

def test_a_crate_in_view_is_retired_after_gone_s_unseen_and_not_before():
    run = _Run()
    run.tick([_box(CUBE_BOX, SPOT)], MIN_HITS)
    held = math.floor(GONE_S[CUBE_BOX] / DT) - 1
    run.tick([], held)
    assert run.loot.crates() == [SPOT]
    run.tick([], 3)
    assert run.loot.crates() == [] and run.loot.objects == []


def test_a_sighting_clears_the_unseen_time():
    """A detector that blinks (the replay's crate gaps reach 1.9 s, 5 of them over 1 s) must not
    add its gaps up into a retirement."""
    run = _Run()
    crate = _box(CUBE_BOX, SPOT)
    run.tick([crate], MIN_HITS)
    for _ in range(10):
        run.tick([], math.floor(GONE_S[CUBE_BOX] / DT) - 2)
        run.tick([crate])
    assert run.loot.crates() == [SPOT]


def test_a_cube_goes_sooner_than_a_crate():
    run = _Run()
    run.tick([_box(CUBE_BOX, SPOT), _box(CUBE_DROPPED, (SPOT[0] + 2, SPOT[1]))], MIN_HITS)
    run.tick([], math.ceil(GONE_S[CUBE_DROPPED] / DT) + 1)
    assert run.loot.cubes() == []
    assert run.loot.crates() == [SPOT]


def test_a_crate_out_of_view_is_never_charged_for_being_missed():
    """The map is sticky, like the sim's channel: a crate the camera has left is still there."""
    run = _Run()
    run.tick([_box(CUBE_BOX, SPOT)], MIN_HITS)
    run.odo = _Odo(position_tiles=(40.0, 0.0))
    run.tick([], 12 * 10)
    assert run.loot.crates() == [SPOT]
    # Back in view and still not there: now it counts.
    run.odo = _Odo()
    run.tick([], math.ceil(GONE_S[CUBE_BOX] / DT) + 2)
    assert run.loot.crates() == []


def test_a_crate_under_the_joystick_is_not_charged_for_being_missed():
    """`plan.valid` knows where the HUD is; a crate drawn under it cannot be seen."""
    hud = _screen_to_tile(300, 850)
    r = PLAN.tile_to_rect([hud])[0]
    assert not PLAN.valid[int(r[1]), int(r[0])], "the joystick is no longer at (300, 850)"
    run = _Run()
    run.tick([_box(CUBE_BOX, SPOT)], MIN_HITS)
    run.odo = _Odo(position_tiles=(SPOT[0] - hud[0], SPOT[1] - hud[1]))
    run.tick([], 12 * 10)
    assert run.loot.crates() == [SPOT]


def test_a_crate_whose_box_would_be_cut_by_the_edge_is_not_charged():
    """The edge rule throws such a box away, so missing it is not evidence either way. The
    spot itself is on screen: only the box would cross the margin."""
    bottom = _screen_to_tile(1000, H - 40)          # anchor on screen, box past the bottom margin
    run = _Run()
    run.tick([_box(CUBE_BOX, SPOT)], MIN_HITS)
    run.odo = _Odo(position_tiles=(SPOT[0] - bottom[0], SPOT[1] - bottom[1]))
    run.tick([], 12 * 10)
    assert run.loot.crates() == [SPOT]


def test_an_unconfirmed_sighting_is_forgotten():
    run = _Run()
    run.tick([_box(CUBE_BOX, SPOT)], MIN_HITS - 1)
    run.tick([], math.ceil(GONE_S[CUBE_BOX] / DT) + 1)
    assert run.loot.objects == []


# -- the odometry interlock ---------------------------------------------------------------------

def test_a_new_segment_drops_every_object():
    run = _Run()
    run.tick([_box(CUBE_BOX, SPOT)], MIN_HITS)
    run.odo = _Odo(segment=1)
    res = run.tick([_box(CUBE_BOX, SPOT)])
    assert res.status == "reset" and run.loot.objects == []


def test_the_first_tick_adopts_the_segment_instead_of_resetting_on_it():
    run = _Run()
    run.odo = _Odo(segment=7)
    assert run.tick([_box(CUBE_BOX, SPOT)]).status == "ok"
    assert len(run.loot.objects) == 1


def test_an_untrusted_stretch_neither_sights_nor_counts_as_time_spent_looking():
    run = _Run()
    run.tick([_box(CUBE_BOX, SPOT)], MIN_HITS)
    run.odo = _Odo(status="uncertain")
    res = run.tick([_box(CUBE_BOX, (SPOT[0] + 5, SPOT[1]))], 12 * 5)
    assert res.status == "uncertain" and len(run.loot.objects) == 1
    run.odo = _Odo()
    run.tick([], 2)
    assert run.loot.crates() == [SPOT]


def test_reset_clears_everything_and_takes_the_new_segment():
    """What the loop calls at the start of a match."""
    run = _Run()
    run.tick([_box(CUBE_BOX, SPOT)], MIN_HITS)
    run.loot.reset(3)
    assert run.loot.objects == [] and run.loot.crates() == []
    run.odo = _Odo(segment=3)
    assert run.tick([_box(CUBE_BOX, SPOT)]).status == "ok"


# -- the occupancy map's fifth gate -------------------------------------------------------------

def test_no_crate_masks_nothing():
    cols, rows = PLAN.size_tiles
    mask = crate_occlusion([_box(CUBE_DROPPED, SPOT), _box(PROJECTILE, SPOT)], PLAN)
    assert mask.shape == (rows, cols) and not mask.any()
    assert not crate_occlusion([], PLAN).any()


def test_a_crate_masks_the_cell_it_stands_on_and_only_a_few_around_it():
    """The sprite is tall, so it covers cells behind the crate too. On the replay one crate box
    masked 5-11 cells (p10-p90), never more than 12, of the patch's 570."""
    mask = crate_occlusion([_box(CUBE_BOX, SPOT)], PLAN)
    col = CELL[0] - PLAN.origin_tile[0]
    row = CELL[1] - PLAN.origin_tile[1]
    assert mask[row, col]
    assert 1 <= mask.sum() <= 12, mask.sum()


def test_the_mask_is_the_same_box_the_detector_drew():
    """Warped through `plan.M` like the frame is, so a box twice as tall covers more cells."""
    short = crate_occlusion([_box(CUBE_BOX, SPOT, h=100.0)], PLAN).sum()
    tall = crate_occlusion([_box(CUBE_BOX, SPOT, h=300.0)], PLAN).sum()
    assert tall > short


def test_box_occlusion_masks_every_box_whatever_its_label():
    """The brawler mask: the entity model's labels are nothing `crate_occlusion` filters for, and
    the same box must mask the same cells under any of them."""
    box = _box(CUBE_BOX, SPOT)
    crate = crate_occlusion([box], PLAN)
    assert crate.any()
    for label in ("player", "enemy", "teammate"):
        relabelled = dataclasses.replace(box, label=label)
        assert not crate_occlusion([relabelled], PLAN).any()
        assert (box_occlusion([relabelled], PLAN) == crate).all()
    assert not box_occlusion([], PLAN).any()


# -- the startup check --------------------------------------------------------------------------

DEPLOY3 = ("blocks_unit", "blocks_projectile", "is_bush", "is_water", "in_zone",
           "enemy_revealed", "hero", "box", "pickup", "projectile")


def test_a_grid_that_reads_crates_needs_a_model_with_the_class():
    with pytest.raises(ValueError, match="Power Cube Box"):
        require_loot_classes({0: CUBE_DROPPED, 1: PROJECTILE}, DEPLOY3)
    with pytest.raises(ValueError, match="Power Cube Dropped"):
        require_loot_classes([CUBE_BOX, PROJECTILE], DEPLOY3)


def test_a_grid_that_reads_neither_needs_neither():
    require_loot_classes({0: PROJECTILE}, [ch for ch in DEPLOY3 if ch not in ("box", "pickup")])
    require_loot_classes({0: CUBE_BOX, 1: CUBE_DROPPED, 2: PROJECTILE}, DEPLOY3)


def test_to_tiles_agrees_with_the_test_boxes():
    """The helper above runs the projection backwards; this is the check that it inverts it."""
    for label in (CUBE_BOX, CUBE_DROPPED):
        det = _box(label, SPOT)
        (_, (tx, ty)), = to_tiles([det], PLAN, anchor_frac=ANCHOR_FRAC[label])
        assert (tx, ty) == pytest.approx(SPOT, abs=1e-6)
