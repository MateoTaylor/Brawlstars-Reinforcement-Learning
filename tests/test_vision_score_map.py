"""The map scorer's arithmetic (`scripts/vision_score_map.py`). What it measures on real footage is in
`occupancy.py` and `classifier.py`; this checks it counts what it says it counts."""
from types import SimpleNamespace

import numpy as np
import pytest

from brawl_sim.constants import Tile
from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.terrain.labeling import CLASS_INDEX, observed_cells
from brawl_vision.terrain.occupancy import OccupancyMap
from scripts.vision_score_map import MapScore, Tally, label_crop, wall_edges, window_frames

F, W, B = CLASS_INDEX[Tile.FLOOR], CLASS_INDEX[Tile.WALL], CLASS_INDEX[Tile.BUSH]


@pytest.fixture(scope="module")
def plan():
    return build_rectify_plan(load_camera_model(), load_hud_mask())


def test_a_tally_is_blocking_recall_and_precision():
    t = Tally()
    t.add(np.array([True, True, False, False]), np.array([True, False, True, False]))
    assert (t.tp, t.fn, t.fp, t.tn) == (1, 1, 1, 1)
    assert t.recall == t.precision == t.f1 == 0.5
    assert Tally().f1 == 0.0, "an empty tally must not divide by zero"


def test_edges_are_walls_beside_a_labelled_free_cell():
    truth = np.array([[W, W, W, W],
                      [W, W, W, F],
                      [W, W, W, -1]])
    assert wall_edges(truth).tolist() == [[False, False, False, True],
                                          [False, False, True, False],
                                          [False, False, False, False]]


def test_a_map_is_scored_only_where_it_saw_a_labelled_cell_in_the_footprint():
    truth = np.array([[W, F, B, -1]])
    best = np.array([[W, W, -1, F]])
    s = MapScore()
    s.add(truth, best, scored=np.array([[True, True, True, True]]))
    assert s.cells == 2                              # unseen and unlabelled cells drop out
    assert (s.blocking.tp, s.blocking.fp) == (1, 1)
    assert s.accuracy == 0.5
    s.add(truth, best, scored=np.array([[False, True, True, True]]))
    assert s.cells == 3 and s.windows == 2


def test_the_window_is_on_the_labels_phase_and_clamped():
    frames = window_frames(label=12, fps=60.0, last=700, rate_hz=12.0, reach_s=1.0)
    assert frames[0] == 2 and frames[-1] == 72 and 12 in frames
    assert all((f - 12) % 5 == 0 for f in frames)
    assert window_frames(label=697, fps=60.0, last=700, reach_s=1.0)[-1] == 697


@pytest.mark.parametrize("d", [(2.0, 0.0), (2.3, -1.0), (-3.6, 1.45)])
def test_a_view_lands_on_the_labels_cells(plan, d):
    """A view `d` tiles from the labelled frame, deposited through the plan registered there, must
    fill the label's cells shifted by round(d) -- the scorer's whole premise."""
    cols, rows = plan.size_tiles
    cells = (np.arange(rows * cols) % 5).reshape(rows, cols).astype(np.int8)
    occ = OccupancyMap(128, 128)
    registered = plan.registered(d)
    occ.update(cells, SimpleNamespace(position_tiles=d, status="ok", segment=0), registered)
    crop = label_crop(occ, plan)
    sx, sy = int(np.round(d[0])), int(np.round(d[1]))
    fp = OccupancyMap(1, 1)._footprint(registered)
    for r, c in zip(*np.nonzero(fp)):
        if 0 <= r + sy < rows and 0 <= c + sx < cols:
            assert crop[r + sy, c + sx] == cells[r, c]
    assert observed_cells(plan).any()
