"""Per-cell terrain labelling and classification. See Terrain_Perception_Build_Plan.md Phase H.

No real labels exist yet -- that is the phase's outstanding input -- so the network and the label
store are exercised against synthetic terrain with a known answer. That checks the machinery is
right; it says nothing about accuracy on real footage, which is exactly the split this file keeps
explicit rather than blurring.
"""
from pathlib import Path

import numpy as np
import pytest

from brawl_sim.constants import TILE_TO_CHAR, Tile
from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.terrain import labeling
from brawl_vision.terrain.classifier import (
    IGNORE_INDEX, Example, TerrainClassifier, TerrainNet, augment, build_examples, evaluate,
    flicker_rate, flip_pair, to_tensor, train,
)
from brawl_vision.terrain.labeling import (
    CLASS_CHARS, CLASSES, UNLABELLED, LabelGrid, cell_features, propose_clusters,
)


@pytest.fixture(scope="module")
def plan():
    return build_rectify_plan(load_camera_model(), load_hud_mask())


def _empty(plan, clip="c", frame=1):
    return LabelGrid(clip=clip, frame=frame, origin_tile=plan.origin_tile,
                     size_tiles=plan.size_tiles, pixels_per_tile=plan.pixels_per_tile)


# ---------------------------------------------------------------------------
# the label store
# ---------------------------------------------------------------------------

def test_a_new_grid_is_entirely_unlabelled(plan):
    g = _empty(plan)
    cols, rows = plan.size_tiles
    assert g.chars.shape == (rows, cols)
    assert not g.labelled.any()
    assert (g.as_class_index() == -1).all()


def test_marker_tiles_are_refused_as_labels(plan):
    """SPAWN and BOX are placement markers, not terrain, and must never be classifier outputs
    (plan Section 2). The legend can spell them, so the label store is where that is enforced."""
    g = _empty(plan)
    for char in (TILE_TO_CHAR[Tile.SPAWN], TILE_TO_CHAR[Tile.BOX]):
        with pytest.raises(ValueError, match="not a terrain class"):
            g.set_cell(0, 0, char)


def test_unlabelled_maps_to_minus_one_not_a_class(plan):
    """-1 rather than a sixth class, so an accidental argmax over the target cannot silently
    produce FLOOR for every cell nobody looked at."""
    g = _empty(plan)
    g.set_cell(2, 3, TILE_TO_CHAR[Tile.WALL])
    idx = g.as_class_index()
    assert idx[2, 3] == labeling.CLASS_INDEX[Tile.WALL]
    assert idx[0, 0] == -1


def test_round_trips_through_disk(plan, tmp_path):
    g = _empty(plan, clip="zone_grows_from_east", frame=700)
    g.set_cell(1, 1, TILE_TO_CHAR[Tile.BUSH])
    g.set_cell(4, 9, TILE_TO_CHAR[Tile.WATER])
    g.notes = "hand check"
    path = tmp_path / "g.json"
    g.save(path)
    back = LabelGrid.load(path, plan)
    assert (back.chars == g.chars).all()
    assert back.clip == g.clip and back.frame == 700 and back.notes == "hand check"


def test_the_file_is_one_line_per_row_so_a_diff_is_readable(plan, tmp_path):
    """The format exists to be reviewed by a human. A changed cell must show up as one character
    on one line, which nested JSON arrays would not give."""
    g = _empty(plan)
    g.set_cell(0, 0, TILE_TO_CHAR[Tile.WALL])
    path = tmp_path / "g.json"
    g.save(path)
    text = path.read_text()
    cols, rows = plan.size_tiles
    assert f'"{TILE_TO_CHAR[Tile.WALL]}{UNLABELLED * (cols - 1)}"' in text
    assert text.count(UNLABELLED * (cols - 1)) >= 1
    assert '"legend"' in text, "a reader should not have to look up what the characters mean"


def test_labels_drawn_against_other_geometry_are_refused(plan, tmp_path):
    """A recalibration that moved the window would leave every label pointing at different world,
    with nothing about the file looking wrong. Silent is the whole danger, so this raises."""
    g = _empty(plan)
    g.origin_tile = (0, 0)
    path = tmp_path / "g.json"
    g.save(path)
    with pytest.raises(ValueError, match="different cells now"):
        LabelGrid.load(path, plan)


def test_load_label_dir_reads_every_file(plan, tmp_path):
    for i in (1, 2, 3):
        _empty(plan, frame=i).save(tmp_path / f"f{i}.json")
    assert [g.frame for g in labeling.load_label_dir(tmp_path, plan)] == [1, 2, 3]


# ---------------------------------------------------------------------------
# cell features and clustering
# ---------------------------------------------------------------------------

def test_cell_features_are_not_transposed(plan):
    """The failure that would be invisible everywhere else: a reshape that swapped rows and
    columns still produces a plausible grid, and every label would then be wrong."""
    ppt = plan.pixels_per_tile
    img = np.zeros((plan.size_px[1], plan.size_px[0], 3), np.uint8)
    img[3 * ppt:4 * ppt, 7 * ppt:8 * ppt] = (0, 0, 255)
    v = cell_features(img, plan)[..., 2]
    assert np.argwhere(v > 1).tolist() == [[3, 7]]


def test_cell_features_capture_spread_not_just_mean(plan):
    """A flat cell and a textured cell can share a mean colour; the standard-deviation half of the
    feature is what tells them apart, and it is what the cluster proposal leans on."""
    ppt = plan.pixels_per_tile
    flat = np.full((plan.size_px[1], plan.size_px[0], 3), 128, np.uint8)
    noisy = flat.copy()
    rng = np.random.default_rng(0)
    noisy[:ppt, :ppt] = rng.integers(0, 255, (ppt, ppt, 3), dtype=np.uint8)
    assert cell_features(flat, plan)[0, 0, 5] < cell_features(noisy, plan)[0, 0, 5]


def test_clusters_group_identical_cells_together(plan):
    ppt = plan.pixels_per_tile
    img = np.zeros((plan.size_px[1], plan.size_px[0], 3), np.uint8)
    cols, rows = plan.size_tiles
    for y in range(rows):
        for x in range(cols):
            img[y * ppt:(y + 1) * ppt, x * ppt:(x + 1) * ppt] = (
                (40, 40, 200) if x < cols // 2 else (40, 200, 40))
    lab, valid = propose_clusters(img, plan, k=2)
    left = lab[:, :cols // 2][valid[:, :cols // 2]]
    right = lab[:, cols // 2:][valid[:, cols // 2:]]
    assert len(set(left.tolist())) == 1 and len(set(right.tolist())) == 1
    assert set(left.tolist()) != set(right.tolist())


def test_clusters_never_claim_cells_outside_the_viewport(plan):
    lab, valid = propose_clusters(np.zeros((plan.size_px[1], plan.size_px[0], 3), np.uint8), plan)
    assert (lab[~valid] == -1).all()


# ---------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------

def test_the_output_grid_is_exactly_the_tile_grid(plan):
    """A stride that did not divide the patch into whole cells would need resampling onto the tile
    grid, smearing every cell into its neighbours."""
    cols, rows = plan.size_tiles
    x = to_tensor(np.zeros((plan.size_px[1], plan.size_px[0], 3), np.uint8))
    assert tuple(TerrainNet()(x).shape) == (1, len(CLASSES), rows, cols)


def test_the_receptive_field_spans_more_than_two_tiles(plan):
    """Not a preference. Phase C measured the wall parallax at ~0.88 tiles, so a wall's visible top
    is nearly a whole cell from the footprint that blocks movement; a head seeing only its own cell
    cannot see the wall it is meant to report."""
    assert TerrainNet.receptive_field_px() >= 2 * plan.pixels_per_tile


def test_the_model_is_small_enough_to_run_on_cpu(plan):
    assert sum(p.numel() for p in TerrainNet().parameters()) < 1_000_000


def test_hue_augmentation_moves_hue_without_destroying_structure():
    """Hue rotation is the reskin strategy: it must change colour a lot and leave edges alone."""
    import cv2
    rng = np.random.default_rng(3)
    img = np.zeros((96, 96, 3), np.uint8)
    img[:48] = (30, 180, 60)
    img[48:] = (200, 60, 40)
    out = augment(img, rng, hue_deg=60, sat=0.0, val=0.0)
    a = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 0].astype(int)
    b = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[..., 0].astype(int)
    assert np.abs(a - b).mean() > 2, "hue did not move"
    edge = lambda z: np.abs(np.diff(cv2.cvtColor(z, cv2.COLOR_BGR2GRAY).astype(int), axis=0)).sum()
    assert edge(out) > 0.25 * edge(img), "the augmentation destroyed the structure it must keep"


def test_hue_is_rotated_on_the_0_179_wheel():
    """OpenCV's H is a HALF-degree wheel. Treating `hue_deg` as raw units is a silent 2x, which
    looks like "augmentation is a bit strong" rather than like a bug."""
    import cv2
    # A hue in the MIDDLE of the wheel: starting at 0 would wrap to 179 and make a +-20 swing
    # measure as a span of 178, which says nothing about the halving.
    img = np.full((16, 16, 3), 0, np.uint8)
    img[:] = cv2.cvtColor(np.array([[[90, 200, 200]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    seen = set()
    for seed in range(40):
        out = augment(img, np.random.default_rng(seed), hue_deg=40, sat=0.0, val=0.0)
        seen.add(int(cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[0, 0, 0]))
    span = max(seen) - min(seen)
    assert span <= 40, f"hue swing {span} exceeds +-20 units, i.e. the degree halving is missing"


def test_flips_move_patch_and_labels_together():
    rng = np.random.default_rng(0)
    rect = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    target = np.arange(16).reshape(4, 4)
    for seed in range(8):
        r, t = flip_pair(rect, target, np.random.default_rng(seed))
        assert r.shape == rect.shape and t.shape == target.shape
        # cell (i, j) of the flipped target must still describe pixel block (i, j) of the patch
        assert set(t.ravel().tolist()) == set(target.ravel().tolist())


def test_excluded_cells_are_dropped_from_the_target(plan):
    g = _empty(plan)
    g.set_cell(0, 0, TILE_TO_CHAR[Tile.WALL])
    g.set_cell(0, 1, TILE_TO_CHAR[Tile.WALL])
    cols, rows = plan.size_tiles
    drop = np.zeros((rows, cols), bool)
    drop[0, 1] = True
    rect = np.zeros((plan.size_px[1], plan.size_px[0], 3), np.uint8)
    ex = build_examples([(g, rect)], plan, exclude={("c", 1): drop})[0]
    assert ex.target[0, 0] == labeling.CLASS_INDEX[Tile.WALL]
    assert ex.target[0, 1] == IGNORE_INDEX


# ---------------------------------------------------------------------------
# training, on synthetic terrain with a known answer
# ---------------------------------------------------------------------------

def _synthetic(plan, seed):
    """Five materials with distinct colour AND texture frequency, so the task cannot be solved by
    reading one pixel -- which is what a real reskin-robust classifier must not rely on either."""
    rng = np.random.default_rng(seed)
    ppt = plan.pixels_per_tile
    cols, rows = plan.size_tiles
    base = [(60, 120, 60), (150, 60, 60), (40, 140, 40), (180, 90, 40), (90, 90, 140)]
    target = rng.integers(0, len(CLASSES), (rows, cols))
    img = np.zeros((plan.size_px[1], plan.size_px[0], 3), np.uint8)
    for y in range(rows):
        for x in range(cols):
            c = int(target[y, x])
            cell = np.zeros((ppt, ppt, 3), np.float32)
            cell[:] = base[c]
            cell += (np.sin(np.linspace(0, (c + 1) * np.pi * 2, ppt))[None, :] * 22)[..., None]
            img[y * ppt:(y + 1) * ppt, x * ppt:(x + 1) * ppt] = np.clip(
                cell + rng.normal(0, 4, cell.shape), 0, 255)
    return img, target.astype(np.int64)


@pytest.fixture(scope="module")
def trained(plan):
    examples = []
    for seed in range(3):
        img, target = _synthetic(plan, seed)
        examples.append(Example(rect=img, target=target, clip=f"synth{seed}"))
    return train(examples, epochs=30, seed=0), examples


def test_training_learns_a_separable_synthetic_task(plan, trained):
    """Proves the loop -- loss, ignore index, target alignment -- is wired correctly. It says
    nothing about real accuracy, which needs real labels."""
    model, examples = trained
    img, target = _synthetic(plan, 99)
    metrics = evaluate(model, [Example(rect=img, target=target, clip="holdout")])
    assert metrics["overall"] > 0.9, f"only {metrics['overall']:.2f} on a separable task"


def test_evaluate_reports_wall_versus_fence_separately(plan, trained):
    """Section 2: both block movement, only WALL blocks shots, so a fence read as a wall tells the
    policy it has cover from a shot that is about to hit it. Hiding that inside a mean is the
    specific thing the plan forbids."""
    model, examples = trained
    m = evaluate(model, examples)
    assert set(m["recall"]) == {t.name for t in CLASSES}
    assert "wall_vs_fence" in m and 0.0 <= m["wall_vs_fence"] <= 1.0


def test_predictions_do_not_flicker_on_a_static_scene(plan, trained):
    model, _ = trained
    img, _ = _synthetic(plan, 7)
    assert flicker_rate(model, [img, img, img]) == 0.0


def test_a_checkpoint_round_trips_and_pins_its_class_order(plan, trained, tmp_path):
    """A checkpoint whose head channels mean something else would permute every prediction
    silently -- the map would fill in confidently wrong."""
    model, _ = trained
    path = tmp_path / "t.pt"
    model.save(path)
    back = TerrainClassifier.load(path)
    img, _ = _synthetic(plan, 5)
    assert (back.predict(img)[0] == model.predict(img)[0]).all()

    import torch
    blob = torch.load(str(path), weights_only=False)
    blob["classes"] = ["FLOOR", "BUSH", "WALL", "WATER", "FENCE"]
    torch.save(blob, str(path))
    with pytest.raises(ValueError, match="channel order"):
        TerrainClassifier.load(path)


def test_predict_rejects_a_grid_that_stopped_matching_the_plan(plan, trained):
    model, _ = trained
    img, _ = _synthetic(plan, 1)
    cells, conf = model.predict(img, plan)
    cols, rows = plan.size_tiles
    assert cells.shape == (rows, cols) and conf.shape == (rows, cols)
    assert conf.min() >= 0.0 and conf.max() <= 1.0


def test_inference_is_cheap_enough_for_four_parallel_games(plan, trained):
    import time
    model, _ = trained
    img, _ = _synthetic(plan, 2)
    model.predict(img)
    t0 = time.perf_counter()
    for _ in range(10):
        model.predict(img)
    ms = (time.perf_counter() - t0) / 10 * 1e3
    assert ms < 30.0, f"{ms:.1f} ms/frame on CPU is too much of the 250 ms budget"


# ---------------------------------------------------------------------------
# the scripts
# ---------------------------------------------------------------------------

def test_training_script_reports_rather_than_inventing_a_number(tmp_path, capsys):
    """With no labels it must say so and stop, not train on nothing and print an accuracy."""
    from scripts import vision_train_terrain
    rc = vision_train_terrain.main(["--labels", str(tmp_path)])
    assert rc == 1
    assert "vision_label" in capsys.readouterr().err


def test_labelling_tool_refuses_a_frame_the_clip_does_not_have():
    from scripts import vision_label
    with pytest.raises(SystemExit):
        vision_label._frame_at("standstill", 10 ** 6)


def test_labelling_tool_names_a_missing_clip_rather_than_raising_a_traceback():
    from scripts import vision_label
    with pytest.raises(SystemExit, match="no_such_clip.mp4"):
        vision_label._frame_at("no_such_clip", 0)


# ---------------------------------------------------------------------------
# finding the frame a label points at
# ---------------------------------------------------------------------------

def test_a_clip_name_resolves_in_the_fixtures_before_training_videos(tmp_path, monkeypatch):
    """Several training_videos recordings are copies of fixture clips under the same name, and
    every label that predates the second directory was drawn on the fixture. Fixtures first keeps
    those labels on the file they were drawn from."""
    fixtures, videos = tmp_path / "fixtures", tmp_path / "training_videos"
    fixtures.mkdir()
    videos.mkdir()
    for path in (videos / "only_there.mp4", fixtures / "both.mp4", videos / "both.mp4"):
        path.write_bytes(b"")
    monkeypatch.setattr(labeling, "CLIP_DIRS", (fixtures, videos))
    assert labeling.find_clip("only_there") == videos / "only_there.mp4"
    assert labeling.find_clip("both") == fixtures / "both.mp4"
    with pytest.raises(FileNotFoundError):
        labeling.find_clip("nowhere")


def test_label_frames_bring_1080p_footage_to_the_calibrated_viewport(plan, monkeypatch):
    """The emulator recordings normalize to 1920x1080 and the camera model was fit at 2002x1126,
    so `rectify` refuses them as they come. The deployed capture resizes the same way."""
    decoded = []

    def walk(path, box, indices):              # walk_frames' contract: (index, normalized image)
        decoded.extend(indices)
        for i in indices:
            yield i, np.zeros((1080, 1920, 3), np.uint8)

    monkeypatch.setattr(labeling, "find_clip", lambda name: Path(f"{name}.mp4"))
    monkeypatch.setattr(labeling, "load_bounds", lambda path: {
        "n_frames": 10, "usable": [0, 5], "content_box": [0, 1920, 0, 1080]})
    monkeypatch.setattr(labeling, "walk_frames", walk)
    frames = labeling.read_label_frames("emulator", [3, 9, 3], plan.viewport)
    assert decoded == [3]             # 9 is past the usable range, and a repeat is decoded once
    assert list(frames) == [3]
    plan.rectify(frames[3])           # raises on any size but the calibrated one


@pytest.mark.vision
def test_label_frames_are_the_frames_clip_reader_numbers():
    """A label is addressed by frame index. The labeller and trainer read through
    `read_label_frames` while everything else reads through `ClipReader`; if the two numbered
    frames differently, a label would train on a neighbouring frame and nothing would look wrong."""
    from brawl_vision.config import VisionConfig
    from brawl_vision.sources import open_source
    want = {0, 37, 120}
    ours = labeling.read_label_frames("showdown_alternate_map", want)
    theirs = {}
    with open_source(labeling.find_clip("showdown_alternate_map"), VisionConfig()) as src:
        for frame in src:
            if frame.index in want:
                theirs[frame.index] = frame.image
            if frame.index >= max(want):
                break
    assert set(ours) == set(theirs) == want
    for i in sorted(want):
        assert np.array_equal(ours[i], theirs[i]), f"frame {i} differs"


# ---------------------------------------------------------------------------
# the labelling tool's event wiring
# ---------------------------------------------------------------------------

def _live_labeller(plan, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from scripts import vision_label as vl
    from brawl_vision.terrain.labeling import propose_clusters
    rect = np.zeros((plan.size_px[1], plan.size_px[0], 3), np.uint8)
    rect[:] = (60, 90, 60)
    grid = LabelGrid(clip="c", frame=0, origin_tile=plan.origin_tile,
                     size_tiles=plan.size_tiles, pixels_per_tile=plan.pixels_per_tile)
    unusable = ~vl.grid_observed(plan)
    clusters, _ = propose_clusters(rect, plan, k=4)
    vl._release_matplotlib_keys()
    lab = vl.Labeller("c", 0, plan, rect, grid, unusable, clusters, tmp_path / "l.json")
    lab.fig._brawl_labeller = lab
    return vl, lab, grid


def test_the_labeller_survives_garbage_collection(plan, tmp_path):
    """The bug that made the tool completely inert: `mpl_connect` stores BOUND METHODS BEHIND WEAK
    REFERENCES, so a Labeller nobody holds a reference to is collected immediately and every
    handler stops firing. The window still opens and draws correctly -- it just ignores every key
    and click, which reads as anything except a lifetime problem.
    """
    import gc

    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent

    vl, lab, grid = _live_labeller(plan, tmp_path)
    fig = lab.fig
    del lab
    gc.collect()
    KeyEvent("key_press_event", fig.canvas, "2")._process()
    assert "WALL" in fig.axes[0].get_title(), "handlers died with the instance"
    plt.close(fig)


def test_matplotlib_does_not_keep_the_keys_the_tool_needs(plan, tmp_path):
    """`s` opens a save dialog, `q` closes the window, `g` toggles the axes grid and `c` walks the
    view stack. Both handlers run, so the tool appears to do something random alongside what was
    asked -- and nothing about that reads as a keybinding conflict."""
    import matplotlib

    import matplotlib.pyplot as plt

    vl, lab, _ = _live_labeller(plan, tmp_path)
    clashes = {name: sorted(set(v) & vl._OURS)
               for name, v in matplotlib.rcParams.items()
               if name.startswith("keymap.") and isinstance(v, list) and set(v) & vl._OURS}
    plt.close(lab.fig)
    assert not clashes, f"matplotlib still claims {clashes}"
