"""Known-map labels: the loader's checks, the labeling tool's first pass, and the Dark Passage label.
KNOWN_MAP_LOCALIZATION_PLAN.md step K1.

The spawn cells below were measured from the image by eye before `scripts/map_label.py` existed,
and the user confirmed them when signing the label off (2026-09-28), so the finder test can fail.
A crate list measured the same way passed its test too and was wrong: the sprites were candles.
Two readings by one reader agree on where a thing is, not on what it is.
"""
import json

import cv2
import numpy as np
import pytest

from brawl_deployment.perception.known_map import (
    MAPS_DIR, KnownMap, check, read_chars, write_chars,
)
from brawl_sim.constants import Tile
from brawl_vision.terrain.labeling import CLASS_INDEX

SPAWNS = [(6, 5), (26, 5), (43, 6), (53, 15), (6, 24), (57, 38), (6, 47), (47, 53), (34, 54),
          (15, 55)]


@pytest.fixture(autouse=True)
def _restore_rcparams():
    """The tool trims matplotlib's global keymaps for its own keys, which is right in its own window;
    a test must not leave that behind for the rest of the suite (test_play_manual reads the
    defaults)."""
    import matplotlib
    with matplotlib.rc_context():
        yield


def _tool():
    from scripts import map_label
    return map_label


def _open_floor() -> np.ndarray:
    chars = np.full((60, 60), ".", dtype="<U1")
    chars[5, 6] = "S"
    chars[10, 20:25] = "#"
    chars[40, 40] = "~"
    chars[30, 30] = "f"
    chars[50, 10] = "b"
    return chars


def _cells(world_points) -> list[tuple[int, int]]:
    """World-tile centres back to `(col, row)`; asserts each sat on a centre."""
    out = []
    for x, y in world_points:
        col, row = x + 30 - 0.5, y + 30 - 0.5
        assert col == int(col) and row == int(row), f"({x}, {y}) is not a tile centre"
        out.append((int(col), int(row)))
    return sorted(out)


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------

def test_the_loader_round_trips_with_centres_at_half_a_tile(tmp_path):
    chars = _open_floor()
    path = tmp_path / "some_map.csv"
    write_chars(path, chars)
    assert (read_chars(path) == chars).all()

    known = KnownMap.load(path)
    assert known.name == "some_map"
    assert known.tiles[30, 30] == Tile.FENCE and known.tiles[40, 40] == Tile.WATER
    # Tile (c, r)'s centre is world (c - 29.5, r - 29.5): the map frame of plan section 2.
    assert known.spawns.tolist() == [[6 - 29.5, 5 - 29.5]]


def test_classes_are_the_classifier_alphabet_with_a_spawn_read_as_floor(tmp_path):
    """What `localize` scores a classified view against (step K2)."""
    path = tmp_path / "some_map.csv"
    write_chars(path, _open_floor())
    classes = KnownMap.load(path).classes
    assert classes[5, 6] == CLASS_INDEX[Tile.FLOOR]
    assert classes[10, 22] == CLASS_INDEX[Tile.WALL]
    assert classes[40, 40] == CLASS_INDEX[Tile.WATER]
    assert classes[30, 30] == CLASS_INDEX[Tile.FENCE]
    assert classes[50, 10] == CLASS_INDEX[Tile.BUSH]
    assert classes[0, 0] == CLASS_INDEX[Tile.FLOOR]


@pytest.mark.parametrize("damage, message", [
    (lambda c: c[:59], "a map is 60 x 60"),
    (lambda c: _set(c, 12, 34, "Z"), "'Z' at (col 34, row 12) is not in the legend"),
    # Boxes spawn semi-randomly each match, so no map image shows one and no label holds one.
    (lambda c: _set(c, 2, 7, "X"), "'X' at (col 7, row 2) is not in the legend"),
    (lambda c: _set(c, 12, 34, "?"), "1 cells are still unlabelled"),
    (lambda c: _set(c, 5, 6, "."), "no spawn"),
])
def test_the_checks_refuse_a_broken_label(tmp_path, damage, message):
    path = tmp_path / "m.csv"
    write_chars(path, damage(_open_floor()))
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        KnownMap.load(path)


def _set(chars, row, col, char):
    chars[row, col] = char
    return chars


@pytest.mark.parametrize("text, message", [
    ("\n".join([",".join("." * 60)] * 7 + [",".join("." * 59)]), "row 7 has 59 cells"),
    (",".join(["##"] + ["."] * 59), "is '##', not one character"),
])
def test_a_malformed_file_is_refused_before_the_checks(tmp_path, text, message):
    path = tmp_path / "m.csv"
    path.write_text(text + "\n")
    with pytest.raises(ValueError, match=message):
        read_chars(path)


@pytest.mark.parametrize("barrier", ["#", "~", "f"])
def test_a_sealed_off_pocket_is_refused(barrier):
    """Walls, water and fences all stop a body (`TILE_BLOCKS_UNIT`), so a ring of any of them
    around one floor cell leaves ground no spawn can reach."""
    chars = _open_floor()
    chars[19:22, 39:42] = barrier
    chars[20, 40] = "."
    with pytest.raises(ValueError, match="2 separate regions"):
        check(chars)


def test_a_ring_of_bush_seals_nothing():
    chars = _open_floor()
    chars[19:22, 39:42] = "b"
    chars[20, 40] = "."
    check(chars)


# ---------------------------------------------------------------------------
# the first pass, on the checked-in image
# ---------------------------------------------------------------------------

def test_the_first_pass_finds_the_measured_spawns():
    ml = _tool()
    img = cv2.imread(str(MAPS_DIR / "dark_passage.png"))
    lat, clarity = ml.fit_lattice(img)
    assert lat.pitch == pytest.approx((12.49, 12.49), abs=0.02)
    assert clarity > 2.0
    assert ml.find_spawns(img, lat) == sorted(SPAWNS, key=lambda cr: (cr[1], cr[0]))


def test_the_signed_off_dark_passage_label_passes_the_checks_with_its_spawns():
    known = KnownMap.load("dark_passage")
    assert _cells(known.spawns) == sorted(SPAWNS)


# ---------------------------------------------------------------------------
# the tool, headless
# ---------------------------------------------------------------------------

def _run_tool(monkeypatch, tmp_path, argv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ml = _tool()
    monkeypatch.setattr(ml, "MAPS_DIR", tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    assert ml.main(argv) == 0
    return plt.gcf()._brawl_labeller


def _key(lab, key, cell=None):
    from matplotlib.backend_bases import KeyEvent
    xy = _display(lab, cell) if cell else (None, None)
    KeyEvent("key_press_event", lab.fig.canvas, key, x=xy[0], y=xy[1])._process()


def _click(lab, cell):
    from matplotlib.backend_bases import MouseEvent
    x, y = _display(lab, cell)
    MouseEvent("button_press_event", lab.fig.canvas, x, y, button=1)._process()
    MouseEvent("button_release_event", lab.fig.canvas, x, y, button=1)._process()


def _display(lab, cell):
    col, row = cell
    (px, py), (ox, oy) = lab.lat.pitch, lab.lat.origin
    return lab.ax.transData.transform((ox + (col + 0.5) * px, oy + (row + 0.5) * py))


def test_a_new_map_opens_on_the_first_pass_and_saves_what_is_painted(monkeypatch, tmp_path,
                                                                      capsys):
    lab = _run_tool(monkeypatch, tmp_path, ["dp", "--image", str(MAPS_DIR / "dark_passage.png")])
    assert (tmp_path / "dp.png").read_bytes() == (MAPS_DIR / "dark_passage.png").read_bytes()
    assert int((lab.chars == "S").sum()) == 10
    assert set(np.unique(lab.chars)) == {"?", "S"}
    assert lab.mode == "clusters"

    _key(lab, "2")
    _click(lab, (0, 8))
    assert lab.chars[8, 0] == "#"

    # A cluster fill names every cell of the cluster under the cursor, except spawns.
    # Two made-up clusters, left and right halves, so the fill's extent is known exactly.
    assert lab.clusters.shape == (60, 60) and len(np.unique(lab.clusters)) == 12
    lab.clusters = np.zeros((60, 60), np.int32)
    lab.clusters[:, :30] = 1
    before = lab.chars.copy()
    _key(lab, "1")
    _key(lab, "c", (20, 40))
    left, marked = lab.clusters == 1, before == "S"
    assert (left & marked).any()
    assert (lab.chars[left & ~marked] == ".").all()
    assert (lab.chars[marked | ~left] == before[marked | ~left]).all()

    # One `u` takes back the whole fill, and nothing before it.
    _key(lab, "u")
    assert (lab.chars == before).all()
    assert lab.chars[8, 0] == "#"

    _key(lab, "s")
    assert (read_chars(tmp_path / "dp.csv") == lab.chars).all()
    meta = json.loads((tmp_path / "dp.json").read_text())
    assert meta["pitch_px"] == [round(v, 4) for v in lab.lat.pitch]
    assert "checks: FAIL" in capsys.readouterr().out, "an unfinished label saved as passing"

    again = _run_tool(monkeypatch, tmp_path, ["dp"])
    assert again.mode == "label"
    assert (again.chars == lab.chars).all()
    assert again.lat == lab.lat.from_dict(meta)


def test_a_different_image_never_replaces_the_one_a_label_was_drawn_on(monkeypatch, tmp_path):
    (tmp_path / "dp.png").write_bytes(b"not the same image")
    with pytest.raises(SystemExit, match="already exists and differs"):
        _run_tool(monkeypatch, tmp_path, ["dp", "--image", str(MAPS_DIR / "dark_passage.png")])


def test_matplotlib_does_not_keep_the_keys_the_tool_needs():
    """`s` opens a save dialog, `q` closes the window, `g` toggles the axes grid and `c` walks the
    view stack; both handlers would run (see vision_label.py)."""
    import matplotlib
    ml = _tool()
    ml._release_matplotlib_keys()
    clashes = {name: sorted(set(v) & ml._OURS) for name, v in matplotlib.rcParams.items()
               if name.startswith("keymap.") and isinstance(v, list) and set(v) & ml._OURS}
    assert not clashes, f"matplotlib still claims {clashes}"
