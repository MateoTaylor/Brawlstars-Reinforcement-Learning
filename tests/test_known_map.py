"""Known-map labels: the loader's checks, the labeling tool's first pass, and the Dark Passage,
Starfruit Supernova and Truth labels.
archive/KNOWN_MAP_LOCALIZATION_PLAN.md step K1.

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


# Starfruit Supernova (2026-11-04). Unlike Dark Passage's, these came from `find_spawns` and were
# then checked by eye against the image's ten rings, each centred within 0.03 of a tile centre, so
# the two tests below pin the label and the finder rather than prove either right. Neither has been
# signed off by the user yet (MAP_ONBOARDING.md).
STARFRUIT_SPAWNS = [(26, 4), (44, 6), (14, 7), (51, 14), (7, 22), (56, 35), (5, 40), (55, 45),
                    (17, 53), (37, 54)]


def test_the_first_pass_fits_starfruit_supernova_and_finds_its_ten_spawns():
    ml = _tool()
    img = cv2.imread(str(MAPS_DIR / "starfruit_supernova.png"))
    lat, clarity = ml.fit_lattice(img)
    assert lat.pitch == pytest.approx((15.74, 15.72), abs=0.02)
    assert clarity > 2.0
    side = json.loads((MAPS_DIR / "starfruit_supernova.json").read_text())
    assert side["pitch_px"] == pytest.approx(lat.pitch, abs=1e-3)
    assert side["origin_px"] == pytest.approx(lat.origin, abs=1e-3)
    assert ml.find_spawns(img, lat) == STARFRUIT_SPAWNS


def test_the_starfruit_supernova_label_passes_the_checks_with_its_spawns():
    known = KnownMap.load("starfruit_supernova")
    assert _cells(known.spawns) == sorted(STARFRUIT_SPAWNS)


# Truth (2026-11-05). As with Starfruit's, these came from `find_spawns`. Each ring was then centred
# by eye within 0.06 of a tile centre (fractional parts .46-.56). One spawn has its own witness:
# the user's clip starts on (12,26). The game draws Truth with exact half-turn symmetry, and the
# label has it too, wall textures included, so a cell edited on one side only fails the second
# test. Teleporter pads are `#` (the user: "recognize them as walls"). Not signed off by the user
# yet (MAP_ONBOARDING.md).
TRUTH_SPAWNS = [(28, 12), (38, 16), (19, 17), (46, 23), (12, 26), (47, 33), (13, 36), (40, 42),
                (21, 43), (31, 47)]


def test_the_first_pass_fits_truth_and_finds_its_ten_spawns():
    ml = _tool()
    img = cv2.imread(str(MAPS_DIR / "truth.png"))
    lat, clarity = ml.fit_lattice(img)
    assert lat.pitch == pytest.approx((15.747, 15.728), abs=0.02)
    assert clarity > 2.0
    side = json.loads((MAPS_DIR / "truth.json").read_text())
    assert side["pitch_px"] == pytest.approx(lat.pitch, abs=1e-3)
    assert side["origin_px"] == pytest.approx(lat.origin, abs=1e-3)
    assert ml.find_spawns(img, lat) == TRUTH_SPAWNS


def test_the_truth_label_passes_the_checks_with_its_spawns_and_its_half_turn_symmetry():
    known = KnownMap.load("truth")
    assert _cells(known.spawns) == sorted(TRUTH_SPAWNS)
    chars = read_chars(MAPS_DIR / "truth.csv")
    turned = chars[::-1, ::-1]
    assert [(int(c), int(r)) for r, c in np.argwhere(chars != turned)] == []


# Doom Shroom (2026-11-10). As with Truth's, these came from `find_spawns`, and each ring was then
# centred by eye within 0.06 of a tile centre (fractional parts .44-.56). The user's clip starts on
# (28,38). Unlike Truth, the map has no symmetry to check the label against. Bounce, heal and
# teleporter pads and speed strips are `.`, and fences are `#` (both the user's rules; the fences
# were `.` until later the same day). Not signed off by the user yet (MAP_ONBOARDING.md).
DOOM_SHROOM_SPAWNS = [(33, 2), (12, 8), (49, 12), (8, 19), (30, 22), (49, 33), (28, 38), (6, 40),
                      (53, 53), (26, 55)]


def test_the_first_pass_fits_doom_shroom_and_finds_its_ten_spawns():
    ml = _tool()
    img = cv2.imread(str(MAPS_DIR / "doom_shroom.png"))
    lat, clarity = ml.fit_lattice(img)
    assert lat.pitch == pytest.approx((15.751, 15.723), abs=0.02)
    assert clarity > 2.0
    side = json.loads((MAPS_DIR / "doom_shroom.json").read_text())
    assert side["pitch_px"] == pytest.approx(lat.pitch, abs=1e-3)
    assert side["origin_px"] == pytest.approx(lat.origin, abs=1e-3)
    assert ml.find_spawns(img, lat) == DOOM_SHROOM_SPAWNS


def test_the_doom_shroom_label_passes_the_checks_with_its_spawns():
    known = KnownMap.load("doom_shroom")
    assert _cells(known.spawns) == sorted(DOOM_SHROOM_SPAWNS)


# Rockwall Brawl (2026-11-11). As with Doom Shroom's, these came from `find_spawns`, and each ring
# was then centred by eye within 0.05 of a tile centre (fractional parts .45-.51). The user's clip
# starts on (37,29). The navy floor checker is faint, so the fit is less clear than the other maps'
# (1.6 against their 2 and more); the rings agree with it. The map is only partly symmetric (its
# bush borders differ corner to corner), so every cell was read off the image. The destructible
# cones are `#` (the user's rule). Signed off by the user on 2026-11-11 with no changes
# (MAP_ONBOARDING.md).
ROCKWALL_BRAWL_SPAWNS = [(23, 2), (37, 2), (2, 22), (57, 22), (23, 29), (37, 29), (2, 36),
                         (57, 36), (23, 57), (37, 57)]


def test_the_first_pass_fits_rockwall_brawl_and_finds_its_ten_spawns():
    ml = _tool()
    img = cv2.imread(str(MAPS_DIR / "rockwall_brawl.png"))
    lat, clarity = ml.fit_lattice(img)
    assert lat.pitch == pytest.approx((15.746, 15.733), abs=0.02)
    assert clarity > 1.5
    side = json.loads((MAPS_DIR / "rockwall_brawl.json").read_text())
    assert side["pitch_px"] == pytest.approx(lat.pitch, abs=1e-3)
    assert side["origin_px"] == pytest.approx(lat.origin, abs=1e-3)
    assert ml.find_spawns(img, lat) == ROCKWALL_BRAWL_SPAWNS


def test_the_rockwall_brawl_label_passes_the_checks_with_its_spawns():
    known = KnownMap.load("rockwall_brawl")
    assert _cells(known.spawns) == sorted(ROCKWALL_BRAWL_SPAWNS)


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


def _click(lab, cell, ax=None):
    from matplotlib.backend_bases import MouseEvent
    x, y = _display(lab, cell, ax)
    MouseEvent("button_press_event", lab.fig.canvas, x, y, button=1)._process()
    MouseEvent("button_release_event", lab.fig.canvas, x, y, button=1)._process()


def _move(lab, cell, ax=None):
    from matplotlib.backend_bases import MouseEvent
    x, y = _display(lab, cell, ax)
    MouseEvent("motion_notify_event", lab.fig.canvas, x, y)._process()


def _display(lab, cell, ax=None):
    """Display pixels of `cell`'s centre on panel `ax` (the image panel by default)."""
    col, row = cell
    (px, py), (ox, oy) = lab.lat.pitch, lab.lat.origin
    return (ax or lab.ax).transData.transform((ox + (col + 0.5) * px, oy + (row + 0.5) * py))


def _pixel(lab, cell, ax) -> tuple[int, int, int]:
    """The colour the window shows at `cell`'s centre on panel `ax`, after a full draw: what the
    user sees, not what the tool's arrays say."""
    lab.fig.canvas.draw()
    rgba = np.asarray(lab.fig.canvas.buffer_rgba())
    x, y = _display(lab, cell, ax)
    return tuple(int(v) for v in rgba[rgba.shape[0] - 1 - int(y), int(x), :3])


def test_a_new_map_opens_on_the_first_pass_and_saves_what_is_painted(monkeypatch, tmp_path,
                                                                      capsys):
    lab = _run_tool(monkeypatch, tmp_path, ["dp", "--image", str(MAPS_DIR / "dark_passage.png")])
    assert (tmp_path / "dp.png").read_bytes() == (MAPS_DIR / "dark_passage.png").read_bytes()
    assert int((lab.chars == "S").sum()) == 10
    assert set(np.unique(lab.chars)) == {"?", "S"}
    assert lab.mode == "none"

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


def _halves(lab):
    """Two made-up clusters, the left and right halves, so what a fill takes is known."""
    lab.clusters = np.zeros((60, 60), np.int32)
    lab.clusters[:, :30] = 1


def test_the_label_panel_shows_every_change_at_once(monkeypatch, tmp_path):
    """2026-11-03, Crystal Eye Castle: the first pass opened on the cluster colours, which no fill
    changes, so naming 2043 cells floor looked like nothing happened. The label panel draws the
    label itself, opaque, whatever the image panel shows; this reads the window's pixels."""
    lab = _run_tool(monkeypatch, tmp_path, ["dp", "--image", str(MAPS_DIR / "dark_passage.png")])
    _halves(lab)
    # The legend's swatches as drawn: the key the user reads each cell's colour against.
    swatch = {char: tuple(round(255 * v) for v in patch.get_facecolor()[:3])
              for char, patch in zip(_tool().KEYS.values(), lab.legend.get_patches())}
    assert len(set(swatch.values())) == len(swatch), "two classes share a colour"
    cell, other = (20, 40), (45, 40)

    def shows(char, at=cell):
        got = _pixel(lab, at, lab.ax_label)
        return max(abs(a - b) for a, b in zip(got, swatch[char])) <= 1

    assert shows("?") and shows("?", other)
    _key(lab, "1")
    _key(lab, "c", cell)
    assert shows("."), "a cluster fill did not show, or not in its legend colour"
    assert shows("?", other), "the fill showed past its cluster"

    _key(lab, "2")
    _click(lab, cell, lab.ax_label)          # either panel takes paint
    assert shows("#")
    _key(lab, "u")
    assert shows(".")

    for mode in ("clusters", "label", "none"):
        _key(lab, "g")
        assert lab.mode == mode
        assert shows("."), f"the label panel changed with the image panel's overlay ({mode})"


def _outline_lines(lab) -> list[tuple[list[int], list[int]]]:
    """Per outline collection (black and yellow, on each panel): the lattice columns its upright
    edges sit on, and the rows its level edges sit on."""
    (px, py), (ox, oy) = lab.lat.pitch, lab.lat.origin
    out = []
    for outline in lab.outlines:
        segs = np.asarray(outline.get_segments()).reshape(-1, 2, 2)
        upright = np.isclose(segs[:, 0, 0], segs[:, 1, 0])
        cols = np.unique(np.round((segs[upright, 0, 0] - ox) / px).astype(int))
        rows = np.unique(np.round((segs[~upright, 0, 1] - oy) / py).astype(int))
        out.append((cols.tolist(), rows.tolist()))
    return out


def test_hovering_names_the_cell_and_outlines_the_cluster_c_would_fill(monkeypatch, tmp_path):
    lab = _run_tool(monkeypatch, tmp_path, ["dp", "--image", str(MAPS_DIR / "dark_passage.png")])
    _halves(lab)
    (px, py), (ox, oy) = lab.lat.pitch, lab.lat.origin
    for cell, ax, cid, edges in [((20, 40), lab.ax_label, 1, [0, 30]),
                                 ((45, 40), lab.ax, 0, [30, 60])]:    # either panel
        _move(lab, cell, ax)
        assert f"(col {cell[0]}, row {cell[1]}) UNLABELLED" in lab.status.get_text()
        assert f"cluster {cid}," in lab.status.get_text()
        assert _outline_lines(lab) == [(edges, [0, 60])] * 4, "not that half's edge"
        for box in lab.cursors:
            assert box.get_visible()
            assert np.allclose(box.get_xy(), (ox + cell[0] * px, oy + cell[1] * py))


def test_closing_the_window_keeps_unsaved_edits(monkeypatch, tmp_path):
    """2026-11-03: the user's first two sessions on Crystal Eye Castle ended at the window's X,
    and nothing was written."""
    from matplotlib.backend_bases import CloseEvent
    lab = _run_tool(monkeypatch, tmp_path, ["dp", "--image", str(MAPS_DIR / "dark_passage.png")])
    CloseEvent("close_event", lab.fig.canvas)._process()
    assert not (tmp_path / "dp.csv").exists(), "closing with no edits wrote a label"
    _key(lab, "2")
    _click(lab, (0, 8))
    CloseEvent("close_event", lab.fig.canvas)._process()
    saved = read_chars(tmp_path / "dp.csv")
    assert saved[8, 0] == "#" and (saved == lab.chars).all()

    review = _run_tool(monkeypatch, tmp_path, ["dp", "--review"])
    _key(review, "1")
    _click(review, (0, 8))
    CloseEvent("close_event", review.fig.canvas)._process()
    assert (read_chars(tmp_path / "dp.csv") == saved).all(), "--review wrote"


def test_marked_cells_are_ringed_and_n_centres_each_in_turn(monkeypatch, tmp_path):
    lab = _run_tool(monkeypatch, tmp_path, ["dp", "--image", str(MAPS_DIR / "dark_passage.png"),
                                            "--mark", "20,13", "(5,6)"])
    assert lab.marks == [(20, 13), (5, 6)]
    for want in [(20, 13), (5, 6), (20, 13)]:
        _key(lab, "n")
        x, y = np.mean(lab.ax.get_xlim()), np.mean(lab.ax.get_ylim())
        assert lab.lat.cell_of(x, y) == want
        assert lab.ax_label.get_xlim() == lab.ax.get_xlim()
    _move(lab, (20, 13))
    assert "[marked]" in lab.status.get_text()
    _key(lab, "h")
    assert (lab.ax.get_xlim(), lab.ax.get_ylim()) == lab.full
    for bad in ("20", "20,13,1", "60,0", "a,b"):
        with pytest.raises(SystemExit, match="--mark"):
            _run_tool(monkeypatch, tmp_path, ["dp", "--mark", bad])


def test_the_scroll_wheel_zooms_both_panels_about_the_cursor(monkeypatch, tmp_path):
    from matplotlib.backend_bases import MouseEvent
    lab = _run_tool(monkeypatch, tmp_path, ["dp", "--image", str(MAPS_DIR / "dark_passage.png")])
    x, y = _display(lab, (20, 40), lab.ax_label)
    under = lab.ax_label.transData.inverted().transform((x, y))
    MouseEvent("scroll_event", lab.fig.canvas, x, y, button="up", step=1)._process()
    assert np.diff(lab.ax.get_xlim())[0] == pytest.approx(lab.full[0][1] / 1.25)
    assert lab.ax_label.get_xlim() == lab.ax.get_xlim()
    lab.fig.canvas.draw()
    assert np.allclose(lab.ax_label.transData.inverted().transform((x, y)), under, atol=1.0)
    for _ in range(3):                       # out stops at the whole image
        MouseEvent("scroll_event", lab.fig.canvas, x, y, button="down", step=-1)._process()
    assert (lab.ax.get_xlim(), lab.ax.get_ylim()) == lab.full
