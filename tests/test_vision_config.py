"""VisionConfig loading and validation, plus the Phase 0 packaging invariants. See
Terrain_Perception_Build_Plan.md Phase 0.

Covers both halves the way brawl_sim splits them: the loader machinery against inline fixtures
(tests/test_config.py's role) and the real configs/vision.yaml end to end
(tests/test_configs_files.py's role). One file, because VisionConfig is a twelfth the size of
EnvConfig and does not warrant two.
"""
import tomllib
from pathlib import Path

import pytest
import yaml

from brawl_vision.config import (
    DEFAULT_CONFIG_PATH,
    VisionConfig,
    load_vision_config,
    validate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _cfg(**overrides) -> VisionConfig:
    return VisionConfig(**overrides)


# ---------------------------------------------------------------------------
# the real file
# ---------------------------------------------------------------------------

def test_shipped_vision_yaml_loads_and_validates():
    validate(load_vision_config(DEFAULT_CONFIG_PATH))


def test_shipped_vision_yaml_sets_every_declared_field():
    """configs/vision.yaml is documentation as much as configuration -- each key carries the
    comment saying whether it is measured or a placeholder. A field that exists in VisionConfig
    but is missing from the file still WORKS (the dataclass default covers it) and is exactly
    the kind of silent omission that leaves a threshold undocumented."""
    from brawl_vision.config import _VISION_CONFIG_FIELDS, _dget

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    missing = [dotted for dotted, _, _ in _VISION_CONFIG_FIELDS
               if _dget(raw, dotted, default=None) is None]
    assert not missing, f"configs/vision.yaml does not mention: {missing}"


def test_every_remaining_placeholder_is_still_obviously_fake():
    """Placeholders in this file ship as values that CANNOT be mistaken for measurements, because
    a plausible-looking guess half-works and costs a day to find. Each is paired with a test that
    fails the moment it is replaced -- the prompt to record what it was measured from.

    The zone threshold used to be here, shipping as the full HSV cube ("everything is gas"). Phase
    G measured it across three maps and the guard moved to
    `tests/test_vision_zone.py::test_the_shipped_threshold_is_no_longer_the_placeholder`, which
    pins both the numbers and their provenance. What remains is the occupancy pair.
    """
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    section = DEFAULT_CONFIG_PATH.read_text().split("occupancy:")[1]
    assert "PLACEHOLDER" in section, (
        "occupancy.min_votes / lock_ratio look measured now -- record what they were measured "
        "against in configs/vision.yaml, then update this test."
    )
    assert raw["occupancy"]["min_votes"] == 5 and raw["occupancy"]["lock_ratio"] == 0.8


# ---------------------------------------------------------------------------
# loader machinery
# ---------------------------------------------------------------------------

def test_absent_key_keeps_the_dataclass_default(tmp_path):
    """The failure mode brawl_sim.config.load_config documents: it shipped with an unreachable
    fallback branch, so every config file was silently required to spell out all ~40 fields and
    adding one broke every previously-valid YAML."""
    path = tmp_path / "sparse.yaml"
    path.write_text("occupancy:\n  min_votes: 9\n")

    cfg = load_vision_config(path)
    assert cfg.occupancy_min_votes == 9           # from the file
    assert cfg.occupancy_lock_ratio == 0.8        # untouched sibling key
    assert cfg.capture_monitor == 1               # untouched section entirely


def test_empty_file_yields_all_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_vision_config(path) == VisionConfig()


def test_overrides_deep_merge_rather_than_replace_a_section(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text("occupancy:\n  min_votes: 5\n  lock_ratio: 0.8\n")

    cfg = load_vision_config(path, overrides={"occupancy": {"min_votes": 12}})
    assert cfg.occupancy_min_votes == 12
    assert cfg.occupancy_lock_ratio == 0.8, "sibling key was clobbered -- merge is not deep"


def test_hsv_lists_coerce_to_int_tuples(tmp_path):
    path = tmp_path / "hsv.yaml"
    path.write_text("zone:\n  hsv_low: [35, 40, 40]\n  hsv_high: [85, 255, 255]\n")

    cfg = load_vision_config(path)
    assert cfg.zone_hsv_low == (35, 40, 40)
    assert isinstance(cfg.zone_hsv_low, tuple)
    assert all(isinstance(v, int) for v in cfg.zone_hsv_low)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_defaults_validate():
    validate(VisionConfig())


@pytest.mark.parametrize("field,value", [
    ("odometry_min_response", -0.1),
    ("odometry_min_response", 1.5),
    ("odometry_max_shift_tiles", 0.0),
    ("zone_min_cell_fraction", 0.0),
    ("zone_min_cell_fraction", 1.5),
    ("occupancy_grid_h", 0),
    ("occupancy_min_votes", 0),
    ("classifier_device", "tpu"),
])
def test_validate_rejects_out_of_range(field, value):
    with pytest.raises(ValueError):
        validate(_cfg(**{field: value}))


def test_validate_rejects_a_0_359_hue():
    """The single easiest Phase G mistake: OpenCV's 8-bit H channel is 0-179, a HALF-degree
    wheel. A 0-359 hue produces an empty mask rather than an error, so it is checked here
    rather than left to a comment."""
    with pytest.raises(ValueError, match=r"hsv_low\[H\]"):
        validate(_cfg(zone_hsv_low=(200, 0, 0)))


def test_validate_allows_a_wrapping_hue_range():
    """Hue is a wheel: a range spanning red wraps past 179 back through 0, and is expressed as
    low > high. Rejecting that would make the most common gas/danger tint inexpressible."""
    validate(_cfg(zone_hsv_low=(170, 50, 50), zone_hsv_high=(10, 255, 255)))


def test_validate_rejects_inverted_saturation():
    """...but S and V do not wrap, so low > high there is a transcription error, not a range."""
    with pytest.raises(ValueError, match="does not wrap|do not wrap"):
        validate(_cfg(zone_hsv_low=(0, 200, 0), zone_hsv_high=(179, 50, 255)))


def test_validate_rejects_a_lock_ratio_an_even_split_would_clear():
    """Five terrain classes, so a 0.2 ratio is cleared by a dead-even vote split -- the first
    cell to reach min_votes would lock whatever it happened to see, permanently."""
    with pytest.raises(ValueError, match="five terrain"):
        validate(_cfg(occupancy_lock_ratio=0.2))


# ---------------------------------------------------------------------------
# Phase 0 packaging invariants
# ---------------------------------------------------------------------------

def test_brawl_vision_is_a_discoverable_package():
    """`include = ["brawl_sim*"]` excluded brawl_vision entirely -- an editable install did not
    expose it, and every import in this file would fail from outside the repo root."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "brawl_vision*" in include


def test_vision_pytest_marker_is_registered():
    """Fixture-dependent tests carry @pytest.mark.vision so they skip on a clone with no
    footage. An unregistered marker is a warning, not an error, so this is the thing that
    actually catches its removal."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    markers = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("vision:") for m in markers)


def test_simulator_does_not_import_the_vision_package():
    """The dependency between these two packages is STRICTLY ONE-WAY: `brawl_vision` may import
    `brawl_sim` (for `constants.Tile`, the shared terrain vocabulary), and `brawl_sim` must never
    import `brawl_vision`.

    This is what makes the test suites separable, and separability is worth real time: the full
    run is ~13 minutes and 816 of its 861 tests are the simulator. As long as this holds, a change
    confined to `brawl_vision/` cannot break a simulator test, so `pytest tests/test_vision_*.py`
    (4 seconds) is sufficient for it. The day someone adds a `brawl_vision` import to the sim,
    that reasoning silently becomes false -- so it is asserted rather than assumed.

    (Root files shared by both -- `pyproject.toml`, `.gitignore`, `configs/`, `scripts/` -- are a
    genuine exception and do warrant the full suite. Phase 0 edited `packages.find` and the pytest
    markers, which really can affect the simulator.)"""
    offenders = []
    for path in (REPO_ROOT / "brawl_sim").rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import brawl_vision", "from brawl_vision")):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
    assert not offenders, (
        "brawl_sim imports brawl_vision at " + ", ".join(offenders) +
        " -- that reverses the intended dependency and means vision changes can now break "
        "simulator tests. Either drop the import or update the test-selection rule in "
        "Terrain_Perception_Build_Plan.md Section 0."
    )


def test_game_footage_is_gitignored_but_ground_truth_is_not():
    """The Supercell-IP rule excludes captured frames and clips BY EXTENSION, not by directory,
    so the hand-authored ground truth living beside them (label CSVs, *.truth.json, *.grid.csv)
    stays in version control. Ignoring the directories instead would drop the annotations --
    the half that is actually yours, actually small, and actually what the fixtures assert
    against."""
    ignore = (REPO_ROOT / ".gitignore").read_text()
    assert "tests/fixtures/vision/**/*.mp4" in ignore
    assert "brawl_vision/data/frames/**/*.png" in ignore
    # No blanket directory exclusion that would swallow the tracked text alongside it.
    for line in ignore.splitlines():
        stripped = line.strip()
        assert stripped not in ("tests/fixtures/vision/", "brawl_vision/data/frames/"), (
            f"{stripped!r} excludes the whole directory, taking the ground truth with it"
        )
