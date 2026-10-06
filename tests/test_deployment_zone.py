"""`brawl_deployment.perception.zone` -- the zone group's supplier.

The interesting cases are the ones where the estimator has to be *wrong in a chosen direction*:
standing in gas, standing off the canvas, standing next to gas it has never seen. Every one of
those is a documented one-sided bias rather than a bug, and the tests say which side.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from brawl_deployment.perception.grid import GasMap
from brawl_deployment.perception.zone import PINNED_NEXT_SHRINK_IN, ZoneEstimator
from brawl_sim.config import load_config
from brawl_vision.terrain.odometry import OdometryResult
from brawl_vision.terrain.zone import ZoneMask

CFG = load_config("configs/default.yaml")
HORIZON = CFG.zone_margin_horizon_tiles          # 10.0 tiles


def _gas(size: int = 128) -> GasMap:
    return GasMap(size, size)


def _mark(gas: GasMap, x0, x1, y0, y1) -> None:
    """Mark a WORLD-tile rectangle as gassed, inclusive bounds."""
    ox, oy = gas.origin
    gas.gassed[y0 - oy:y1 - oy + 1, x0 - ox:x1 - ox + 1] = True


def _est() -> ZoneEstimator:
    return ZoneEstimator(CFG)


def _deposit(gas: GasMap, *, segment: int, gassed: bool) -> int:
    """One frame through `GasMap.update`'s own path, on the producers' real result types: a 21x13
    frame, fully observed and all gas or none, with the camera at world (0, 0) in `segment`."""
    cols, rows = 21, 13
    cells = np.full((rows, cols), gassed, bool)
    zone = ZoneMask(pixels=np.zeros((1, 1), bool), cells=cells, cell_fraction=cells.astype(float),
                    observed=np.ones((rows, cols), bool), coverage=float(gassed))
    odo = OdometryResult(delta_tiles=(0.0, 0.0), position_tiles=(0.0, 0.0), response=1.0,
                         agreement_tiles=0.0, inlier_ratio=1.0, n_windows=11, status="ok",
                         segment=segment)
    return gas.update(zone, SimpleNamespace(size_tiles=(cols, rows), origin_tile=(-10, -6)), odo)


# ---------------------------------------------------------------------------------------------
# hero_margin: four signed distances.
# ---------------------------------------------------------------------------------------------

def test_a_clean_map_reports_the_horizon_on_every_side():
    """No gas seen yet is the opening state of every match. Saturating is right: the sensor has
    looked as far as it can and found nothing, which is a different statement from "the edge is
    here" and must not be reported as zero."""
    assert _est().hero_margin(_gas(), (0.0, 0.0)) == (HORIZON,) * 4


def test_gas_to_the_west_shortens_only_the_west_margin():
    """`hero_margin` is `(x - lo.x, hi.x - x, y - lo.y, hi.y - y)`, so the first component is the
    distance to the safe rect's LEFT edge and nothing else may move."""
    gas = _gas()
    _mark(gas, -40, -5, -40, 40)          # a wall of gas 5 tiles west of the origin
    margin = _est().hero_margin(gas, (0.0, 0.0))
    assert margin[0] == pytest.approx(5.0)
    assert margin[1:] == (HORIZON, HORIZON, HORIZON)


@pytest.mark.parametrize("ray,rect,expected_index", [
    ("west", (-40, -5, -40, 40), 0),
    ("east", (5, 40, -40, 40), 1),
    ("north", (-40, 40, -40, -5), 2),     # y increases DOWNWARD, so -y is up the screen
    ("south", (-40, 40, 5, 40), 3),
])
def test_each_ray_lands_in_its_own_component(ray, rect, expected_index):
    """Pins the component order against the schema. Getting this wrong swaps two columns the
    policy reads, which is invisible in any single frame and looks like a confused agent."""
    gas = _gas()
    _mark(gas, *rect)
    margin = _est().hero_margin(gas, (0.0, 0.0))
    assert margin[expected_index] == pytest.approx(5.0)
    assert sum(1 for m in margin if m != HORIZON) == 1


def test_standing_in_gas_makes_every_margin_negative():
    """The hero is outside the safe rect on all four sides, so all four margins are negative and
    their magnitude is the distance to clear ground -- not to more gas."""
    gas = _gas()
    _mark(gas, -3, 3, -3, 3)              # a blob with the hero inside it
    margin = _est().hero_margin(gas, (0.0, 0.0))
    assert all(m < 0 for m in margin)
    assert margin == pytest.approx((-4.0, -4.0, -4.0, -4.0))


def test_the_clamp_is_symmetric_because_the_blindness_is():
    """Deep in a large gas field the nearest clear ground is past the sensor, exactly as distant
    gas is when standing safe. A one-sided clamp would model a sensor this is not (9.16)."""
    gas = _gas()
    _mark(gas, -40, 40, -40, 40)
    assert _est().hero_margin(gas, (0.0, 0.0)) == (-HORIZON,) * 4


def test_gas_beyond_the_horizon_is_not_reported():
    """The saturation the deployed spec's `hero_margin_local` sibling was created to match. Gas
    at 15 tiles is outside what the camera reaches, so reporting 15 would be training the policy
    on a sensor range the live loop does not have."""
    gas = _gas()
    _mark(gas, -40, -15, -40, 40)
    assert _est().hero_margin(gas, (0.0, 0.0))[0] == HORIZON


def test_the_values_are_hero_margin_local_even_though_the_column_is_hero_margin():
    """The deployed spec names `zone.hero_margin`, which the sim emits UNCLAMPED. This supplies the
    clamped shape, because a bounded sensor is what deployment has. Priced at -1.9 pp (9.15's
    "clamped at 10 tiles" row) and fixed properly by `agent_obs_deploy2.yaml`, which names
    `hero_margin_local`. Pinned so the substitution stays visible rather than looking like a bug."""
    import yaml
    spec = yaml.safe_load(open("configs/agent_obs_deploy.yaml").read())
    group = next(g for g in spec["groups"] if g["name"] == "zone")
    assert "zone.hero_margin" in group["fields"]              # the unclamped column...
    assert "zone.hero_margin_local" not in group["fields"]

    gas = _gas()
    _mark(gas, -40, -25, -40, 40)                             # gas at 25 tiles: a real distance
    assert _est().hero_margin(gas, (0.0, 0.0))[0] == HORIZON  # ...fed a clamped value


def test_the_horizon_comes_from_the_config_not_from_this_module():
    """9.16 is emphatic: two copies of the number in two packages is how the sim and the deployed
    estimator drift apart, and the drift is silent -- the spec keeps naming the right column."""
    import dataclasses
    narrow = dataclasses.replace(CFG, zone_margin_horizon_tiles=4.0)
    gas = _gas()
    _mark(gas, -40, -6, -40, 40)                              # gas at 6 tiles
    assert _est().hero_margin(gas, (0.0, 0.0))[0] == pytest.approx(6.0)
    assert ZoneEstimator(narrow).hero_margin(gas, (0.0, 0.0))[0] == 4.0


def test_the_scan_follows_the_hero_across_the_world_frame():
    """`hero_pos` is world tiles, not camera-relative. Feeding the camera-relative value would put
    the scan origin wherever the match started and read as a plausible margin the whole way."""
    gas = _gas()
    _mark(gas, 20, 40, -40, 40)                               # gas from x=20 east
    assert _est().hero_margin(gas, (15.0, 0.0))[1] == pytest.approx(5.0)
    assert _est().hero_margin(gas, (0.0, 0.0))[1] == HORIZON  # 20 tiles away: past the horizon


def test_a_hero_off_the_canvas_reports_full_margins_not_zeros():
    """Zeros would claim the hero is standing exactly on all four edges at once -- a specific and
    false statement. Off the canvas is unseen, and unseen reads optimistic here as everywhere."""
    assert _est().hero_margin(_gas(), (10_000.0, 10_000.0)) == (HORIZON,) * 4


def test_unseen_cells_read_as_safe_which_is_the_documented_bias():
    """`GasMap` is sticky and never guesses: a cell it has not seen stays clear forever. So the
    estimate is one-sided OPTIMISTIC, in exactly the places the hero walked away from. Pinned here
    so it stays a known residual rather than becoming a surprise (9.16)."""
    gas = _gas()
    _mark(gas, -40, -5, -40, 40)
    gas.seen[:] = False                   # nothing was ever actually observed
    assert _est().hero_margin(gas, (0.0, 0.0))[0] == pytest.approx(5.0)
    # ...and with no deposit at all, the same position reads perfectly safe.
    assert _est().hero_margin(_gas(), (0.0, 0.0))[0] == HORIZON


# ---------------------------------------------------------------------------------------------
# The other three fields.
# ---------------------------------------------------------------------------------------------

def test_active_is_false_until_gas_is_seen_and_then_latches():
    """Matches the sim, where `zone.active` never turns back off within an episode."""
    gas = _gas()
    est = _est()
    assert est.active(gas) is False
    _mark(gas, 0, 0, 0, 0)
    assert est.active(gas) is True


def test_active_survives_a_new_segment_and_only_a_new_match_clears_it():
    """The sim's `zone_seen` is cleared by an episode reset and by nothing else. `GasMap` is sticky
    only within a segment: a cut or a lattice re-lock empties it, so read on its own the flag would
    fall back to 0 mid-match whenever that happens with the gas out of view. The estimator's latch
    holds it, and `reset`, which the loop calls at a match start and nowhere else, clears it."""
    gas, est = _gas(), _est()
    assert _deposit(gas, segment=0, gassed=True) > 0
    assert est.active(gas) is True
    _deposit(gas, segment=1, gassed=False)           # a new segment, with no gas in view
    assert not gas.gassed.any(), "the map itself does not outlive the segment"
    assert est.active(gas) is True
    est.reset()                                      # the next match
    assert est.active(gas) is False
    _deposit(gas, segment=1, gassed=True)
    assert est.active(gas) is True, "and that match's first gas latches it again"


def test_safe_area_frac_is_one_on_a_clean_map_and_falls_with_observed_gas():
    gas = _gas()
    est = _est()
    assert est.safe_area_frac(gas) == pytest.approx(1.0)
    _mark(gas, -30, -1, -30, 29)                              # exactly half the 60x60 map
    assert est.safe_area_frac(gas) == pytest.approx(0.5, abs=0.02)


def test_safe_area_frac_only_counts_gas_inside_the_map_extent():
    """The canvas is 128x128 and the map is 60x60. Gas deposited in the margin is real gas the
    hero walked past, but it is not map area, and counting it would drive the fraction negative."""
    gas = _gas()
    _mark(gas, -60, -31, -60, 60)                             # entirely west of the map
    assert _est().safe_area_frac(gas) == pytest.approx(1.0)


def test_safe_area_frac_stays_in_range_when_the_whole_canvas_is_gassed():
    gas = _gas()
    gas.gassed[:] = True
    assert _est().safe_area_frac(gas) == 0.0


def test_next_shrink_in_is_pinned_at_zero_and_says_so():
    """A deliberate exception to "never feed a constant", because it is a PRICED one: zeroing cost
    1.5 pp and feeding a plausible wrong countdown cost 4.4 pp (9.14). Zero is also what an
    overdue or disabled schedule already looks like in training -- `clamp(zone_next_t - time,
    min=0)` -- so it is a value the column's distribution contains for the right reason."""
    assert PINNED_NEXT_SHRINK_IN == 0.0
    assert _est().estimate(_gas(), (0.0, 0.0))["next_shrink_in"] == 0.0


# ---------------------------------------------------------------------------------------------
# The whole group.
# ---------------------------------------------------------------------------------------------

def _zone_fields(spec_path) -> set[str]:
    import yaml
    spec = yaml.safe_load(open(spec_path).read())
    group = next(g for g in spec["groups"] if g["name"] == "zone")
    return {f.split(".", 1)[1] for f in group["fields"]}


# Globbed rather than listed, so the next deploy spec is covered the day it is added. A listed tuple
# is how agent_obs_deploy2/3 went unchecked here: the first live run on deploy3 raised
# "no value supplied for 'zone.hero_margin_local'" on its first decision.
DEPLOY_SPECS = sorted(str(p).replace("\\", "/") for p in Path("configs").glob("agent_obs_deploy*.yaml"))


def test_every_deploy_spec_is_covered_here():
    assert len(DEPLOY_SPECS) >= 3, DEPLOY_SPECS


@pytest.mark.parametrize("spec_path", DEPLOY_SPECS)
def test_estimate_supplies_every_zone_field_each_deploy_spec_names(spec_path):
    """The contract with `assemble`: every zone field the loaded spec names, minus the `zone.`
    prefix. A missing key raises in `_require` -- on the first decision of a live match, which is
    the worst place to learn it, so it is checked here for every spec that could be deployed."""
    missing = _zone_fields(spec_path) - set(_est().estimate(_gas(), (0.0, 0.0)))
    assert not missing, f"{spec_path} names zone fields the estimator never supplies: {missing}"


def test_estimate_supplies_nothing_no_deploy_spec_reads():
    """The other direction. An extra key is silently ignored by `_put_zone`, so a field every spec
    has since dropped would keep being computed -- and, for a pinned one, keep looking supplied."""
    wanted = set().union(*(_zone_fields(p) for p in DEPLOY_SPECS))
    assert set(_est().estimate(_gas(), (0.0, 0.0))) == wanted


def test_both_margin_names_are_one_scan():
    """`hero_margin` borrows `hero_margin_local`'s values on the first deploy spec; the two keys
    must never be able to disagree about the same gas."""
    gas = _gas()
    _mark(gas, -40, -5, -40, 40)
    zone = _est().estimate(gas, (0.0, 0.0))
    assert zone["hero_margin_local"] == zone["hero_margin"]
    assert zone["hero_margin_local"][0] == pytest.approx(5.0)


def test_the_margins_hold_no_state_between_calls():
    """`GasMap.reset` on a segment change must not need a matching reset here. Holding margins of
    its own would leave them pointing at a world frame that moved, which reads as the gas
    teleporting rather than as a reset. `active` is the one field with state: a latch with no
    position in it, which is meant to outlive the segment (the latch tests above)."""
    est = _est()
    gas = _gas()
    _mark(gas, -40, -5, -40, 40)
    assert est.hero_margin(gas, (0.0, 0.0))[0] == pytest.approx(5.0)
    gas.reset(segment=1)
    assert est.hero_margin(gas, (0.0, 0.0)) == (HORIZON,) * 4


@pytest.mark.parametrize("spec_path", DEPLOY_SPECS)
def test_the_group_round_trips_through_the_assembler(spec_path):
    """The real contract: what this produces has to satisfy the assembler, in its units, without
    the caller reshaping anything. `_require` raises on a missing field, so this also proves the
    estimator covers the spec -- but it would NOT catch a margin of the wrong length, which is why
    the shape is asserted too."""
    from brawl_deployment.perception.assemble import ObservationAssembler
    from brawl_sim.core import obs_select

    asm = ObservationAssembler(obs_select.load_agent_spec(spec_path, CFG), CFG)
    gas = _gas()
    _mark(gas, -40, -5, -40, 40)
    zone = _est().estimate(gas, (0.0, 0.0))

    full = {}
    asm._put_zone(full, zone)
    assert set(full["zone"]) == _zone_fields(spec_path)
    margin = next(k for k in full["zone"] if k.startswith("hero_margin"))
    assert tuple(full["zone"][margin].shape) == (1, 4)
    assert full["zone"][margin][0, 0].item() == pytest.approx(5.0)
    if "safe_area_frac" in full["zone"]:
        assert tuple(full["zone"]["safe_area_frac"].shape) == (1,)


def test_the_assembler_refuses_a_group_this_estimator_would_never_produce():
    """The other direction: `_require` is the interlock that keeps a supplier failure from being
    quietly filled with zero, so it has to actually fire on a dropped field."""
    from brawl_deployment.perception.assemble import ObservationAssembler
    from brawl_sim.core import obs_select

    asm = ObservationAssembler(obs_select.load_agent_spec("configs/agent_obs_deploy.yaml", CFG),
                               CFG)
    zone = _est().estimate(_gas(), (0.0, 0.0))
    del zone["safe_area_frac"]
    with pytest.raises(ValueError, match="zone.safe_area_frac"):
        asm._put_zone({}, zone)
