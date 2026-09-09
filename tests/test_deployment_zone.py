"""`brawl_deployment.perception.zone` -- the zone group's supplier.

The interesting cases are the ones where the estimator has to be *wrong in a chosen direction*:
standing in gas, standing off the canvas, standing next to gas it has never seen. Every one of
those is a documented one-sided bias rather than a bug, and the tests say which side.
"""
import pytest

from brawl_deployment.perception.grid import GasMap
from brawl_deployment.perception.zone import PINNED_NEXT_SHRINK_IN, ZoneEstimator
from brawl_sim.config import load_config

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
    """Matches the sim, where the zone never turns back off. Latching is free -- `GasMap` is
    sticky -- so this needs no state of its own."""
    gas = _gas()
    est = _est()
    assert est.active(gas) is False
    _mark(gas, 0, 0, 0, 0)
    assert est.active(gas) is True


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

def test_estimate_supplies_exactly_the_deploy_specs_zone_fields():
    """The contract with `assemble`: the keys here must be the spec's zone field names, minus the
    `zone.` prefix. A missing key raises in `_require`; an extra one is silently ignored, which is
    the direction worth a test."""
    import yaml
    spec = yaml.safe_load(open("configs/agent_obs_deploy.yaml").read())
    group = next(g for g in spec["groups"] if g["name"] == "zone")
    wanted = {f.split(".", 1)[1] for f in group["fields"]}
    assert set(_est().estimate(_gas(), (0.0, 0.0))) == wanted


def test_the_estimator_holds_no_state_between_calls():
    """`GasMap.reset` on a segment change must not need a matching reset here. Holding margins of
    its own would leave them pointing at a world frame that moved, which reads as the gas
    teleporting rather than as a reset."""
    est = _est()
    gas = _gas()
    _mark(gas, -40, -5, -40, 40)
    assert est.hero_margin(gas, (0.0, 0.0))[0] == pytest.approx(5.0)
    gas.reset(segment=1)
    assert est.hero_margin(gas, (0.0, 0.0)) == (HORIZON,) * 4


def test_the_group_round_trips_through_the_assembler():
    """The real contract: what this produces has to satisfy the assembler, in its units, without
    the caller reshaping anything. `_require` raises on a missing field, so this also proves the
    estimator covers the spec -- but it would NOT catch a `hero_margin` of the wrong length, which
    is why the shape is asserted too."""
    from brawl_deployment.perception.assemble import ObservationAssembler
    from brawl_sim.core import obs_select

    asm = ObservationAssembler(obs_select.load_agent_spec("configs/agent_obs_deploy.yaml", CFG),
                               CFG)
    gas = _gas()
    _mark(gas, -40, -5, -40, 40)
    zone = _est().estimate(gas, (0.0, 0.0))

    full = {}
    asm._put_zone(full, zone)
    assert set(full["zone"]) == set(zone)
    assert tuple(full["zone"]["hero_margin"].shape) == (1, 4)
    assert tuple(full["zone"]["safe_area_frac"].shape) == (1,)
    assert full["zone"]["hero_margin"][0, 0].item() == pytest.approx(5.0)


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
