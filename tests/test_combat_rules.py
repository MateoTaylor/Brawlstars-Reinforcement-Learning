"""bots/combat_rules.py -- the data-driven fire/aim rule that replaced the four archetype
modules in Step E1.

The point of this file is that it is almost entirely KIND-AGNOSTIC. Each test takes one kind,
rewrites a single field of its rule, and asserts the behavior follows the FIELD rather than the
Kind -- because the reason E1 exists is that a sixth brawler should be a configs/brawlers.yaml
block, and a rule table that only happens to work for the five shipped brawlers would not deliver
that. Per-brawler behavior (Brock holds LOS, Grom lobs over walls, Buzz needs its cone, Shelly
holds fire on crossers) is still covered end-to-end in tests/test_sniper.py, test_artillery.py,
test_melee.py and test_rifle.py.
"""
import math

import pytest
import torch

from tests.bot_fixtures import FakeBank, build_targeting, cfg_and_params, fresh_state, grid
from brawl_sim.bots import combat_rules
from brawl_sim.config import validate
from brawl_sim.constants import AimModel, Kind, Tile

# The victim is entity 0 and the shooter entity 1, on a clear east-west line.
SHOOTER, VICTIM = 1, 0
_VICTIM_POS = (5.0, 10.0)
_SHOOTER_POS = (11.0, 10.0)


def _setup(kind=Kind.BOT_SNIPER, wall=False, n_enemies=1, map_h=20, map_w=20,
           victim=_VICTIM_POS, shooter=_SHOOTER_POS, victim_vel=(0.0, 0.0)):
    """A shooter of `kind` at `shooter` with a stationary-or-moving victim at `victim`, no noise.

    Noise is zeroed everywhere so aim assertions can be EXACT rather than approximate -- the
    noise models get their own test below, where the distributions are what is being checked.
    """
    cfg, params, gen = cfg_and_params(n_enemies=n_enemies, map_h=map_h, map_w=map_w)
    params.aim_noise_std_rad.zero_()
    params.aim_noise_tiles.zero_()
    state = fresh_state(cfg, params, enemy_kind=kind)
    tiles = grid(map_h, map_w)
    if wall:
        # A full column between them: `vis` is bush-only so the victim is still TARGETED, which
        # is exactly the premise the LOS gate exists for.
        tiles[:, int((victim[0] + shooter[0]) / 2)] = Tile.WALL
    bank = FakeBank(tiles)
    state.ent_pos[0, VICTIM] = torch.tensor(victim)
    state.ent_pos[0, SHOOTER] = torch.tensor(shooter)
    state.ent_vel[0, VICTIM] = torch.tensor(victim_vel)
    # Face the victim, so the cone gate is satisfied for any kind that has one unless a test
    # deliberately turns the shooter away.
    state.ent_facing[0, SHOOTER] = math.atan2(victim[1] - shooter[1], victim[0] - shooter[0])
    return cfg, params, gen, state, bank


def _run(state, bank, params, cfg, gen):
    _vis, tgt = build_targeting(state, bank, params, cfg)
    return combat_rules.combat(state, tgt, bank, params, cfg, gen)


def _fires(state, bank, params, cfg, gen):
    return bool(_run(state, bank, params, cfg, gen)[0][0, SHOOTER])


# =================================================================================================
# fire_needs_los
# =================================================================================================

def test_fire_needs_los_is_what_decides_shooting_through_a_wall_not_the_kind():
    k = int(Kind.BOT_SNIPER)
    for needs_los, expected_through_wall in ((1, False), (0, True)):
        cfg, params, gen, state, bank = _setup(wall=True)
        params.fire_needs_los[:, k] = needs_los
        assert _fires(state, bank, params, cfg, gen) is expected_through_wall, needs_los

    # ...and with no wall in the way, the flag changes nothing.
    for needs_los in (0, 1):
        cfg, params, gen, state, bank = _setup(wall=False)
        params.fire_needs_los[:, k] = needs_los
        assert _fires(state, bank, params, cfg, gen) is True, needs_los


# =================================================================================================
# fire_range_fraction
# =================================================================================================

def test_fire_range_fraction_holds_fire_in_the_outer_band_of_attack_range():
    k = int(Kind.BOT_SNIPER)
    cfg, params, gen, _s, _b = _setup()
    attack_range = float(params.attack_range[0, k])
    fraction = 0.5
    inside = attack_range * fraction - 0.5      # inside the band it will shoot in
    outside = attack_range * fraction + 0.5     # past the fraction, still inside attack_range
    assert outside < attack_range               # precondition: fire_gate itself would allow it

    for dist, expected in ((inside, True), (outside, False)):
        cfg, params, gen, state, bank = _setup(
            shooter=(_VICTIM_POS[0] + dist, _VICTIM_POS[1]), map_h=30, map_w=30)
        params.fire_range_fraction[:, k] = fraction
        assert _fires(state, bank, params, cfg, gen) is expected, dist


def test_fire_range_fraction_of_zero_means_no_restriction():
    """0 is what a kind that never mentions the field resolves to, and it must be read as 1.0 --
    a literal 0 would mean `dist <= 0`, i.e. a bot that can never fire at all."""
    k = int(Kind.BOT_SNIPER)
    cfg, params, gen, _s, _b = _setup()
    attack_range = float(params.attack_range[0, k])
    just_inside = attack_range - 0.05

    cfg, params, gen, state, bank = _setup(
        shooter=(_VICTIM_POS[0] + just_inside, _VICTIM_POS[1]), map_h=30, map_w=30)
    params.fire_range_fraction[:, k] = 0.0
    assert _fires(state, bank, params, cfg, gen) is True


# =================================================================================================
# lateral hold
# =================================================================================================

def _lateral_setup(limit, hold_fraction, crossing_speed, dist_fraction):
    k = int(Kind.BOT_RIFLE)
    cfg, params, gen, _s, _b = _setup(kind=Kind.BOT_RIFLE)
    attack_range = float(params.attack_range[0, k])
    dist = attack_range * dist_fraction
    cfg, params, gen, state, bank = _setup(
        kind=Kind.BOT_RIFLE, map_h=30, map_w=30,
        shooter=(_VICTIM_POS[0] + dist, _VICTIM_POS[1]),
        victim_vel=(0.0, crossing_speed))  # purely lateral: the line between them is east-west
    params.fire_range_fraction[:, k] = 1.0
    params.fire_lateral_speed_limit[:, k] = limit
    params.fire_lateral_hold_range_fraction[:, k] = hold_fraction
    return cfg, params, gen, state, bank


def test_lateral_hold_fires_below_the_speed_limit_and_holds_above_it():
    for speed, expected in ((1.0, True), (9.0, False)):
        cfg, params, gen, state, bank = _lateral_setup(
            limit=2.0, hold_fraction=0.6, crossing_speed=speed, dist_fraction=0.8)
        assert _fires(state, bank, params, cfg, gen) is expected, speed


def test_lateral_hold_only_applies_beyond_its_range_fraction():
    """Close in, the fan is tight enough to land on a mover -- so the same crossing speed that
    holds fire at 0.8 of range must NOT hold it at 0.3."""
    for dist_fraction, expected in ((0.3, True), (0.8, False)):
        cfg, params, gen, state, bank = _lateral_setup(
            limit=2.0, hold_fraction=0.6, crossing_speed=9.0, dist_fraction=dist_fraction)
        assert _fires(state, bank, params, cfg, gen) is expected, dist_fraction


def test_lateral_speed_limit_of_zero_disables_the_hold_entirely():
    """The `limit > 0` term in combat_rules is load-bearing: without it an unset limit would be
    read as `lateral_speed > 0`, holding fire against ANY moving target for EVERY kind."""
    cfg, params, gen, state, bank = _lateral_setup(
        limit=0.0, hold_fraction=0.6, crossing_speed=50.0, dist_fraction=0.9)
    assert _fires(state, bank, params, cfg, gen) is True


def test_a_stationary_loot_box_never_triggers_the_lateral_hold():
    """Targeting reports zero velocity for a box, so the hold falls out arithmetically rather
    than needing its own branch -- pinned because a box's `vel` comes from a `torch.where` that
    could silently start passing the enemy's velocity through."""
    cfg, params, gen, state, bank = _setup(kind=Kind.BOT_RIFLE, map_h=30, map_w=30)
    k = int(Kind.BOT_RIFLE)
    params.fire_lateral_speed_limit[:, k] = 0.01     # anything moving at all would be held
    params.fire_lateral_hold_range_fraction[:, k] = 0.0
    # No visible enemy: put the victim far away and out of the picture, leave a box in reach.
    state.ent_alive[0, VICTIM] = False
    state.box_alive[0, 0] = True
    state.box_pos[0, 0] = torch.tensor([_SHOOTER_POS[0] - 2.0, _SHOOTER_POS[1]])
    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert bool(tgt.is_box[0, SHOOTER])
    assert torch.all(tgt.vel[0, SHOOTER] == 0)
    fire, _dir, _pt = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert bool(fire[0, SHOOTER]) is True


# =================================================================================================
# the cone gate -- derived from attack_arc_rad rather than carried as its own field
# =================================================================================================

def test_a_positive_attack_arc_gates_fire_on_facing_and_a_zero_arc_does_not():
    k = int(Kind.BOT_SNIPER)
    for arc, facing_away, expected in ((0.0, True, True),    # no arc: facing is irrelevant
                                       (0.6, False, True),   # arc, facing the victim
                                       (0.6, True, False)):  # arc, facing away
        cfg, params, gen, state, bank = _setup()
        params.attack_arc_rad[:, k] = arc
        if facing_away:
            state.ent_facing[0, SHOOTER] += math.pi
        assert _fires(state, bank, params, cfg, gen) is expected, (arc, facing_away)


def test_hitscan_sweep_widens_the_cone_gate_by_half_the_sweep():
    """A swept attack fans across `hitscan_sweep_rad`, so the reachable half-angle is
    (sweep + arc) / 2. Without the sweep term Buzz would refuse to start a sweep whose later
    sub-swings land squarely (Step C2)."""
    k = int(Kind.BOT_MELEE)
    cfg, params, gen, _s, _b = _setup(kind=Kind.BOT_MELEE)
    arc = float(params.attack_arc_rad[0, k])
    # Off-bearing by more than arc/2 but less than (arc + sweep)/2.
    offset = arc / 2.0 + 0.1
    sweep_needed = 2.0 * (offset - arc / 2.0) + 0.2

    for sweep, expected in ((0.0, False), (sweep_needed, True)):
        cfg, params, gen, state, bank = _setup(
            kind=Kind.BOT_MELEE, shooter=(_VICTIM_POS[0] + 1.5, _VICTIM_POS[1]))
        params.hitscan_sweep_rad[:, k] = sweep
        state.ent_facing[0, SHOOTER] += offset
        assert _fires(state, bank, params, cfg, gen) is expected, sweep


# =================================================================================================
# aim models
# =================================================================================================

def test_direct_aims_at_the_targets_current_position_with_no_lead():
    cfg, params, gen, state, bank = _setup(victim_vel=(0.0, 6.0))
    params.aim_model[:, int(Kind.BOT_SNIPER)] = int(AimModel.DIRECT)
    _fire, aim_dir, aim_point = _run(state, bank, params, cfg, gen)

    assert torch.allclose(aim_point[0, SHOOTER], state.ent_pos[0, VICTIM])
    to_target = state.ent_pos[0, VICTIM] - state.ent_pos[0, SHOOTER]
    assert torch.allclose(aim_dir[0, SHOOTER], to_target / to_target.norm(), atol=1e-6)


def test_lead_aims_ahead_of_a_crossing_target_and_places_the_point_at_target_distance():
    k = int(Kind.BOT_SNIPER)
    cfg, params, gen, state, bank = _setup(victim_vel=(0.0, 6.0))
    params.aim_model[:, k] = int(AimModel.LEAD)
    params.lead_target_fraction[:, k] = 1.0
    _fire, aim_dir, aim_point = _run(state, bank, params, cfg, gen)

    # The victim moves +y, so a leading shot must point +y of the straight line to it.
    assert aim_dir[0, SHOOTER, 1] > 0
    # aim_point is aim_dir scaled to the target's own distance, not to the lead solution's.
    dist = (state.ent_pos[0, VICTIM] - state.ent_pos[0, SHOOTER]).norm()
    reach = (aim_point[0, SHOOTER] - state.ent_pos[0, SHOOTER]).norm()
    assert torch.allclose(reach, dist, atol=1e-5)

    # ...and with lead_target_fraction 0 it degenerates to the straight bearing.
    cfg, params, gen, state, bank = _setup(victim_vel=(0.0, 6.0))
    params.aim_model[:, k] = int(AimModel.LEAD)
    params.lead_target_fraction[:, k] = 0.0
    _fire, aim_dir, _pt = _run(state, bank, params, cfg, gen)
    to_target = state.ent_pos[0, VICTIM] - state.ent_pos[0, SHOOTER]
    assert torch.allclose(aim_dir[0, SHOOTER], to_target / to_target.norm(), atol=1e-6)


def test_lob_with_a_fixed_flight_time_uses_the_closed_form_intercept():
    """With `proj_flight_seconds > 0` the shell lands in that many seconds whatever the distance,
    so the lead is exactly `vel * fraction * flight` -- independent of proj_speed and of range."""
    k = int(Kind.BOT_ARTILLERY)
    vel = (0.0, 3.0)
    cfg, params, gen, state, bank = _setup(kind=Kind.BOT_ARTILLERY, victim_vel=vel)
    params.aim_model[:, k] = int(AimModel.LOB)
    params.proj_flight_seconds[:, k] = 2.0
    params.lead_target_fraction[:, k] = 0.5
    params.proj_speed[:, k] = 999.0  # must not matter
    _fire, _dir, aim_point = _run(state, bank, params, cfg, gen)

    expected = state.ent_pos[0, VICTIM] + torch.tensor(vel) * (0.5 * 2.0)
    assert torch.allclose(aim_point[0, SHOOTER], expected, atol=1e-5)


def test_lob_without_a_flight_time_falls_back_to_the_speed_based_intercept():
    """flight_seconds == 0 is the pre-Grom constant-speed arc, and a lobbed kind configured that
    way must still lead -- a slower shell has to lead further than a fast one."""
    k = int(Kind.BOT_ARTILLERY)
    params_by_speed = {}
    for speed in (3.0, 30.0):
        cfg, params, gen, state, bank = _setup(kind=Kind.BOT_ARTILLERY, victim_vel=(0.0, 4.0))
        params.aim_model[:, k] = int(AimModel.LOB)
        params.proj_flight_seconds[:, k] = 0.0
        params.lead_target_fraction[:, k] = 1.0
        params.proj_speed[:, k] = speed
        _fire, _dir, aim_point = _run(state, bank, params, cfg, gen)
        params_by_speed[speed] = float(aim_point[0, SHOOTER, 1] - state.ent_pos[0, VICTIM, 1])

    assert params_by_speed[3.0] > params_by_speed[30.0] > 0


def test_aim_noise_is_angular_for_lead_and_positional_for_lob():
    """The two noise models are not interchangeable: LEAD rotates the bearing (so the aim point
    stays exactly `dist` away) while LOB scatters the landing point (so its distance varies)."""
    n_envs = 400
    for kind, model, std_field, expect_fixed_reach in (
        (Kind.BOT_SNIPER, AimModel.LEAD, "aim_noise_std_rad", True),
        (Kind.BOT_ARTILLERY, AimModel.LOB, "aim_noise_tiles", False),
    ):
        k = int(kind)
        cfg, params, gen = cfg_and_params(n_enemies=1, n_envs=n_envs)
        state = fresh_state(cfg, params, n_envs=n_envs, enemy_kind=kind)
        bank = FakeBank(grid(20, 20))
        state.ent_pos[:, VICTIM] = torch.tensor(_VICTIM_POS)
        state.ent_pos[:, SHOOTER] = torch.tensor(_SHOOTER_POS)
        state.ent_facing[:, SHOOTER] = math.pi
        params.aim_model[:, k] = int(model)
        params.lead_target_fraction[:, k] = 0.0
        getattr(params, std_field)[:, k] = 0.10
        _fire, aim_dir, aim_point = _run(state, bank, params, cfg, gen)

        reach = (aim_point[:, SHOOTER] - state.ent_pos[:, SHOOTER]).norm(dim=-1)
        dist = (state.ent_pos[:, VICTIM] - state.ent_pos[:, SHOOTER]).norm(dim=-1)
        if expect_fixed_reach:
            assert torch.allclose(reach, dist, atol=1e-4)
            bearing = torch.atan2(aim_dir[:, SHOOTER, 1], aim_dir[:, SHOOTER, 0])
            offset = torch.remainder(bearing - math.pi + math.pi, 2 * math.pi) - math.pi
            assert abs(float(offset.std()) - 0.10) < 0.02
            assert abs(float(offset.mean())) < 0.02
        else:
            assert float((reach - dist).abs().max()) > 0.05
            scatter = aim_point[:, SHOOTER] - state.ent_pos[:, VICTIM]
            assert abs(float(scatter[:, 1].std()) - 0.10) < 0.02


# =================================================================================================
# generalization: a kind's whole rule is data, not code
# =================================================================================================

def test_a_kind_can_be_redefined_into_a_different_archetype_from_params_alone():
    """The E1 acceptance test. BOT_MELEE ships as a point-blank DIRECT cone swinger with no LOS
    requirement; rewriting only its per-kind params -- no Python -- must turn it into a
    long-range LOS-gated leading shooter with a range fraction, which is what "a sixth brawler is
    a brawlers.yaml block" has to mean."""
    k = int(Kind.BOT_MELEE)

    def make(**pos_kwargs):
        cfg, params, gen, state, bank = _setup(kind=Kind.BOT_MELEE, map_h=30, map_w=30,
                                               **pos_kwargs)
        params.attack_range[:, k] = 10.0
        params.attack_arc_rad[:, k] = 0.0        # no cone any more
        params.hitscan_sweep_rad[:, k] = 0.0
        params.fire_needs_los[:, k] = 1
        params.fire_range_fraction[:, k] = 0.5   # holds fire past 5 tiles
        params.aim_model[:, k] = int(AimModel.LEAD)
        params.lead_target_fraction[:, k] = 1.0
        params.proj_speed[:, k] = 4.0
        return cfg, params, gen, state, bank

    # Facing away no longer matters, since it has no arc.
    cfg, params, gen, state, bank = make(shooter=(_VICTIM_POS[0] + 4.0, _VICTIM_POS[1]))
    state.ent_facing[0, SHOOTER] += math.pi
    assert _fires(state, bank, params, cfg, gen) is True

    # ...but the range fraction does: 7 tiles is inside attack_range 10 and outside 0.5 * 10.
    cfg, params, gen, state, bank = make(shooter=(_VICTIM_POS[0] + 7.0, _VICTIM_POS[1]))
    assert _fires(state, bank, params, cfg, gen) is False

    # ...and so does LOS, which the shipped melee rule ignores entirely.
    cfg, params, gen, state, bank = make(shooter=(_VICTIM_POS[0] + 4.0, _VICTIM_POS[1]), wall=True)
    assert _fires(state, bank, params, cfg, gen) is False

    # ...and it now leads a crossing target instead of pointing straight at it.
    cfg, params, gen, state, bank = make(shooter=(_VICTIM_POS[0] + 4.0, _VICTIM_POS[1]),
                                         victim_vel=(0.0, 6.0))
    _fire, aim_dir, _pt = _run(state, bank, params, cfg, gen)
    assert aim_dir[0, SHOOTER, 1] > 0


# =================================================================================================
# the shipped roster's rules -- identity, not balance
# =================================================================================================

def test_shipped_roster_rules_match_the_documented_table():
    """Pins each brawler's rule SHAPE (does it need a line, how does it aim), not its tuned
    numbers -- those are game statistics and are meant to move. See bots/combat_rules.py's table.
    """
    cfg, params, _gen = cfg_and_params(map_h=40, map_w=40)
    expected_los = {Kind.BOT_SNIPER: 1, Kind.BOT_ARTILLERY: 0, Kind.BOT_MELEE: 0, Kind.BOT_RIFLE: 1}
    expected_aim = {Kind.BOT_SNIPER: AimModel.LEAD, Kind.BOT_ARTILLERY: AimModel.LOB,
                    Kind.BOT_MELEE: AimModel.DIRECT, Kind.BOT_RIFLE: AimModel.LEAD}
    for kind, los in expected_los.items():
        assert int(params.fire_needs_los[0, int(kind)]) == los, kind
        assert int(params.aim_model[0, int(kind)]) == int(expected_aim[kind]), kind

    # Only the shotgun holds fire on distance or on crossers; every other kind's rule is bare.
    for kind in (Kind.BOT_SNIPER, Kind.BOT_ARTILLERY, Kind.BOT_MELEE):
        assert float(params.fire_range_fraction[0, int(kind)]) == 0.0, kind
        assert float(params.fire_lateral_speed_limit[0, int(kind)]) == 0.0, kind
    assert 0 < float(params.fire_range_fraction[0, int(Kind.BOT_RIFLE)]) < 1.0
    assert float(params.fire_lateral_speed_limit[0, int(Kind.BOT_RIFLE)]) > 0

    # The cone gate is derived, so exactly the kinds with an arc are the kinds that swing one.
    for kind in (Kind.BOT_SNIPER, Kind.BOT_ARTILLERY, Kind.BOT_RIFLE):
        assert float(params.attack_arc_rad[0, int(kind)]) == 0.0, kind
    assert float(params.attack_arc_rad[0, int(Kind.BOT_MELEE)]) > 0

    # Every bot has a preferred engagement distance for KITE to hold; the hero's is never read.
    for kind in (Kind.BOT_SNIPER, Kind.BOT_ARTILLERY, Kind.BOT_MELEE, Kind.BOT_RIFLE):
        assert 0 < float(params.desired_range_fraction[0, int(kind)]) <= 1.0, kind


# =================================================================================================
# config plumbing and validation
# =================================================================================================

def test_an_unknown_aim_model_name_is_rejected_by_name():
    from brawl_sim.config import build_params, load_config
    import yaml
    cfg = load_config("configs/default.yaml")
    spec = {**yaml.safe_load(open("configs/default.yaml").read()),
            **yaml.safe_load(open("configs/brawlers.yaml").read())}
    spec["bot_sniper"] = {**spec["bot_sniper"], "aim_model": "PARABOLIC"}
    gen = torch.Generator(device="cpu").manual_seed(0)
    with pytest.raises(KeyError, match="PARABOLIC"):
        build_params(cfg, n_envs=2, device="cpu", gen=gen, spec=spec)


@pytest.mark.parametrize("field,value,match", [
    ("fire_range_fraction", 1.5, "fire_range_fraction"),
    ("desired_range_fraction", 1.5, "desired_range_fraction"),
])
def test_validate_rejects_out_of_range_fractions(field, value, match):
    cfg, params, _gen = cfg_and_params(map_h=40, map_w=40)
    getattr(params, field)[:, int(Kind.BOT_SNIPER)] = value
    with pytest.raises(ValueError, match=match):
        validate(cfg, params)


def test_validate_rejects_a_lateral_hold_fraction_with_no_speed_limit():
    """A range fraction paired with a 0 speed limit is a rule that silently does nothing -- the
    class of config mistake that costs a whole training run before anyone notices."""
    cfg, params, _gen = cfg_and_params(map_h=40, map_w=40)
    params.fire_lateral_speed_limit[:, int(Kind.BOT_SNIPER)] = 0.0
    params.fire_lateral_hold_range_fraction[:, int(Kind.BOT_SNIPER)] = 0.6
    with pytest.raises(ValueError, match="fire_lateral_speed_limit"):
        validate(cfg, params)


def test_validate_rejects_a_bot_with_no_desired_range_but_allows_the_hero():
    """0 would make a KITE bot of that kind hold range 0, i.e. charge to point blank. The hero
    is exempt: its movement comes from decode_action and never reads the field."""
    cfg, params, _gen = cfg_and_params(map_h=40, map_w=40)
    params.desired_range_fraction[:, int(Kind.BOT_MELEE)] = 0.0
    with pytest.raises(ValueError, match="desired_range_fraction"):
        validate(cfg, params)

    cfg, params, _gen = cfg_and_params(map_h=40, map_w=40)
    assert float(params.desired_range_fraction[0, int(Kind.HERO_MORTIS)]) == 0.0
    validate(cfg, params)  # must not raise
