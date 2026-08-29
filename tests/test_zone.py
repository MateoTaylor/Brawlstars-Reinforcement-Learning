from pathlib import Path

import torch
import yaml

from brawl_sim.config import load_config, build_params
from brawl_sim.core import zone
from brawl_sim.core.state import allocate

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _cfg_and_params(n_envs=1, **overrides):
    cfg = load_config(CONFIGS_DEFAULT, overrides=overrides or None)
    import yaml
    spec = {
        **yaml.safe_load(open(CONFIGS_DEFAULT).read()),
        **yaml.safe_load(open("configs/brawlers.yaml").read()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    return cfg, params, gen


_TEST_MAX_HP = 8000.0


def _fresh_state(cfg, n_envs=1, max_hp=_TEST_MAX_HP):
    """`allocate` zero-inits everything, including ent_max_hp. That was harmless while zone damage
    was a flat HP/s, but since Step B3 the rate is a FRACTION of max HP -- a zero-HP body takes
    zero zone damage, so every damage assertion here would trivially compare 0 to 0. Give the
    fixture a real body."""
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.ent_max_hp.fill_(max_hp)
    state.ent_hp.fill_(max_hp)
    return state


# ---- init_zone -----------------------------------------------------------------------

def test_init_zone_sets_full_map_rect_and_start_time():
    cfg, params, gen = _cfg_and_params(n_envs=1, world={"map_h": 20, "map_w": 30})
    state = _fresh_state(cfg)
    mask = torch.ones(1, dtype=torch.bool)

    zone.init_zone(state, mask, params, cfg)
    assert torch.allclose(state.zone_lo[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(state.zone_hi[0], torch.tensor([30.0, 20.0]))
    assert torch.allclose(state.zone_next_t[0], params.zone_start_time[0])
    assert int(state.zone_step[0]) == 0


def test_init_zone_only_touches_masked_rows():
    cfg, params, gen = _cfg_and_params(n_envs=4)
    state = _fresh_state(cfg, n_envs=4)
    full_mask = torch.ones(4, dtype=torch.bool)
    zone.init_zone(state, full_mask, params, cfg)

    # perturb everything, then re-init only rows 1 and 3
    state.zone_lo.fill_(5.0)
    state.zone_hi.fill_(6.0)
    state.zone_step.fill_(7)
    partial_mask = torch.tensor([False, True, False, True])
    zone.init_zone(state, partial_mask, params, cfg)

    for row in (0, 2):
        assert torch.allclose(state.zone_lo[row], torch.tensor([5.0, 5.0]))
        assert int(state.zone_step[row]) == 7
    for row in (1, 3):
        assert torch.allclose(state.zone_lo[row], torch.tensor([0.0, 0.0]))
        assert int(state.zone_step[row]) == 0


# ---- step_zone (acceptance: exact first-shrink time) -----------------------------------

def test_first_shrink_at_exact_configured_time():
    # zone_start_time is DERIVED (start_fraction * max_episode_steps * dt), so the expectation is
    # computed from the same config rather than hardcoded. It used to assert a literal 30.0, which
    # broke the moment configs/default.yaml's start_fraction was tuned 0.2 -> 0.1 -- a legitimate
    # balance change failing a test that was only ever checking the arithmetic.
    cfg, params, gen = _cfg_and_params(n_envs=1)
    spec = yaml.safe_load((CONFIGS / "default.yaml").read_text())
    expected = float(spec["zone"]["start_fraction"]) * cfg.max_episode_steps * cfg.dt

    state = _fresh_state(cfg)
    mask = torch.ones(1, dtype=torch.bool)
    zone.init_zone(state, mask, params, cfg)
    assert abs(params.zone_start_time[0].item() - expected) < 1e-4

    expected_tick = round(expected / cfg.dt)
    shrink_tick = None
    for step in range(expected_tick + 100):
        state.time += cfg.dt
        prev_step = int(state.zone_step[0])
        zone.step_zone(state, params, cfg)
        if int(state.zone_step[0]) != prev_step:
            shrink_tick = step
            shrink_time = state.time[0].item()
            break

    # float32 accumulation of N additions of dt doesn't land on an exact binary value, so the
    # crossing can fall on either side of the nominal tick -- assert it is within one tick, with
    # the resulting time within one dt, rather than hardcoding a single index.
    assert shrink_tick is not None
    assert abs(shrink_tick - (expected_tick - 1)) <= 1
    assert abs(shrink_time - expected) < cfg.dt + 1e-3


def test_shrink_moves_lo_and_hi_by_tiles_per_step():
    cfg, params, gen = _cfg_and_params(n_envs=1)
    state = _fresh_state(cfg)
    mask = torch.ones(1, dtype=torch.bool)
    zone.init_zone(state, mask, params, cfg)

    lo_before = state.zone_lo.clone()
    hi_before = state.zone_hi.clone()
    state.time.fill_(params.zone_start_time[0].item())
    zone.step_zone(state, params, cfg)

    step_size = params.zone_tiles_per_step[0].item()
    assert torch.allclose(state.zone_lo[0], lo_before[0] + step_size)
    assert torch.allclose(state.zone_hi[0], hi_before[0] - step_size)
    assert int(state.zone_step[0]) == 1


def test_exactly_one_shrink_per_call_even_if_far_overdue():
    cfg, params, gen = _cfg_and_params(n_envs=1)
    state = _fresh_state(cfg)
    mask = torch.ones(1, dtype=torch.bool)
    zone.init_zone(state, mask, params, cfg)

    state.time.fill_(params.zone_start_time[0].item() + 50 * params.zone_step_seconds[0].item())
    zone.step_zone(state, params, cfg)
    assert int(state.zone_step[0]) == 1  # not 51


def test_rect_never_inverts_after_many_shrinks():
    cfg, params, gen = _cfg_and_params(n_envs=1, world={"map_h": 20, "map_w": 20})
    state = _fresh_state(cfg)
    mask = torch.ones(1, dtype=torch.bool)
    zone.init_zone(state, mask, params, cfg)

    t = params.zone_start_time[0].item()
    for _ in range(500):
        t += params.zone_step_seconds[0].item()
        state.time.fill_(t)
        zone.step_zone(state, params, cfg)
        width = (state.zone_hi[0] - state.zone_lo[0])
        assert torch.all(width >= 2.0 - 1e-4)


def test_zone_disabled_no_shrink_no_damage():
    cfg, params, gen = _cfg_and_params(n_envs=1, zone={"enabled": False})
    state = _fresh_state(cfg)
    mask = torch.ones(1, dtype=torch.bool)
    zone.init_zone(state, mask, params, cfg)
    lo_before, hi_before = state.zone_lo.clone(), state.zone_hi.clone()

    state.time.fill_(10000.0)  # way past any conceivable shrink schedule
    zone.step_zone(state, params, cfg)
    assert torch.equal(state.zone_lo, lo_before)
    assert torch.equal(state.zone_hi, hi_before)

    state.ent_alive.fill_(True)
    state.ent_pos[0, :, 0] = -100.0  # every entity "outside" any rect
    dmg = zone.zone_damage(state, params, cfg)
    assert torch.all(dmg == 0)


# ---- zone_damage ------------------------------------------------------------------------

def test_zone_damage_outside_vs_inside():
    cfg, params, gen = _cfg_and_params(n_envs=1, entities={"n_enemies": 1})
    state = _fresh_state(cfg)
    mask = torch.ones(1, dtype=torch.bool)
    zone.init_zone(state, mask, params, cfg)
    state.zone_lo[0] = torch.tensor([5.0, 5.0])
    state.zone_hi[0] = torch.tensor([15.0, 15.0])
    state.ent_alive.fill_(True)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])  # inside
    state.ent_pos[0, 1] = torch.tensor([20.0, 10.0])  # outside

    dmg = zone.zone_damage(state, params, cfg)
    expected = params.zone_hp_fraction[0].item() * state.ent_max_hp[0, 1].item() * cfg.dt
    assert dmg[0, 0].item() == 0.0
    assert abs(dmg[0, 1].item() - expected) < 1e-3


def test_zone_damage_ignores_dead_entities():
    cfg, params, gen = _cfg_and_params(n_envs=1, entities={"n_enemies": 1})
    state = _fresh_state(cfg)
    mask = torch.ones(1, dtype=torch.bool)
    zone.init_zone(state, mask, params, cfg)
    state.zone_lo[0] = torch.tensor([5.0, 5.0])
    state.zone_hi[0] = torch.tensor([15.0, 15.0])
    state.ent_alive[0, 0] = False
    state.ent_pos[0, 0] = torch.tensor([20.0, 10.0])  # outside, but dead

    dmg = zone.zone_damage(state, params, cfg)
    assert dmg[0, 0].item() == 0.0


def test_zone_damage_escalates_with_zone_step():
    cfg, params, gen = _cfg_and_params(n_envs=1, entities={"n_enemies": 0})
    state = _fresh_state(cfg)
    mask = torch.ones(1, dtype=torch.bool)
    zone.init_zone(state, mask, params, cfg)
    state.zone_lo[0] = torch.tensor([5.0, 5.0])
    state.zone_hi[0] = torch.tensor([15.0, 15.0])
    state.ent_alive.fill_(True)
    state.ent_pos[0, 0] = torch.tensor([20.0, 10.0])

    state.zone_step.fill_(0)
    dmg_step0 = zone.zone_damage(state, params, cfg)[0, 0].item()
    state.zone_step.fill_(5)
    dmg_step5 = zone.zone_damage(state, params, cfg)[0, 0].item()

    growth = params.zone_fraction_growth[0].item()
    max_hp = state.ent_max_hp[0, 0].item()
    assert abs((dmg_step5 - dmg_step0) - 5 * growth * max_hp * cfg.dt) < 1e-3
    assert growth == 0 or dmg_step5 > dmg_step0


def test_zone_kills_any_body_in_the_same_time_regardless_of_max_hp():
    """bot_overhaul.md D15: nothing survives more than ~5 seconds in the zone, whoever it is.

    This is THE property the flat-DPS -> proportional change exists to create, so it is asserted
    directly rather than inferred from the formula. Under the old flat `dps: 1000` these three
    bodies died in 6s / 10s / 34s; a fully cube-stacked 34000 HP entity could ignore the zone for
    most of an episode. They must now all take the identical number of ticks."""
    cfg, params, gen = _cfg_and_params(n_envs=3, entities={"n_enemies": 0})
    state = _fresh_state(cfg, n_envs=3)
    mask = torch.ones(3, dtype=torch.bool)
    zone.init_zone(state, mask, params, cfg)
    state.zone_lo[:] = torch.tensor([5.0, 5.0])
    state.zone_hi[:] = torch.tensor([15.0, 15.0])
    state.ent_alive.fill_(True)
    state.ent_pos[:, 0] = torch.tensor([20.0, 10.0])  # all outside

    # Brock, Buzz, and a fully cube-stacked Buzz (16 cubes at +15% each = 3.4x).
    bodies = torch.tensor([6000.0, 10000.0, 34000.0])
    state.ent_max_hp[:, 0] = bodies
    state.ent_hp[:, 0] = bodies

    ticks = torch.zeros(3, dtype=torch.int64)
    for _ in range(2000):
        dmg = zone.zone_damage(state, params, cfg)
        alive = state.ent_hp[:, 0] > 0
        state.ent_hp[:, 0] = torch.clamp(state.ent_hp[:, 0] - dmg[:, 0], min=0.0)
        ticks += alive.to(torch.int64)
        if not bool(alive.any()):
            break

    assert torch.equal(ticks, ticks[0].expand(3)), (
        f"bodies {bodies.tolist()} died in {ticks.tolist()} ticks; the zone must not care about HP"
    )
    seconds = ticks[0].item() * cfg.dt
    expected = 1.0 / params.zone_hp_fraction[0].item()
    assert abs(seconds - expected) <= cfg.dt, (
        f"took {seconds:.2f}s to die, expected ~{expected:.2f}s (1 / max_hp_fraction_per_second)"
    )


# ---- zone_grid --------------------------------------------------------------------------

def test_zone_grid_matches_outside_predicate_world_origin():
    cfg, params, gen = _cfg_and_params(n_envs=1, world={"map_h": 10, "map_w": 10})
    state = _fresh_state(cfg)
    mask = torch.ones(1, dtype=torch.bool)
    zone.init_zone(state, mask, params, cfg)
    state.zone_lo[0] = torch.tensor([3.0, 3.0])
    state.zone_hi[0] = torch.tensor([7.0, 7.0])

    grid = zone.zone_grid(state, cfg, out_h=10, out_w=10, origin=(0.0, 0.0))
    assert grid.shape == (1, 10, 10)

    # cell (row=0, col=0) -> tile center (0.5, 0.5), well outside [3,7]x[3,7]
    assert bool(grid[0, 0, 0])
    # cell (row=5, col=5) -> tile center (5.5, 5.5), inside [3,7]x[3,7]
    assert not bool(grid[0, 5, 5])


def test_zone_grid_view_origin_shifts_the_window():
    cfg, params, gen = _cfg_and_params(n_envs=1, world={"map_h": 20, "map_w": 20})
    state = _fresh_state(cfg)
    mask = torch.ones(1, dtype=torch.bool)
    zone.init_zone(state, mask, params, cfg)
    state.zone_lo[0] = torch.tensor([3.0, 3.0])
    state.zone_hi[0] = torch.tensor([7.0, 7.0])

    # a 4x4 view window whose origin is (5, 5): covers tiles x,y in [5,9)
    grid = zone.zone_grid(state, cfg, out_h=4, out_w=4, origin=torch.tensor([[5.0, 5.0]]))
    # cell (0,0) -> tile center (5.5, 5.5): inside [3,7]x[3,7]
    assert not bool(grid[0, 0, 0])
    # cell (3,3) -> tile center (8.5, 8.5): outside [3,7]x[3,7]
    assert bool(grid[0, 3, 3])


# ---- batched / no-NaN smoke --------------------------------------------------------------

def test_batched_smoke():
    cfg, params, gen = _cfg_and_params(n_envs=8, entities={"n_enemies": 3})
    state = _fresh_state(cfg, n_envs=8)
    mask = torch.ones(8, dtype=torch.bool)
    zone.init_zone(state, mask, params, cfg)

    for _ in range(50):
        state.time += cfg.dt
        zone.step_zone(state, params, cfg)
        state.ent_pos.uniform_(0, 60)
        state.ent_alive.fill_(True)
        dmg = zone.zone_damage(state, params, cfg)
        assert dmg.shape == (8, cfg.n_entities)
        assert not torch.any(torch.isnan(dmg))

    grid = zone.zone_grid(state, cfg, out_h=20, out_w=40, origin=torch.zeros(8, 2))
    assert grid.shape == (8, 20, 40)
    assert not torch.any(torch.isnan(state.zone_lo))
    assert not torch.any(torch.isnan(state.zone_hi))
