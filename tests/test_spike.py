"""Spike: the slow shell that deals nothing, and the six-arm star that does all the work.

Spike is the first brawler with `split_count` != 4 (the arm count was a module constant until he
existed), the first with a SECOND lobbing `Proj` member, and the first whose shell carries damage
it never deals to anyone. These tests pin those three things plus the damage-vs-offset profile
they combine into. Grom's own cross is owned by tests/test_projectiles.py and is only
cross-checked here where Spike could have broken it.
"""
import math
from dataclasses import dataclass

import pytest
import torch
import yaml

from brawl_sim.config import build_params, load_config, validate
from brawl_sim.constants import TILE_BLOCKS_PROJ, Kind, Proj, ProjClass, Tile
from brawl_sim.core import projectiles as proj
from brawl_sim.core.state import allocate

CONFIGS_DEFAULT = "configs/default.yaml"
SPIKE = int(Kind.BOT_SPIKE)
SHARD_DAMAGE = 1080.0


@dataclass
class _FakeBank:
    blocks_proj: torch.Tensor


def _setup(map_h=40, map_w=40, wall_cols=()):
    cfg = load_config(CONFIGS_DEFAULT, overrides={
        "world": {"map_h": map_h, "map_w": map_w},
        "observation": {"include_world_grid": False},
    })
    spec = {
        **yaml.safe_load(open(CONFIGS_DEFAULT).read()),
        **yaml.safe_load(open("configs/brawlers.yaml").read()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=1, device="cpu", gen=gen, spec=spec)

    state = allocate(cfg, n_envs=1, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    # Everyone is parked in a corner unless a test places them: an entity left at the origin would
    # sit inside the star of every shell these tests fire.
    state.ent_pos.fill_(float(map_w) - 2.0)
    state.map_id.fill_(0)

    tiles = torch.full((map_h, map_w), int(Tile.FLOOR), dtype=torch.int64)
    tiles[0, :] = tiles[-1, :] = tiles[:, 0] = tiles[:, -1] = int(Tile.WALL)
    for col in wall_cols:
        tiles[:, col] = int(Tile.WALL)
    return cfg, params, state, _FakeBank(blocks_proj=TILE_BLOCKS_PROJ[tiles].unsqueeze(0))


def _throw(state, cfg, params, kind, origin, landing):
    """Fires one shell from entity 1 and returns the damage it carries."""
    k = int(kind)
    state.ent_kind[0, 1] = k
    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 1] = True
    org = torch.zeros(1, cfg.n_entities, 2)
    org[0, 1] = torch.tensor(origin)
    aim = torch.zeros(1, cfg.n_entities, 2)
    aim[0, 1] = torch.tensor(landing)
    damage = params.base_damage[:, k].unsqueeze(-1).expand(1, cfg.n_entities).clone()
    kinds = torch.full((1, cfg.n_entities), k, dtype=torch.int64)
    proj.spawn_volley(state, fire, org, aim - org, aim, kinds, damage, params, cfg)
    return float(params.base_damage[0, k])


def _run(state, bank, params, cfg, ticks=200):
    """Runs until the buffer empties. Returns (damage to entity 0, ticks to detonation)."""
    total, detonated_at = 0.0, -1
    for i in range(ticks):
        if not bool(state.prj_alive.any()):
            break
        dmg_ent, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
        total += float(dmg_ent[0, 0])
        still_flying = bool((state.prj_class[0][state.prj_alive[0]] == int(ProjClass.ARTILLERY)).any())
        if detonated_at < 0 and not still_flying:
            detonated_at = i + 1
    return total, detonated_at


def _detonate(state, bank, params, cfg, limit=200):
    """Steps until the shell has gone off and stops on THAT tick, so the shards are inspected
    exactly as spawned. Running even one tick further has already moved them 0.33 tiles, which is
    the difference between reading a shard's dist_left as 4.0 and as 3.67."""
    for i in range(limit):
        proj.step_projectiles(state, bank, params, cfg)
        if not bool((state.prj_class[0][state.prj_alive[0]] == int(ProjClass.ARTILLERY)).any()):
            return i + 1
    raise AssertionError("the shell never detonated")


def _damage_at(offset, landing=(20.0, 20.0), origin=(13.0, 20.0), **kw):
    """Total damage a lone victim standing at `landing + offset` takes from one Spike shell."""
    cfg, params, state, bank = _setup(**kw)
    state.ent_pos[0, 0] = torch.tensor([landing[0] + offset[0], landing[1] + offset[1]])
    _throw(state, cfg, params, Kind.BOT_SPIKE, origin, landing)
    return _run(state, bank, params, cfg)[0]


def _polar(dist, degrees):
    return (dist * math.cos(math.radians(degrees)), dist * math.sin(math.radians(degrees)))


# ---- the shell ------------------------------------------------------------------

def test_the_shell_is_an_artillery_class_spike_shell():
    """Two claims that are easy to get silently wrong. SPIKE_SHELL is a separate `Proj` member, so
    the shell is only ARTILLERY-class because constants.PROJ_CLASS_OF says so -- the rule it
    replaced (`proj_kind == ARTILLERY_SHELL`) would have made it an ordinary PROJECTILE, which
    damages the first body it touches and never reaches _spawn_splits at all."""
    cfg, params, state, bank = _setup()
    _throw(state, cfg, params, Kind.BOT_SPIKE, [13.0, 20.0], [20.0, 20.0])

    live = state.prj_alive[0]
    assert int(live.sum()) == 1
    assert int(state.prj_kind[0][live][0]) == int(Proj.SPIKE_SHELL)
    assert int(state.prj_class[0][live][0]) == int(ProjClass.ARTILLERY)


def test_the_shell_flies_at_a_constant_speed_rather_than_a_fixed_time():
    """The spec calls it "a standard slow moving projectile (similar to Brock's)" taking "around
    1.5 seconds to fire its full distance". That is a SPEED (8 / 1.5 = 5.33 tiles/s), so a
    half-range throw must take half as long -- the opposite of Grom, whose timed lob takes 1.25 s
    whatever the distance. Both throws are measured because one alone cannot tell the models apart.
    """
    cfg, params, _, _ = _setup()
    assert float(params.proj_flight_seconds[0, SPIKE]) == 0.0

    far_cfg, far_params, far_state, far_bank = _setup()
    _throw(far_state, far_cfg, far_params, Kind.BOT_SPIKE, [12.0, 20.0], [20.0, 20.0])   # 8 tiles
    far_ticks = _run(far_state, far_bank, far_params, far_cfg)[1]

    near_cfg, near_params, near_state, near_bank = _setup()
    _throw(near_state, near_cfg, near_params, Kind.BOT_SPIKE, [16.0, 20.0], [20.0, 20.0])  # 4 tiles
    near_ticks = _run(near_state, near_bank, near_params, near_cfg)[1]

    speed = float(params.proj_speed[0, SPIKE])
    assert abs(far_ticks * cfg.dt - 8.0 / speed) <= 2 * cfg.dt, "the 8-tile throw took the wrong time"
    assert abs(near_ticks * cfg.dt - 4.0 / speed) <= 2 * cfg.dt, "the 4-tile throw took the wrong time"
    assert near_ticks < far_ticks * 0.75, (
        f"a half-range throw took {near_ticks} ticks against {far_ticks} for full range -- that is "
        f"a fixed-flight-time lob, not a constant-speed shot"
    )


def test_the_landing_itself_deals_nothing_even_at_dead_centre():
    """"His main projectile ... deals 0 damage when it lands." `aoe_radius: 0` is only half of
    that: the blast test is `ent_dist <= prj_aoe`, which is TRUE at exact coincidence, so without
    step_projectiles' `prj_aoe > 0` guard a victim standing precisely on the landing point would
    eat the shell's full 1080. Not a float curiosity -- a LOB bot with no aim noise aims at a
    stationary target's exact position."""
    cfg, params, state, bank = _setup()
    assert float(params.aoe_radius[0, SPIKE]) == 0.0

    landing = (20.0, 20.0)
    state.ent_pos[0, 0] = torch.tensor(landing)   # exactly on the landing point
    _throw(state, cfg, params, Kind.BOT_SPIKE, [13.0, 20.0], landing)
    assert _run(state, bank, params, cfg)[0] == 0.0


def test_the_blast_path_is_live_and_it_is_aoe_radius_zero_that_silences_it():
    """The complement of the test above: giving Spike a blast radius makes the same shot deal the
    shell's damage. Without this, "no damage on landing" would also pass if the detonation path
    were broken outright -- which would take Grom down with it."""
    cfg, params, state, bank = _setup()
    params.aoe_radius[:, SPIKE] = 0.6            # pretend he is Grom-shaped for one shot

    landing = (20.0, 20.0)
    state.ent_pos[0, 0] = torch.tensor(landing)
    full = _throw(state, cfg, params, Kind.BOT_SPIKE, [13.0, 20.0], landing)
    assert abs(_run(state, bank, params, cfg)[0] - full) < 1e-3


# ---- the star -------------------------------------------------------------------

def test_the_shell_splits_into_six_evenly_spaced_shards():
    cfg, params, state, bank = _setup()
    landing = torch.tensor([20.0, 20.0])
    _throw(state, cfg, params, Kind.BOT_SPIKE, [13.0, 20.0], landing.tolist())
    _detonate(state, bank, params, cfg)

    live = state.prj_alive[0]
    assert int(live.sum()) == 6, "expected a six-arm star"
    bearings = sorted(round(math.degrees(math.atan2(v[1], v[0])) % 360.0, 3)
                      for v in state.prj_vel[0][live].tolist())
    assert bearings == [0.0, 60.0, 120.0, 180.0, 240.0, 300.0], (
        f"the arms are not a regular hexagon starting at +x: {bearings}"
    )
    # The ring is centred on the landing point, so the shards' midpoint IS where the shell went off.
    assert torch.allclose(state.prj_pos[0][live].mean(dim=0), landing, atol=1e-4)


def test_each_shard_carries_the_shells_full_damage_over_four_tiles_in_point_six_seconds():
    """Three numbers straight off the spec. The shard SPEED is derived from two of them
    (split_distance / split_seconds) rather than read from proj_speed, which for Spike describes
    the shell instead."""
    cfg, params, state, bank = _setup()
    full = _throw(state, cfg, params, Kind.BOT_SPIKE, [13.0, 20.0], [20.0, 20.0])
    _detonate(state, bank, params, cfg)

    live = state.prj_alive[0]
    distance = float(params.split_distance[0, SPIKE])
    seconds = float(params.split_seconds[0, SPIKE])
    assert full == SHARD_DAMAGE
    assert torch.allclose(state.prj_damage[0][live], torch.full((6,), full))
    assert torch.allclose(state.prj_dist_left[0][live], torch.full((6,), distance))
    assert torch.allclose(state.prj_vel[0][live].norm(dim=-1), torch.full((6,), distance / seconds))

    ticks = 0
    while bool(state.prj_alive.any()) and ticks < 200:
        proj.step_projectiles(state, bank, params, cfg)
        ticks += 1
    assert abs(ticks * cfg.dt - seconds) <= 2 * cfg.dt, (
        f"the arms took {ticks * cfg.dt:.2f}s to finish, expected ~{seconds}s"
    )


def test_shards_are_ordinary_projectiles_that_walls_stop():
    """The shell arcs (ProjClass.ARTILLERY has no wall collision at all), but the shards must not
    -- otherwise a Spike could pour six 1080 spikes through solid cover."""
    landing = (20.0, 20.0)
    open_ground = _damage_at((3.0, 0.0), landing=landing)
    behind_wall = _damage_at((3.0, 0.0), landing=landing, wall_cols=(22,))
    assert open_ground == SHARD_DAMAGE, "an arm should reach 3 tiles down its own bearing"
    assert behind_wall == 0.0, "a shard went through a wall"


# ---- what it actually does to somebody ------------------------------------------

def test_a_near_miss_lands_two_or_three_spikes():
    """The behaviour the spec describes: "when his attack hits an enemy, it hurts by detonating
    right next to them (so they take around 2-4 of the mini-projectiles)". All the damage comes
    from how many arms sweep the victim, so it is asserted as a shard COUNT."""
    for dist in (0.1, 0.3, 0.5):
        for bearing in range(0, 360, 30):
            damage = _damage_at(_polar(dist, bearing))
            spikes = damage / SHARD_DAMAGE
            assert spikes == int(spikes), f"partial shard damage {damage} at {dist}/{bearing} deg"
            assert 2 <= spikes <= 3, (
                f"a shell landing {dist} tiles from a victim (bearing {bearing} deg) landed "
                f"{spikes:.0f} spikes; expected 2-3"
            )


def test_the_star_thins_out_with_distance_instead_of_stopping():
    """Past ~1.4 tiles the 60-degree arms stop overlapping, so the solid disc becomes six separate
    lines: still lethal on a bearing, harmless between two. That gap is the counterplay, and a
    victim standing ON an arm has to still be caught right out to the end of it."""
    on_arm, between_arms = 0.0, 30.0

    assert _damage_at(_polar(2.5, on_arm)) == SHARD_DAMAGE
    assert _damage_at(_polar(4.5, on_arm)) == SHARD_DAMAGE
    assert _damage_at(_polar(2.5, between_arms)) == 0.0
    # Reach = rim (0.7) + split_distance (4.0) + hit radius (0.7). Well past it, nothing lands.
    assert _damage_at(_polar(7.0, on_arm)) == 0.0


def test_one_shell_cannot_land_more_than_three_spikes():
    """The cap that makes the shards' spawn ring load-bearing. They start at `unit_radius +
    proj_radius` from the landing point -- exactly their own hit radius -- so no victim is ever
    inside more than three arms at once. Spawning them at the centre instead would put all SIX on
    a victim standing there: 6480 damage, enough to delete a 6000 HP brawler with one shell."""
    worst = max(
        _damage_at(_polar(dist, bearing))
        for dist in (0.2, 0.5, 0.8, 1.1, 1.4)
        for bearing in range(0, 61, 10)  # one sixth of the ring covers every distinct geometry
    )
    assert worst <= 3 * SHARD_DAMAGE, f"a single shell landed {worst / SHARD_DAMAGE:.0f} spikes"


# ---- the roster around him ------------------------------------------------------

def test_groms_cross_is_untouched_by_the_six_arm_ring():
    """`split_count` replaced a module-level constant that WAS Grom's arm count. The derived ring
    has to reproduce his four world-aligned arms exactly -- `cos(pi/2)` is 6.1e-17, not 0, so this
    only holds because core/projectiles._ring snaps near-zeros."""
    cfg, params, state, bank = _setup()
    _throw(state, cfg, params, Kind.BOT_ARTILLERY, [13.0, 20.0], [20.0, 20.0])
    _detonate(state, bank, params, cfg)

    live = state.prj_alive[0]
    assert int(live.sum()) == 4
    grom = int(Kind.BOT_ARTILLERY)
    speed = float(params.split_distance[0, grom] / params.split_seconds[0, grom])
    assert sorted(state.prj_vel[0][live].tolist()) == sorted(
        [[speed, 0.0], [-speed, 0.0], [0.0, speed], [0.0, -speed]]
    )


def test_no_other_kind_splits():
    """`split_count` defaults to 0, and _spawn_splits gates on it, so the field costs the rest of
    the roster one comparison. A kind that grew a ring by accident would be firing free
    projectiles on every shot."""
    _, params, _, _ = _setup()
    for kind in Kind:
        expected = {Kind.BOT_ARTILLERY: 4, Kind.BOT_SPIKE: 6}.get(kind, 0)
        assert int(params.split_count[0, int(kind)]) == expected, f"{kind!r} has the wrong ring"


@pytest.mark.parametrize("count, message", [
    (0, "split_count is 0"),
    (proj.MAX_SPLITS + 1, "exceeds core/projectiles.MAX_SPLITS"),
])
def test_validate_rejects_a_ring_that_would_silently_misfire(count, message):
    """Both failures load fine and then quietly do the wrong thing forever: 0 arms detonates into
    nothing, and more arms than MAX_SPLITS is truncated by the fixed-width grid _spawn_splits
    allocates against."""
    cfg, params, _, _ = _setup()
    params.split_count[:, SPIKE] = count
    with pytest.raises(ValueError, match=message):
        validate(cfg, params)
