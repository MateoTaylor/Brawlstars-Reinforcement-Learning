"""Bull: the second shotgun, and the first brawler that is purely a config block.

Every other brawler added since the archetypes were data-driven brought a mechanic with it --
Edgar lifesteal, Spike a variable split ring. Bull brings none: he reuses Shelly's constant-speed
pellet fan exactly, and differs only in numbers. So these tests are mostly *comparative* -- they
pin the ways he is deliberately not Shelly (harder, tighter, shorter, slower, tankier) rather than
re-testing `spawn_volley`, which tests/test_projectiles.py owns.

The one structural claim is `BULL_SLUG`: sharing Shelly's `Proj` member would have been invisible
in every behavioural test here and still wrong, because `kind_onehot` is the only categorical
weapon signal the policy receives.
"""
import math
from dataclasses import dataclass

import torch
import yaml

from brawl_sim.config import build_params, load_config
from brawl_sim.constants import TILE_BLOCKS_PROJ, AimModel, Kind, Proj, ProjClass, Tile
from brawl_sim.core import projectiles as proj
from brawl_sim.core.state import allocate

CONFIGS_DEFAULT = "configs/default.yaml"
BULL = int(Kind.BOT_BULL)
SHELLY = int(Kind.BOT_RIFLE)
PELLET_DAMAGE = 880.0


@dataclass
class _FakeBank:
    blocks_proj: torch.Tensor


def _setup(map_h=40, map_w=40):
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
    # Everyone parked in a corner unless a test places them, so no bystander eats a stray pellet.
    state.ent_pos.fill_(float(map_w) - 2.0)
    state.map_id.fill_(0)

    tiles = torch.full((map_h, map_w), int(Tile.FLOOR), dtype=torch.int64)
    tiles[0, :] = tiles[-1, :] = tiles[:, 0] = tiles[:, -1] = int(Tile.WALL)
    return cfg, params, state, _FakeBank(blocks_proj=TILE_BLOCKS_PROJ[tiles].unsqueeze(0))


def _volley(state, cfg, params, kind, origin, aim):
    """Fires one full volley from entity 1, aimed down +x at `aim`."""
    k = int(kind)
    state.ent_kind[0, 1] = k
    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 1] = True
    org = torch.zeros(1, cfg.n_entities, 2)
    org[0, 1] = torch.tensor(origin)
    tgt = torch.zeros(1, cfg.n_entities, 2)
    tgt[0, 1] = torch.tensor(aim)
    damage = params.base_damage[:, k].unsqueeze(-1).expand(1, cfg.n_entities).clone()
    kinds = torch.full((1, cfg.n_entities), k, dtype=torch.int64)
    proj.spawn_volley(state, fire, org, tgt - org, tgt, kinds, damage, params, cfg)


def _damage_at(kind, dist, lateral=0.0, origin=(10.0, 20.0)):
    """Damage a victim `dist` ahead (and `lateral` to the side) takes from one volley."""
    cfg, params, state, bank = _setup()
    state.ent_pos[0, 0] = torch.tensor([origin[0] + dist, origin[1] + lateral])
    _volley(state, cfg, params, kind, origin, (origin[0] + dist, origin[1]))
    total = 0.0
    for _ in range(150):
        if not bool(state.prj_alive.any()):
            break
        dmg_ent, _, _, _ = proj.step_projectiles(state, bank, params, cfg)
        total += float(dmg_ent[0, 0])
    return total


def _bearings(state):
    live = state.prj_alive[0]
    return sorted(math.degrees(math.atan2(v[1], v[0])) for v in state.prj_vel[0][live].tolist())


# ---- the volley -----------------------------------------------------------------

def test_one_ammo_fires_five_880_pellets():
    """"Each use of ammo from his shotgun fires 5 rounds, which each deal 880 damage" -- so a
    point-blank volley is 4400, the largest single burst in the roster."""
    cfg, params, state, bank = _setup()
    _volley(state, cfg, params, Kind.BOT_BULL, (10.0, 20.0), (17.0, 20.0))

    live = state.prj_alive[0]
    assert int(live.sum()) == 5
    assert torch.allclose(state.prj_damage[0][live], torch.full((5,), PELLET_DAMAGE))
    assert abs(_damage_at(Kind.BOT_BULL, 1.5) - 5 * PELLET_DAMAGE) < 1e-3


def test_the_pellets_are_bull_slugs_not_shelly_arrows():
    """A separate `Proj` member is the only structural change Bull needed, and no behavioural test
    can catch its absence -- the pellets would fly identically. It matters because
    configs/agent_obs.yaml feeds the policy `projectiles.kind_onehot` and NOT
    `projectiles.owner_kind`, so this enum value is the agent's only categorical way to tell an
    880-per-pellet volley from Shelly's 600 before it lands."""
    cfg, params, state, bank = _setup()
    _volley(state, cfg, params, Kind.BOT_BULL, (10.0, 20.0), (17.0, 20.0))

    live = state.prj_alive[0]
    assert int(state.prj_kind[0][live][0]) == int(Proj.BULL_SLUG)
    assert int(Proj.BULL_SLUG) != int(Proj.RIFLE_ARROW)
    # Ordinary pellets: they damage in flight and die on the first thing they touch.
    assert bool((state.prj_class[0][live] == int(ProjClass.PROJECTILE)).all())
    assert bool((state.prj_aoe[0][live] == 0).all())
    assert int(params.split_count[0, BULL]) == 0, "Bull does not split"


def test_the_fan_is_a_total_span_of_half_a_radian():
    """`proj_spread_rad` is the TOTAL tip-to-tip span in RADIANS, not a half-angle and not degrees.
    Pinned explicitly because that exact confusion is what put "60 degrees" in Shelly's docs for a
    0.6 rad fan -- see test_shellys_fan_is_34_degrees_not_60."""
    cfg, params, state, bank = _setup()
    _volley(state, cfg, params, Kind.BOT_BULL, (10.0, 20.0), (17.0, 20.0))

    spread = float(params.proj_spread_rad[0, BULL])
    bearings = _bearings(state)
    assert abs(math.radians(bearings[-1] - bearings[0]) - spread) < 1e-4
    assert abs((bearings[-1] - bearings[0]) - 28.65) < 0.05, "expected ~28.6 deg tip to tip"
    assert abs(bearings[2]) < 1e-4, "the middle pellet must fly straight down the aim line"


# ---- the ways he is deliberately not Shelly -------------------------------------

def test_bull_hits_harder_than_shelly_at_every_range_he_can_reach():
    """Same pellet count, more damage each: 4400 against 3000 point-blank. This is the trade for
    the shorter range, and it should hold at every distance where both connect."""
    for dist in (1.0, 2.0, 4.0, 6.0):
        bull, shelly = _damage_at(Kind.BOT_BULL, dist), _damage_at(Kind.BOT_RIFLE, dist)
        assert shelly > 0, f"Shelly whiffed at {dist} -- the comparison is meaningless"
        assert bull > shelly, f"at {dist} tiles Bull dealt {bull} against Shelly's {shelly}"


def test_bulls_reach_is_exactly_one_tile_shorter():
    """"1 tile shorter range." Asserted as a REACH, at the distance where each stops connecting --
    the pellet dies after `attack_range` of travel but still damages within its hit radius, so the
    observable cutoff sits about 0.7 tiles past the configured range for both of them."""
    assert float(params_of(BULL, "attack_range")) == float(params_of(SHELLY, "attack_range")) - 1.0

    assert _damage_at(Kind.BOT_BULL, 7.5) > 0        # inside Bull's reach
    assert _damage_at(Kind.BOT_BULL, 8.5) == 0       # past it
    assert _damage_at(Kind.BOT_RIFLE, 8.5) > 0       # Shelly still connects a full tile further


def test_bulls_fan_is_tighter_than_shellys():
    """"Slightly less width to his circle." A narrower span means more of the volley stays on a
    target that is not dead-centre, which is what makes his shorter range survivable."""
    cfg, params, _, _ = _setup()
    assert float(params.proj_spread_rad[0, BULL]) < float(params.proj_spread_rad[0, SHELLY])

    # At 4 tiles and half a tile off-axis, Bull still lands 3 pellets where Shelly is down to 2.
    assert _damage_at(Kind.BOT_BULL, 4.0, lateral=0.5) == _damage_at(Kind.BOT_BULL, 4.0)
    assert _damage_at(Kind.BOT_RIFLE, 4.0, lateral=0.5) < _damage_at(Kind.BOT_RIFLE, 4.0)


def test_bull_is_slower_and_tankier_than_shelly():
    cfg, params, _, _ = _setup()
    assert float(params.move_speed[0, BULL]) < float(params.move_speed[0, SHELLY])
    assert float(params.base_hp[0, BULL]) == 10000.0
    assert float(params.base_hp[0, BULL]) > float(params.base_hp[0, SHELLY])
    # 2.0 s per ammo point, and the cooldown freezes reload, so sustained is 0.25 + 2.0.
    assert float(params.reload_seconds[0, BULL]) == 2.0
    assert float(params.reload_seconds[0, BULL]) > float(params.reload_seconds[0, SHELLY])


def test_bull_fights_closer_in_than_shelly():
    """`desired_range_fraction` is what a KITE bot of this kind holds. A 10000 HP brawler with a
    7-tile shotgun should not be hanging back at the edge of its reach."""
    cfg, params, _, _ = _setup()
    bull_tiles = float(params.desired_range_fraction[0, BULL] * params.attack_range[0, BULL])
    shelly_tiles = float(params.desired_range_fraction[0, SHELLY] * params.attack_range[0, SHELLY])
    assert bull_tiles < shelly_tiles, (
        f"Bull holds {bull_tiles:.1f} tiles against Shelly's {shelly_tiles:.1f}"
    )


def test_bull_shares_shellys_fire_rule_shape():
    """He is the same archetype, so the fire rule should be hers, not a second set of hand-tuned
    thresholds. If this ever diverges it should be a deliberate balance decision, not drift."""
    cfg, params, _, _ = _setup()
    for field in ("fire_needs_los", "fire_range_fraction", "fire_lateral_speed_limit",
                  "fire_lateral_hold_range_fraction", "aim_model"):
        assert float(getattr(params, field)[0, BULL]) == float(getattr(params, field)[0, SHELLY]), \
            f"{field} diverged from Shelly's"
    assert int(params.aim_model[0, BULL]) == int(AimModel.LEAD)


# ---- the documentation bug this brawler surfaced --------------------------------

def test_shellys_fan_is_34_degrees_not_60():
    """A regression test for a DOC bug, kept because it bit twice. `proj_spread_rad: 0.6` was
    written up as "60 degrees total (30 either side)" in both CHARACTER_DETAILS.md and the yaml's
    own comment -- 0.6 radians misread as a degree figure. It is 34.4 degrees, and the value was
    hand-tuned rather than converted, so the docs were the wrong half of the pair to trust."""
    cfg, params, state, bank = _setup()
    _volley(state, cfg, params, Kind.BOT_RIFLE, (10.0, 20.0), (18.0, 20.0))

    bearings = _bearings(state)
    span = bearings[-1] - bearings[0]
    assert abs(span - 34.38) < 0.05, f"Shelly's fan spans {span:.2f} deg"
    assert abs(bearings[-1] - 17.19) < 0.05, "the outermost pellet sits at +17.2 deg, not +30"


def params_of(kind_index, field):
    cfg, params, _, _ = _setup()
    return getattr(params, field)[0, kind_index]
