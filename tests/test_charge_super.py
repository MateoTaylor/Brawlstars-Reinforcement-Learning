"""Bull's super, Bulldozer: the CHARGE super type (BULL_HERO_PLAN.md Phase 4, D2-D4).

Its own file, as Mortis's bolt has test_super.py: the charge spans hero (start_charge,
advance_dash), status (airborne, slow), env (the fire and the meter) and projectiles (the meter
counts pellets). Literals, not params. The charge stands still 0.20 s, then runs 11.0 tiles at
6.7 tiles/s, so 0.20 + 11 / 6.7 = 1.842 s = 37 ticks of 0.05 s: 4 standing, 32 of 0.335 tiles and
a last one of 0.28. It hits players whose centre comes within 1.0 tile of his, once each for a flat
2000, pushes them 2.5 tiles and launches them for 0.5 s. Where he lands, everyone else within 3.0
tiles is slowed 40 % for 1.5 s. Nine pellet hits fill the meter. A unit body is 0.4 tiles.
"""
import math
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import _deep_merge, build_params, load_config, validate
from brawl_sim.constants import TILE_BLOCKS_UNIT, Kind, Tile
from brawl_sim.core import hero, status, terrain
from brawl_sim.core.state import allocate
from brawl_sim.env import BrawlVecEnv

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
TINY = yaml.safe_load((CONFIGS / "presets" / "debug_tiny.yaml").read_text())
BULL = {"entities": {"hero": "bull"}}
MORTIS = {"entities": {"hero": "mortis"}}

_ATTACK, _SUPER, _GADGET = 1, 2, 3
_EAST, _PLUS_Y, _WEST, _MINUS_Y = 1, 5, 9, 13


def _spec(overrides=None) -> dict:
    spec = {
        **yaml.safe_load((CONFIGS / "default.yaml").read_text()),
        **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text()),
    }
    return _deep_merge(spec, overrides or {})


# ---- direct: start_charge + advance_dash on a hand-built map --------------------------------

class _FakeBank:
    def __init__(self, tiles):
        self.blocks_unit = TILE_BLOCKS_UNIT[tiles].unsqueeze(0)


def _open_map(h=30, w=30):
    """Floor inside a wall ring. tiles[y, x]."""
    tiles = torch.full((h, w), int(Tile.FLOOR), dtype=torch.int64)
    tiles[0, :] = tiles[-1, :] = tiles[:, 0] = tiles[:, -1] = int(Tile.WALL)
    return tiles


def _cfg_and_params(hero_overrides=BULL, spec_overrides=None, world=True):
    overrides = _deep_merge({"world": {"map_h": 30, "map_w": 30}} if world else {}, hero_overrides)
    cfg = load_config(CONFIGS / "default.yaml", overrides=overrides)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    spec = _spec(_deep_merge(overrides, spec_overrides or {}))
    return cfg, build_params(cfg, n_envs=1, device="cpu", gen=gen, spec=spec)


def _direct(tiles=None, victims=()):
    """Bull at (5, 15) on a 30x30 map, every bot a huge-HP sniper parked along y = 3, far from his
    line, except the `victims` [(slot, (x, y))] placed by hand."""
    cfg, params = _cfg_and_params()
    state = allocate(cfg, n_envs=1, device="cpu", verbose=False)
    state.map_id.fill_(0)
    state.ent_alive.fill_(True)
    state.ent_kind[0, 0] = int(Kind.HERO)
    state.ent_kind[0, 1:] = int(Kind.BOT_SNIPER)
    state.ent_max_hp.fill_(1e9)
    state.ent_hp.fill_(1e9)
    for e in range(1, cfg.n_entities):
        state.ent_pos[0, e] = torch.tensor([2.0 + 2.0 * e, 3.0])
    state.ent_pos[0, 0] = torch.tensor([5.0, 15.0])
    for slot, xy in victims:
        state.ent_pos[0, slot] = torch.tensor(xy)
    return cfg, params, state, _FakeBank(_open_map() if tiles is None else tiles)


def _charge(cfg, params, state, bank, direction=(1.0, 0.0), ticks=40):
    """Fire Bull's charge from where he stands and run `ticks` dash ticks. Returns the summed
    dmg_by (E,E) and dmg_box (B,), and the tick the charge ended on."""
    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    aim = torch.zeros(1, cfg.n_entities, 2)
    aim[0, 0] = torch.tensor(direction)
    hero.start_charge(state, fire, aim, bank, params, cfg)
    dmg_by = torch.zeros(cfg.n_entities, cfg.n_entities)
    dmg_box = torch.zeros(cfg.max_boxes)
    ended = None
    for tick in range(1, ticks + 1):
        _, by, box = hero.advance_dash(state, params, cfg, bank)
        dmg_by += by[0]
        dmg_box += box[0]
        if ended is None and float(state.ent_dash_t[0, 0]) == 0:
            ended = tick
    return dmg_by, dmg_box, ended


@pytest.mark.parametrize("victim, pushed_to", [
    ((10.0, 15.0), (12.5, 15.0)),   # head-on: contact as his centre reaches 9, pushed along the charge
    ((10.0, 15.6), (12.0, 17.1)),   # glancing: contact at (9.2, 15), pushed along (0.8, 0.6)
    ((5.0, 15.0), (7.5, 15.0)),     # dead on his start: no away, so along the charge
])
def test_the_push_is_2_5_tiles_straight_away_from_bulls_centre_at_contact(victim, pushed_to):
    cfg, params, state, bank = _direct(victims=[(1, victim)])
    dmg_by, _, ended = _charge(cfg, params, state, bank)
    assert ended == 37
    assert dmg_by[0, 1].item() == 2000.0, "hit once, for the flat 2000"
    assert torch.allclose(state.ent_pos[0, 1], torch.tensor(pushed_to), atol=1e-4)
    assert float(state.ent_airborne_t[0, 1]) == 0.5


def test_a_graze_exactly_one_tile_off_his_line_is_pushed_straight_across_it():
    """Players exactly 1.0 off his line are touched at the tangent, so each goes straight across
    it, 2.5 tiles, with nothing along the charge. In float32 nearly half such grazes pass the
    capsule test yet find no crossing of the circle (the discriminant rounds below zero). Their
    contact is then the closest approach; the tick's start would tilt the push up to 18 degrees
    toward his heading (atan of 0.335 / 1.0)."""
    xs = [6.0 + 1.07 * k for k in range(9)]
    cfg, params, state, bank = _direct(victims=[(1 + k, (x, 16.0)) for k, x in enumerate(xs)])
    dmg_by, _, _ = _charge(cfg, params, state, bank)
    assert dmg_by[0, 1:].tolist() == [2000.0] * 9
    assert torch.allclose(state.ent_pos[0, 1:], torch.tensor([[x, 18.5] for x in xs]), atol=2e-3)


def test_a_wall_stops_the_push_where_the_body_first_touches_it_and_does_not_slide_it():
    """A player 0.5 off the line is met at (9.134, 15) and pushed along (0.866, 0.5). A wall from
    y = 16 sits 0.1 above its body, so it stops 0.1 / 0.5 = 0.2 tiles along the push, within the
    march's 0.03-tile refinement, and still on the push's own line."""
    tiles = _open_map()
    tiles[16, :] = int(Tile.WALL)
    cfg, params, state, bank = _direct(tiles, victims=[(1, (10.0, 15.5))])
    start = state.ent_pos[0, 1].clone()
    _charge(cfg, params, state, bank)
    moved = state.ent_pos[0, 1] - start
    away = torch.tensor([math.sqrt(0.75), 0.5])
    assert 0.16 < float(moved @ away) <= 0.2
    assert abs(float(moved[0] * away[1] - moved[1] * away[0])) < 1e-5, "slid along the wall"
    assert not bool(terrain.circle_blocked(bank.blocks_unit, state.map_id, state.ent_pos[:, 1],
                                           params.unit_radius, cfg).any())


def test_a_charge_into_a_wall_stops_short_of_it_and_ends_sooner():
    """D3 (user, 2026-10-23): the first wall stops him, as it stops Mortis's dash. A wall column
    from x = 12 meets his body at a centre of 11.6, so he lands just short of it on tick 24 (0.20 +
    6.6 / 6.7 = 1.18 s) at full speed, rather than slowing to fill 37 ticks."""
    tiles = _open_map()
    tiles[:, 12] = int(Tile.WALL)
    cfg, params, state, bank = _direct(tiles)
    _, _, ended = _charge(cfg, params, state, bank)
    assert ended == 24
    assert 11.55 < float(state.ent_pos[0, 0, 0]) < 11.6
    assert float(state.ent_pos[0, 0, 1]) == 15.0
    assert not bool(terrain.circle_blocked(bank.blocks_unit, state.map_id, state.ent_pos[:, 0],
                                           params.unit_radius, cfg).any())
    assert not bool(state.ent_dash_super.any()) and float(state.ent_dash_windup_t.abs().sum()) == 0


def test_the_charge_passes_crates_untouched():
    """As the bolt does. The crate sits 0.5 off his line, inside the 1.0 reach."""
    cfg, params, state, bank = _direct()
    state.box_alive.fill_(False)
    state.box_alive[0, 0] = True
    state.box_pos[0, 0] = torch.tensor([10.5, 15.5])
    _, dmg_box, ended = _charge(cfg, params, state, bank)
    assert ended == 37
    assert float(dmg_box.sum()) == 0.0
    assert abs(float(state.ent_pos[0, 0, 0]) - 16.0) < 1e-4, "the crate stopped him"


def test_a_charge_spec_must_hand_advance_dash_the_bank_and_mortis_has_no_charge():
    """The charge is behind a Python switch on charge_ray_tiles, the longest rush in the spec: 11.0
    with Bull, 0 with Mortis, whose super is a bolt in every column."""
    cfg, params, state, _ = _direct()
    assert params.charge_ray_tiles == 11.0 and params.knockback_ray_tiles == 2.5
    with pytest.raises(ValueError, match="bank"):
        hero.advance_dash(state, params, cfg)
    _, mortis = _cfg_and_params(MORTIS)
    assert mortis.charge_ray_tiles == 0 and mortis.knockback_ray_tiles == 0
    assert int(mortis.super_type.max()) == 0


def test_validate_refuses_a_charge_that_cannot_move_and_a_bolt_with_a_charge_speed():
    for hero_name, block, match in (
        (BULL, {"hero_bull": {"super_charge_speed": 0}}, "CHARGE but super_charge_speed is 0"),
        (MORTIS, {"hero_mortis": {"super_charge_speed": 5.0}}, "super_type is BOLT"),
    ):
        cfg, params = _cfg_and_params(hero_name, block, world=False)
        with pytest.raises(ValueError, match=match):
            validate(cfg, params)
    cfg, params = _cfg_and_params(BULL, world=False)
    validate(cfg, params)


# ---- through the env: the fire, the timing, the meter -----------------------------------------
# A 3-entity tiny env, one tick per step, both bots frozen by the override (the hero row's -1 lets
# the action through) and regen off, so HP read before the fire step counts every hit.

def _duel(bull_at=(4.0, 10.0), bot1=(3.0, 3.0), bot2=(3.0, 17.0), facing=0.0, debug=False):
    over = _deep_merge(TINY, _deep_merge(BULL, {"sim": {"action_repeat": 1},
                                                "regen": {"enabled": False},
                                                "engine": {"debug_checks": debug}}))
    cfg = load_config(CONFIGS / "default.yaml", overrides=over)
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, spec=_spec(over), verbose=False)
    env.reset()
    st = env.state
    assert env.cfg.n_entities == 3
    for slot, xy in enumerate((bull_at, bot1, bot2)):
        st.ent_pos[0, slot] = torch.tensor(xy)
    st.ent_facing.zero_()
    st.ent_facing[0, 0] = facing
    st.box_alive.fill_(False)
    override = torch.zeros(1, env.cfg.n_entities, 2, dtype=torch.int64)
    override[:, 0, 0] = -1
    return env, override


def _step(env, override, attack=0, move=0):
    return env.step(torch.tensor([[move, attack]], dtype=torch.int64), override)


def _attack_mask(env):
    return hero.action_mask(env.state, env.params, env.cfg)["attack"][0]


def test_the_charge_stands_still_for_the_wind_up_then_runs_11_tiles_in_37_ticks():
    """A +y move bin held all the way neither walks nor steers him, and every attack is masked
    until he lands."""
    env, override = _duel()
    st = env.state
    st.ent_super_charge[0, 0] = 9
    _step(env, override, _SUPER, move=_EAST)        # a moving super follows the move bin
    for tick in range(1, 38):
        if tick > 1:
            _step(env, override, move=_PLUS_Y)
        x, y = float(st.ent_pos[0, 0, 0]), float(st.ent_pos[0, 0, 1])
        assert y == 10.0, f"tick {tick}: walked or steered"
        if tick <= 4:
            assert x == 4.0, f"tick {tick}: moved in the wind-up"
        elif tick < 37:
            assert abs(x - (4.0 + 0.335 * (tick - 4))) < 1e-4, f"tick {tick}"
        if tick < 37:
            assert float(st.ent_dash_t[0, 0]) > 0, f"ended on tick {tick}"
            assert not bool(_attack_mask(env)[1:].any()), f"tick {tick}: an attack opened mid-charge"
    assert float(st.ent_dash_t[0, 0]) == 0
    assert abs(x - 15.0) < 1e-4
    assert bool(_attack_mask(env)[1]), "landed and still masked"


def test_nothing_is_hit_in_the_wind_up_and_each_player_once_for_a_flat_2000():
    """Bot 1 stands 0.9 tiles ahead, inside the reach from the start: untouched for the 4 standing
    ticks, hit on the first moving one. Bot 2 is 0.5 off the line 5 tiles on. Each loses exactly
    2000 over the whole charge, Bull's 6 power cubes notwithstanding."""
    env, override = _duel(bot1=(4.9, 10.0), bot2=(9.0, 10.5))
    st = env.state
    st.ent_cubes[0, 0] = 6
    st.ent_super_charge[0, 0] = 9
    hp1, hp2 = float(st.ent_hp[0, 1]), float(st.ent_hp[0, 2])
    _step(env, override, _SUPER, move=_EAST)
    for _ in range(3):
        _step(env, override)
    assert float(st.ent_hp[0, 1]) == hp1, "hit in the wind-up"
    _step(env, override)
    assert hp1 - float(st.ent_hp[0, 1]) == 2000.0, "not hit on the first moving tick"
    for _ in range(40):
        _step(env, override)
    assert hp1 - float(st.ent_hp[0, 1]) == 2000.0
    assert hp2 - float(st.ent_hp[0, 2]) == 2000.0


def test_a_player_the_charge_hits_is_airborne_for_10_ticks():
    """0.5 s. Bot 1 at (9, 10) is met as Bull's centre reaches 8, on tick 16."""
    env, override = _duel(bot1=(9.0, 10.0))
    st = env.state
    st.ent_super_charge[0, 0] = 9
    _step(env, override, _SUPER, move=_EAST)
    airborne = []
    for tick in range(2, 41):
        _step(env, override)
        if bool(status.airborne(st)[0, 1]):
            airborne.append(tick)
    assert airborne == list(range(16, 26))


@pytest.mark.parametrize("off, slowed", [(2.9, True), (3.1, False)])
def test_the_landing_slows_everyone_within_3_tiles_airborne_included(off, slowed):
    """He lands at (15, 10). Bot 1 stood on the line at (13, 10): hit as his centre reached 12 on
    tick 28, pushed to 15.5, and still in the air on tick 37 when he lands 0.5 tiles from it. Bot 2
    stands `off` tiles north of the landing, well off the line."""
    env, override = _duel(bot1=(13.0, 10.0), bot2=(15.0, 10.0 + off))
    st = env.state
    st.ent_super_charge[0, 0] = 9
    _step(env, override, _SUPER, move=_EAST)
    for _ in range(35):
        _step(env, override)
    assert float(st.ent_dash_t[0, 0]) > 0 and float(st.ent_slow_t.abs().sum()) == 0, "slowed early"
    _step(env, override)                             # tick 37: the landing
    assert float(st.ent_dash_t[0, 0]) == 0
    assert bool(status.airborne(st)[0, 1])
    assert float(st.ent_slow_t[0, 1]) == 1.5
    assert float(st.ent_slow_frac[0, 1]) == pytest.approx(0.4)
    assert (float(st.ent_slow_t[0, 2]) == 1.5) == slowed
    assert float(st.ent_slow_t[0, 0]) == 0, "slowed the charger himself"


def test_nine_pellets_fill_the_meter_and_the_charges_own_hit_adds_nothing():
    """The user's count (2026-10-23): two full volleys, or 9 pellet hits. Bot 1 stands 2 tiles
    ahead, where all five pellets of an idle volley along the facing land: 5, then 9 (capped).
    The super, aimed at bot 1 by the idle rule, takes 2000 off it and puts nothing back."""
    env, override = _duel(bot1=(6.0, 10.0))
    st = env.state
    for volley, meter in ((1, 5), (2, 9)):
        st.ent_hp[0, 1] = st.ent_max_hp[0, 1]
        _step(env, override, _ATTACK)
        for _ in range(12):
            _step(env, override)
        assert int(st.ent_super_charge[0, 0]) == meter, f"after volley {volley}"
    assert bool(_attack_mask(env)[2]), "a full meter did not open the super"
    st.ent_hp[0, 1] = st.ent_max_hp[0, 1]
    before = float(st.ent_hp[0, 1])
    _step(env, override, _SUPER)
    for _ in range(40):
        _step(env, override)
    assert before - float(st.ent_hp[0, 1]) == 2000.0
    assert int(st.ent_super_charge[0, 0]) == 0


@pytest.mark.parametrize("dx, at_it", [(11.9, True), (12.1, False)])
def test_the_idle_super_aims_at_an_enemy_within_12_tiles_and_along_the_facing_past_it(dx, at_it):
    """11.0 of rush and 1.0 of reach, on centres: no body term, unlike the bolt's reach."""
    env, override = _duel(bull_at=(4.0, 4.0), bot1=(4.0 + dx, 4.0), bot2=(17.0, 18.0),
                          facing=math.pi / 2)
    env.state.ent_super_charge[0, 0] = 9
    _step(env, override, _SUPER)
    want = [1.0, 0.0] if at_it else [0.0, 1.0]
    assert torch.allclose(env.state.ent_dash_dir[0, 0], torch.tensor(want), atol=1e-5)


def test_the_state_invariants_hold_through_charges_in_every_direction():
    """engine.debug_checks runs state.check_invariants after every step: the charge's dash_t
    bound, its flag and its wind-up, against walls on two of the four."""
    env, override = _duel(debug=True)
    for move in (_EAST, _PLUS_Y, _WEST, _MINUS_Y):
        env.state.ent_super_charge[0, 0] = 9
        _step(env, override, _SUPER, move=move)
        assert bool(env.state.ent_dash_super[0, 0])
        for _ in range(40):
            _step(env, override)
        assert not bool(env.state.ent_dash_super[0, 0])


# ---- the charger's window, obs `hero.slow_window` (D6, Phase 6a) ----------------------------
# His copy of his landing slow's clock: 1.5 s = 30 ticks from the landing tick, counted down beside
# the slow itself. A stomp stuns the players a slow still holds (test_stomper.py), so the window
# tells the policy whether a stomp will stun without the obs saying whom the landing caught.

def test_the_window_is_the_landing_slow_on_every_player_it_caught_tick_for_tick():
    """Bot 1 stood on his line at (13, 10): hit on tick 28 and still in the air when he lands at
    (15, 10) on tick 37. Bot 2 stands 2.9 tiles north of the landing. The window is 0 until tick
    37 and 1.5 on it, then equals both bots' slow on every tick, all three reaching 0 on tick 67;
    the obs reads it as the state holds it throughout."""
    env, override = _duel(bot1=(13.0, 10.0), bot2=(15.0, 12.9), debug=True)
    st = env.state
    st.ent_super_charge[0, 0] = 9
    obs, *_ = _step(env, override, _SUPER, move=_EAST)
    for tick in range(1, 71):
        if tick > 1:
            obs, *_ = _step(env, override)
        window = float(st.ent_slow_window_t[0, 0])
        assert float(obs["hero"]["slow_window"][0]) == window, f"tick {tick}"
        if tick < 37:
            assert window == 0, f"tick {tick}: open before the landing"
            continue
        if tick == 37:
            assert window == 1.5
        assert window == float(st.ent_slow_t[0, 1]) == float(st.ent_slow_t[0, 2]), f"tick {tick}"
        assert (window > 0) == (tick < 67), f"tick {tick}"


def test_a_landing_that_catches_nobody_opens_the_window_all_the_same():
    """The live shadow knows when its charge landed, never whom it slowed, so the window does not
    ask: both bots stand far from the landing at (15, 10), nobody is slowed, and it opens at 1.5."""
    env, override = _duel(debug=True)
    st = env.state
    st.ent_super_charge[0, 0] = 9
    _step(env, override, _SUPER, move=_EAST)
    for _ in range(36):
        _step(env, override)
    assert float(st.ent_dash_t[0, 0]) == 0
    assert float(st.ent_slow_window_t[0, 0]) == 1.5
    assert float(st.ent_slow_t.abs().sum()) == 0


@pytest.mark.parametrize("idle, window, stunned", [(22, 0.40, True), (23, 0.35, False)])
def test_a_stomp_tapped_after_a_window_over_0_35_stuns_and_one_after_0_35_slows(idle, window, stunned):
    """The rule `hero.slow_window` carries: the tap's own tick and the 6 to the pulse count the slow
    down 7 more times, 0.35 s, so a stomp tapped right after an obs reading w lands inside the
    landing's slow only when w > 0.35. Bot 2 stands 2 tiles north of the landing at (15, 10),
    inside both the landing's 3.0 tiles and the pulse's 2.67. Read 22 ticks after the landing the
    window is 0.40, and the pulse on tick 66 stuns bot 2 with 0.05 of its slow left; read 23 ticks
    after, it is 0.35, the slow ends on tick 67 as the pulse lands, and the pulse slows it afresh."""
    env, override = _duel(bot2=(15.0, 12.0), debug=True)
    st = env.state
    st.ent_super_charge[0, 0] = 9
    _step(env, override, _SUPER, move=_EAST)
    for _ in range(36 + idle):
        obs, *_ = _step(env, override)
    assert float(obs["hero"]["slow_window"][0]) == pytest.approx(window, abs=1e-5)
    _step(env, override, _GADGET)
    for _ in range(6):
        _step(env, override)
    assert float(st.ent_stun_t[0, 2]) == pytest.approx(1.4 if stunned else 0.0)
    assert float(st.ent_slow_t[0, 2]) == pytest.approx(0.05 if stunned else 1.0, abs=1e-5)
    assert float(st.ent_slow_frac[0, 2]) == pytest.approx(0.4 if stunned else 0.5)
