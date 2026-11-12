"""Bull as the agent (BULL_HERO_PLAN.md). Phase 1: `entities.hero` swaps which brawlers.yaml block
fills Kind.HERO's column, and nothing else moves. Phase 2: his volley goes down the move bin
(value 1) or at the auto-aim target (value 4). Mortis staying bit-identical is checked by a
whole-rollout digest when the seam changes, not here: these pin the seam's own contract."""
import math
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import (
    BOT_YAML_NAMES, _deep_merge, build_params, kind_yaml_names, load_config, spec_hero, validate,
)
from brawl_sim.constants import Kind, Proj
from brawl_sim.env import BrawlVecEnv

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
TINY = yaml.safe_load((CONFIGS / "presets" / "debug_tiny.yaml").read_text())
BULL = {"entities": {"hero": "bull"}}


def _spec(overrides=None) -> dict:
    spec = {
        **yaml.safe_load((CONFIGS / "default.yaml").read_text()),
        **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text()),
    }
    return _deep_merge(spec, overrides or {})


def _params(overrides=None):
    cfg = load_config(CONFIGS / "default.yaml", overrides=overrides)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=4, device="cpu", gen=gen, spec=_spec(overrides))
    validate(cfg, params)
    return cfg, params


def _tiny_env(overrides=None, n_envs=4):
    merged = _deep_merge(TINY, overrides or {})
    cfg = load_config(CONFIGS / "default.yaml", overrides=merged)
    return BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=0, spec=_spec(merged), verbose=False)


def test_the_hero_names_row_0_and_the_bots_keep_theirs():
    assert kind_yaml_names() == ("hero_mortis",) + BOT_YAML_NAMES
    assert kind_yaml_names("bull") == ("hero_bull",) + BOT_YAML_NAMES
    assert len(kind_yaml_names("bull")) == len(Kind)
    assert spec_hero({}) == "mortis", "a spec that names no hero plays Mortis"
    assert spec_hero(_spec(BULL)) == "bull"


def test_entities_hero_bull_fills_column_0_from_hero_bull_and_leaves_every_bot_column_alone():
    """The row swap: column 0 reads hero_bull's block, and columns 1..7 are the same tensors a
    Mortis run builds. Read off the BUILT params so a misspelled key in the block fails here."""
    _, mortis = _params()
    cfg, bull = _params(BULL)
    assert cfg.hero == "bull"
    h = int(Kind.HERO)
    block = _spec()["hero_bull"]
    assert float(bull.base_hp[0, h]) == block["base_hp"] == 10000
    assert float(bull.base_damage[0, h]) == block["base_damage"] == 880
    assert int(bull.proj_count[0, h]) == 5
    assert int(bull.proj_kind[0, h]) == int(Proj.BULL_SLUG)
    assert abs(float(bull.move_speed[0, h]) - block["move_speed"]) < 1e-6
    # No dash is what routes his attack to spawn_volley. The super is Phase 4's charge (two full
    # volleys of pellets fill it, the user's count).
    assert float(bull.dash_distance[0, h]) == 0
    assert float(bull.long_dash_seconds[0, h]) == 0
    assert int(bull.super_charge_hits[0, h]) == 9
    # Phase 5: Stomper and Berserker (user decisions 2026-10-23); the wind-up and the stun as
    # measured 2026-11-01 (Phase 0 d, e).
    assert int(bull.gadget_type[0, h]) == 1, "STOMP"
    assert float(bull.gadget_cooldown[0, h]) == 20.0
    assert abs(float(bull.stomp_radius[0, h]) - 2.67) < 1e-6
    assert abs(float(bull.stomp_windup_seconds[0, h]) - 0.30) < 1e-6
    assert float(bull.stomp_slow_frac[0, h]) == 0.5
    assert float(bull.stomp_slow_seconds[0, h]) == 1.0
    assert abs(float(bull.stomp_stun_seconds[0, h]) - 1.4) < 1e-6
    assert abs(float(bull.reload_boost_below_hp_frac[0, h]) - 0.6) < 1e-6
    assert float(bull.reload_boost_mult[0, h]) == 2.0
    for attr in mortis.__slots__:
        a, b = getattr(mortis, attr), getattr(bull, attr)
        if isinstance(a, torch.Tensor) and a.dim() == 2:
            assert torch.equal(a[:, 1:], b[:, 1:]), f"{attr}: a bot column moved with the hero"


def test_a_named_hero_must_have_a_block_and_a_known_name():
    spec = _spec(BULL)
    del spec["hero_bull"]
    cfg = load_config(CONFIGS / "default.yaml", overrides=BULL)
    gen = torch.Generator(device="cpu")
    with pytest.raises(ValueError, match="no 'hero_bull' block"):
        build_params(cfg, n_envs=2, device="cpu", gen=gen, spec=spec)
    with pytest.raises(ValueError, match="entities.hero must be one of"):
        kind_yaml_names("shelly")


def test_cfg_and_spec_naming_different_heroes_is_refused():
    """BrawlVecEnv's docstring trap, for the one key both halves read: a cfg loaded with the
    override and a spec built without it would train Mortis's stats under Bull's name."""
    cfg = load_config(CONFIGS / "default.yaml", overrides=_deep_merge(TINY, BULL))
    with pytest.raises(ValueError, match="entities.hero"):
        BrawlVecEnv(cfg, n_envs=2, device="cpu", seed=0, spec=_spec(TINY), verbose=False)


def test_bulls_attack_fires_his_volley_and_never_dashes():
    """Value 1 with no dash_distance takes the ranged branch of env._attack_phase: five BULL_SLUG
    pellets owned by slot 0, and the dash timer never starts. No crates: at 9 tiles/s a pellet
    covers 2.25 tiles in the step, and a crate that close takes three of the five."""
    env = _tiny_env(BULL)
    env.reset()
    env.state.box_alive.fill_(False)
    attack = torch.zeros(env.n_envs, 2, dtype=torch.int64)
    attack[:, 1] = 1
    env.step(attack)
    s = env.state
    mine = s.prj_alive & (s.prj_owner == 0)
    assert bool((s.prj_kind[mine] == int(Proj.BULL_SLUG)).all())
    assert torch.all(mine.sum(dim=1) == 5), "one volley of five pellets per env"
    assert torch.all(s.ent_dash_t[:, 0] == 0)
    for _ in range(8):
        obs, *_ = env.step(attack)
        assert all(bool(torch.isfinite(v).all()) for v in _float_leaves(obs)), \
            "a Mortis-only field (dash, long dash) divided by Bull's 0"


def _float_leaves(obs):
    if isinstance(obs, dict):
        for v in obs.values():
            yield from _float_leaves(v)
    elif isinstance(obs, torch.Tensor) and obs.is_floating_point():
        yield obs


def test_the_heros_weapon_is_bot_bulls_but_for_the_measured_reload():
    """One brawler, two blocks: his gun and clip started as bot_bull's. move_speed is left out on
    purpose: which of the two speeds is right for the bot is the plan's open D7. Of the Phase 0
    measurements, the user set attack_range 5.2 and proj_speed 9.0 on both blocks (2026-11-01)
    and reload_seconds 1.6 on the hero only (2026-10-28), where the bot keeps its quoted 2.0."""
    spec = _spec()
    assert (spec["hero_bull"]["reload_seconds"], spec["bot_bull"]["reload_seconds"]) == (1.6, 2.0)
    assert (spec["hero_bull"]["attack_range"], spec["hero_bull"]["proj_speed"]) == (5.2, 9.0)
    for key in ("base_hp", "base_damage", "max_ammo", "attack_cooldown", "proj_kind", "attack_range",
                "proj_speed", "proj_radius", "proj_count", "proj_spread_rad"):
        assert spec["hero_bull"][key] == spec["bot_bull"][key], key


def test_his_0_25_s_cooldown_refires_on_the_fifth_tick():
    """0.25 s is five ticks, so Bull, asked to fire on every tick at action_repeat 1, fires on
    every fifth while his clip lasts. Until 2026-10-23 `ent_attack_cd` counted down with a plain
    float32 clamp, which leaves ~1e-8 s after the fifth tick, and every 0.25-s kind (Bull, the
    sniper, artillery, rifle and spike) fired on every sixth."""
    env = _tiny_env({**BULL, "sim": {"action_repeat": 1}}, n_envs=1)
    env.reset()
    st = env.state
    st.ent_ammo[0, 0] = 3.0
    st.ent_attack_cd[0, 0] = 0.0
    st.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    st.ent_pos[0, 1] = torch.tensor([2.0, 2.0])      # both bots behind him and out of his 5.2 tiles
    st.ent_pos[0, 2] = torch.tensor([2.0, 18.0])
    st.ent_facing[0, 0] = 0.0                        # east
    override = torch.zeros((1, env.cfg.n_entities, 2), dtype=torch.int64)
    override[0, 0, 0] = -1                           # the hero acts; the bots stand and hold fire

    fired = []
    for tick in range(12):
        before = int(st.ent_shots_fired[0, 0])
        env.step(torch.tensor([[0, 1]], dtype=torch.int64), override=override)
        if int(st.ent_shots_fired[0, 0]) > before:
            fired.append(tick)
    assert fired == [0, 5, 10]


def test_the_viewer_draws_the_hero_the_run_plays():
    from brawl_sim.render.viewer import entity_sprite_paths
    assert entity_sprite_paths("bull")[Kind.HERO].name == "hero_bull.png"
    assert entity_sprite_paths("mortis")[Kind.HERO].name == "hero_mortis.png"
    assert entity_sprite_paths("bull")[Kind.BOT_BULL].name == "bot_bull.png"


# ---- Phase 2: where the volley goes (BULL_HERO_PLAN.md §5.2) --------------------------------
# Literals, not params: Bull's reach is attack_range 5.2 + proj_radius 0.30 (+ unit_radius 0.40 for
# an enemy's body), a pellet is 880, and move value 1 is east (+x), 5 is +y, 13 is -y.

_ATTACK, _AUTO = 1, 4
_EAST, _PLUS_Y, _MINUS_Y = 1, 5, 13
_FAR = [3.0, 3.0]


def _bull_duel(enemy_at, facing=0.0):
    """Bull at (10,10) with `facing`, bot 1 at `enemy_at`, bot 2 far away, every crate dead, one
    tick per step, both bots frozen by the override (the hero row's -1 lets the action through, as
    in test_attack_in_reach). Bot 1 keeps its own HP: cube pickup rebuilds ent_max_hp from the
    kind every tick, so a hand-set HP would not stick, and it must outlast a whole volley."""
    env = _tiny_env(_deep_merge(BULL, {"sim": {"action_repeat": 1}, "action": {"auto_aim": True}}),
                    n_envs=1)
    env.reset()
    st = env.state
    assert env.cfg.n_entities == 3 and env.cfg.auto_aim
    st.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    st.ent_pos[0, 1] = torch.tensor(enemy_at)
    st.ent_pos[0, 2] = torch.tensor(_FAR)
    st.ent_facing.zero_()
    st.ent_facing[0, 0] = facing
    assert float(st.ent_hp[0, 1]) > 5 * 880, "a full volley would kill bot 1 and hide hits"
    st.box_alive.fill_(False)
    override = torch.zeros(1, env.cfg.n_entities, 2, dtype=torch.int64)
    override[:, 0, 0] = -1
    return env, override


def _fire(env, override, attack, move=0):
    return env.step(torch.tensor([[move, attack]], dtype=torch.int64), override)


def _volley_dir(env):
    """The direction of the volley just fired: the fan is symmetric, so its pellets' summed
    velocity points down the aim."""
    s = env.state
    mine = s.prj_alive[0] & (s.prj_owner[0] == 0)
    assert int(mine.sum()) == 5, "one volley of five pellets"
    v = s.prj_vel[0][mine].sum(dim=0)
    return v / v.norm()


def _hp(env):
    return float(env.state.ent_hp[0, 1])


def _damage_taken(env, override, before, ticks=12):
    """Let the volley fly for `ticks` (0.6 s, out past the 5.2 range at 9 tiles/s) and return bot
    1's HP lost since `before`, read BEFORE the fire step. A hit on the trigger tick itself must
    count: reading after it hid the free 880 a ranged attack's zero-width melee cone used to land
    there."""
    for _ in range(ticks):
        _fire(env, override, 0)
    return before - _hp(env)


def _close(direction, xy):
    return torch.allclose(direction, torch.tensor(xy), atol=1e-4)


def test_value_1_fires_bulls_volley_along_the_move_bin_not_the_facing_or_the_enemy():
    """The aimed attack goes where he is running: facing +x with an enemy 2 tiles along it, a +y
    move bin sends the volley +y and the enemy takes nothing, on the fire step or after it."""
    env, override = _bull_duel([12.0, 10.0])
    before = _hp(env)
    _fire(env, override, _ATTACK, move=_PLUS_Y)
    assert int(env.state.ent_shots_fired[0, 0]) == 1
    assert _close(_volley_dir(env), [0.0, 1.0])
    assert _damage_taken(env, override, before) == 0.0


def test_an_idle_value_1_fires_along_the_facing():
    env, override = _bull_duel(_FAR, facing=math.pi / 2)
    _fire(env, override, _ATTACK)
    assert _close(_volley_dir(env), [0.0, 1.0])


def test_value_4_aims_bulls_volley_at_an_enemy_off_his_line_and_lands_it():
    """An enemy 2 tiles away, 60 degrees off both the facing and the +x move bin. Value 1 down the
    bin misses it; value 4 on the same bin turns the volley onto it and at least four of the five
    pellets land (the fan is +/-0.25 rad, so at 2 tiles all five should)."""
    at = [10.0 + 2.0 * math.cos(math.pi / 3), 10.0 + 2.0 * math.sin(math.pi / 3)]

    env, override = _bull_duel(at)
    before = _hp(env)
    _fire(env, override, _ATTACK, move=_EAST)
    assert _close(_volley_dir(env), [1.0, 0.0])
    assert _damage_taken(env, override, before) == 0.0

    env, override = _bull_duel(at)
    before = _hp(env)
    _fire(env, override, _AUTO, move=_EAST)
    assert _close(_volley_dir(env), [math.cos(math.pi / 3), math.sin(math.pi / 3)])
    assert _damage_taken(env, override, before) >= 4 * 880


@pytest.mark.parametrize("dx, auto_wins", [(5.8, True), (6.0, False)])
def test_value_4_reaches_5_9_tiles_and_falls_back_to_the_bin_past_it(dx, auto_wins):
    """5.2 range + 0.30 pellet + 0.40 enemy body. Past it there is no target, and value 4 fires
    like value 1: down the move bin."""
    env, override = _bull_duel([10.0 + dx, 10.0])
    _fire(env, override, _AUTO, move=_MINUS_Y)
    assert _close(_volley_dir(env), [1.0, 0.0] if auto_wins else [0.0, -1.0])


def test_value_4_with_nothing_in_reach_and_no_move_fires_along_the_facing():
    env, override = _bull_duel(_FAR, facing=math.pi)
    _fire(env, override, _AUTO)
    assert _close(_volley_dir(env), [-1.0, 0.0])


def test_weapon_reach_is_mortis_dash_and_bulls_volley():
    """Mortis: dash_distance 2.67 + dash_radius 0.70 uncharged, more when the long dash is
    charged. Bull: attack_range 5.2 + proj_radius 0.30, charged or not (he has no dash)."""
    from brawl_sim.core import hero
    from brawl_sim.core.state import allocate

    for overrides, uncharged, grows in ((None, 3.37, True), (BULL, 5.50, False)):
        cfg, params = _params(overrides)
        state = allocate(cfg, n_envs=4, device="cpu", verbose=False)
        state.ent_kind[:, 0] = int(Kind.HERO)
        assert torch.allclose(hero.weapon_reach(state, params, charged=False),
                              torch.full((4,), uncharged), atol=1e-5)
        state.ent_attack_idle_t[:, 0] = 100.0               # long past any long_dash_seconds
        charged = hero.weapon_reach(state, params, charged=True)
        assert bool(torch.all(charged > uncharged + 0.1)) == grows
        assert bool(torch.isfinite(charged).all())


def test_in_reach_radius_is_mortis_dash_and_where_most_of_bulls_volley_lands():
    """Mortis: 2.67 + 0.70 + 0.40 = 3.77, his uncharged dash. Bull: the middle three of his five
    pellets sit at 0 and +/-0.125 rad (0.5 rad spread over 4 gaps), where an enemy body counts
    (user decision 2026-10-23). A pellet 0.125 rad off would touch one out to 0.70 / sin(0.125) =
    5.61 tiles, but at the 5.2 range (2026-11-01) its tip binds first: 5.2 cos(0.125) +
    sqrt(0.70^2 - (5.2 sin(0.125))^2) = 5.42."""
    from brawl_sim.core import hero
    from brawl_sim.core.state import allocate

    for overrides, radius in ((None, 3.77), (BULL, 5.4234)):
        cfg, params = _params(overrides)
        state = allocate(cfg, n_envs=4, device="cpu", verbose=False)
        state.ent_kind[:, 0] = int(Kind.HERO)
        state.ent_attack_idle_t[:, 0] = 100.0               # a charged long dash must not count
        assert torch.allclose(hero.in_reach_radius(state, params), torch.full((4,), radius),
                              atol=1e-3)


@pytest.mark.parametrize("dx, at_least, at_most", [(5.35, 3, 5), (5.70, 0, 2)])
def test_most_of_the_volley_lands_inside_the_in_reach_radius_and_not_past_it(dx, at_least, at_most):
    """The radius against the sim's own pellets (swept hits): an auto-aimed volley at an enemy just
    inside 5.42 tiles lands three or more of five, past it two or fewer. A pellet's last tick
    sweeps its whole 0.45 tiles, so 12 ticks carry it 5.4 tiles, and three still land out to
    5.55: the radius errs short by 0.13."""
    env, override = _bull_duel([10.0 + dx, 10.0])
    before = _hp(env)
    _fire(env, override, _AUTO)
    landed = _damage_taken(env, override, before, ticks=20) / 880
    assert at_least <= landed <= at_most


@pytest.mark.parametrize("dx, expected", [(5.0, 1), (5.35, 1), (5.50, 0), (5.8, 0)])
def test_bulls_attack_in_reach_is_where_most_of_his_volley_lands(dx, expected):
    """The reward's in-reach flag for Bull reads 5.42 tiles to an enemy's centre, where Mortis reads
    his 3.77-tile dash (test_attack_in_reach). 5.8 is inside his 5.9-tile auto-aim reach but out of
    this one: only the middle pellet could land there."""
    env, override = _bull_duel([10.0 + dx, 10.0])
    *_, info = _fire(env, override, _ATTACK)
    assert int(env.state.ent_shots_fired[0, 0]) == 1
    assert info["attack_in_reach_tick"].tolist() == [expected]
