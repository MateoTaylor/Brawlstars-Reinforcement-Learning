"""Bull's gadget, Stomper, and his star power, Berserker (BULL_HERO_PLAN.md Phase 5: D4, D9, Q1, Q3
and Q5, all the user's 2026-10-23).

Its own file, as the charge has test_charge_super.py: the stomp spans hero (start_stomp,
advance_stomp, gadget_ready), status (the slow and the stun), env (the tap, phase 5b, the reward's
flag) and config (validate). Literals, not params. dt is 0.05 s. A tap arms the stomp, and 0.30 s
= 6 ticks later a pulse around Bull's centre catches every other living player within 2.67 tiles,
on centres: one already slowed is stunned 1.4 s (28 ticks) and keeps its slow, any other is
slowed 50 % for 1.0 s (20 ticks). The wind-up and the stun were measured 2026-11-01
(BULL_HERO_PLAN.md Phase 0 d, e; the wiki said 0.25 and 1.0). A player in the air is skipped, and
nobody takes damage. The cooldown
is 20.0 s = 400 ticks, charged at spawn, and the charge super holds it closed until he lands.
Berserker: below 60 % of his max HP, cubes' bonus included, his 1.6 s reload runs twice as fast,
0.03125 ammo a tick becoming 0.0625. Bull walks 2.68 tiles/s and bot_rifle 2.40. Move value 1 is +x;
attack values are 1 attack, 2 super, 3 gadget.
"""
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import _deep_merge, build_params, load_config, validate
from brawl_sim.constants import Kind
from brawl_sim.core import hero, projectiles, status
from brawl_sim.core.state import allocate, check_invariants
from brawl_sim.env import BrawlVecEnv

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
TINY = yaml.safe_load((CONFIGS / "presets" / "debug_tiny.yaml").read_text())
BULL = {"entities": {"hero": "bull"}}
MORTIS = {"entities": {"hero": "mortis"}}

_SUPER, _GADGET = 2, 3
_EAST = 1


def _spec(overrides=None) -> dict:
    spec = {
        **yaml.safe_load((CONFIGS / "default.yaml").read_text()),
        **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text()),
    }
    return _deep_merge(spec, overrides or {})


# ---- through the env ------------------------------------------------------------------------
# A 3-entity tiny env, one tick per step unless asked, two rifle bots frozen by the override (the
# hero row's -1 lets his action through), regen off, and `check_invariants` after every step.

def _duel(bull_at=(10.0, 10.0), bot1=(3.0, 3.0), bot2=(3.0, 17.0), action_repeat=1, debug=True):
    over = _deep_merge(TINY, _deep_merge(BULL, {
        "sim": {"action_repeat": action_repeat, "max_episode_steps": 2000},
        "regen": {"enabled": False},
        "engine": {"debug_checks": debug},
        "entities": {"randomize_enemy_types": False, "fixed_enemy_types": ["rifle", "rifle"]},
    }))
    cfg = load_config(CONFIGS / "default.yaml", overrides=over)
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, spec=_spec(over), verbose=False)
    env.reset()
    st = env.state
    assert env.cfg.n_entities == 3
    for slot, xy in enumerate((bull_at, bot1, bot2)):
        st.ent_pos[0, slot] = torch.tensor(xy)
    st.ent_facing.zero_()
    st.box_alive.fill_(False)
    override = torch.zeros(1, env.cfg.n_entities, 2, dtype=torch.int64)
    override[:, 0, 0] = -1
    return env, override


def _step(env, override, attack=0, move=0):
    return env.step(torch.tensor([[move, attack]], dtype=torch.int64), override)


def _attack_mask(env):
    return hero.action_mask(env.state, env.params, env.cfg)["attack"][0]


def _bot1_walks(override, fire=0):
    """Bot 1 walks +x (and asks to fire, if `fire`); bot 2 stays frozen."""
    walk = override.clone()
    walk[0, 1] = torch.tensor([_EAST, fire])
    return walk


def _stomp(env, override):
    """Tap on tick 1, idle through tick 7, the pulse."""
    _step(env, override, _GADGET)
    for _ in range(6):
        _step(env, override)


def test_a_fresh_player_is_slowed_by_half_for_20_ticks_from_the_6th_tick_after_the_tap():
    """Bot 1 walks +x from a tile east of Bull: 0.12 a tick until the pulse on tick 7, 0.06 from
    the very tick it lands (phase 5b runs before the walk) through the 19 after it, then 0.12. The
    tap spends no ammo, starts the 20 s cooldown and reveals him, as Mortis's throw does, and the
    pulse leaves Bull himself, the far bot and every HP alone."""
    env, override = _duel(bot1=(11.0, 10.0))
    st = env.state
    walk = _bot1_walks(override)
    hp = st.ent_hp[0].clone()
    ammo = float(st.ent_ammo[0, 0])
    xs = [float(st.ent_pos[0, 1, 0])]
    for tick in range(1, 31):
        _step(env, walk, _GADGET if tick == 1 else 0)
        xs.append(float(st.ent_pos[0, 1, 0]))
        if tick == 1:
            assert float(st.ent_stomp_t[0, 0]) == pytest.approx(0.30)
            assert float(st.ent_gadget_cd[0, 0]) == 20.0
            assert float(st.ent_ammo[0, 0]) == ammo and float(st.ent_attack_cd[0, 0]) == 0
            assert float(st.ent_reveal_t[0, 0]) == 1.0
        if tick < 7:
            assert float(st.ent_slow_t[0, 1]) == 0, f"slowed on tick {tick}"
        if tick == 7:
            assert float(st.ent_stomp_t[0, 0]) == 0
            assert float(st.ent_slow_t[0, 1]) == 1.0
            assert float(st.ent_slow_frac[0, 1]) == 0.5
            assert float(st.ent_stun_t.abs().sum()) == 0, "a fresh player was stunned"
            assert float(st.ent_slow_t[0, 0]) == 0, "slowed the stomper himself"
            assert float(st.ent_slow_t[0, 2]) == 0
    walked = [b - a for a, b in zip(xs, xs[1:])]          # walked[t - 1] is tick t's
    for tick in range(1, 31):
        want = 0.06 if 7 <= tick <= 26 else 0.12
        assert walked[tick - 1] == pytest.approx(want, abs=1e-4), f"tick {tick}"
    assert torch.equal(st.ent_hp[0], hp), "the stomp dealt damage"


def test_a_player_already_slowed_is_stunned_for_28_ticks_and_its_slow_runs_on():
    """Bot 1 carries a 40 % slow with 3.0 s left (D4: a slow from anything counts). It walks 0.072
    a tick until the pulse on tick 7 and stands from the pulse tick through the 27 after it. Asked
    to fire on every one of those, it fires nothing until tick 35, its first free tick, when it
    walks again too: the stun stops the walk and the fire of the tick it lands on. Its slow is not
    replaced."""
    env, override = _duel(bot1=(11.0, 10.0))
    st = env.state
    st.ent_slow_t[0, 1] = 3.0
    st.ent_slow_frac[0, 1] = 0.4
    shots = int(st.ent_shots_fired[0, 1])
    xs = [float(st.ent_pos[0, 1, 0])]
    for tick in range(1, 36):
        _step(env, _bot1_walks(override, fire=int(tick >= 7)), _GADGET if tick == 1 else 0)
        xs.append(float(st.ent_pos[0, 1, 0]))
        assert bool(status.disabled(st)[0, 1]) == (7 <= tick <= 34), f"tick {tick}"
        if tick == 7:
            assert float(st.ent_stun_t[0, 1]) == pytest.approx(1.4)
            assert float(st.ent_slow_t[0, 1]) == pytest.approx(2.65, abs=1e-5)
            assert float(st.ent_slow_frac[0, 1]) == pytest.approx(0.4)
        if tick <= 34:
            assert int(st.ent_shots_fired[0, 1]) == shots, f"fired while stunned, tick {tick}"
    assert int(st.ent_shots_fired[0, 1]) == shots + 1, "no shot on the first free tick"
    walked = [b - a for a, b in zip(xs, xs[1:])]
    for tick in range(1, 36):
        want = 0.0 if 7 <= tick <= 34 else 0.072
        assert walked[tick - 1] == pytest.approx(want, abs=1e-4), f"tick {tick}"


def test_the_charge_then_the_stomp_stuns_a_player_the_charge_hit_once_it_has_landed():
    """The combo the user described. Bot 1 stood on his line at (13, 10): hit on tick 28, pushed
    to (15.5, 10) and in the air through tick 37, when Bull lands at (15, 10) and slows it 40 % for
    1.5 s. The tap on tick 38, the first the gadget is open again (Q5), lands on tick 44, stunning
    it 1.4 s with 1.15 s of the slow left. The stomp takes no HP: the 2000 is the charge's."""
    env, override = _duel(bull_at=(4.0, 10.0), bot1=(13.0, 10.0))
    st = env.state
    st.ent_super_charge[0, 0] = 9
    hp = float(st.ent_hp[0, 1])
    _step(env, override, _SUPER, move=_EAST)
    for _ in range(36):
        _step(env, override)
    assert float(st.ent_dash_t[0, 0]) == 0 and float(st.ent_slow_t[0, 1]) == 1.5
    _step(env, override, _GADGET)                          # tick 38
    for _ in range(5):
        _step(env, override)
    assert float(st.ent_stun_t[0, 1]) == 0 and not bool(status.airborne(st)[0, 1])
    _step(env, override)                                   # tick 44: the pulse
    assert float(st.ent_stun_t[0, 1]) == pytest.approx(1.4)
    assert float(st.ent_slow_t[0, 1]) == pytest.approx(1.15, abs=1e-5)
    assert hp - float(st.ent_hp[0, 1]) == 2000.0


@pytest.mark.parametrize("slowed", [False, True])
def test_a_player_in_the_air_is_skipped_slowed_or_not(slowed):
    """D4 and §5.5: the air is neither slowed nor stunned. Bot 1's 1.0 s in the air, set before the
    tap, outlasts the pulse on tick 7; the 3.0 s slow, when it carries one, just runs on."""
    env, override = _duel(bot1=(11.0, 10.0))
    st = env.state
    st.ent_airborne_t[0, 1] = 1.0
    if slowed:
        st.ent_slow_t[0, 1] = 3.0
        st.ent_slow_frac[0, 1] = 0.4
    _stomp(env, override)
    assert bool(status.airborne(st)[0, 1])
    assert float(st.ent_stun_t[0, 1]) == 0
    assert float(st.ent_slow_t[0, 1]) == pytest.approx(2.65 if slowed else 0.0, abs=1e-5)
    assert float(st.ent_slow_frac[0, 1]) == pytest.approx(0.4 if slowed else 0.0)


@pytest.mark.parametrize("dx, dy, caught", [
    (2.6, 0.0, True), (2.75, 0.0, False),      # on centres: a body's 0.4 adds nothing
    (1.85, 1.85, True), (1.95, 1.95, False),   # 2.616 and 2.758 along the diagonal
    (0.0, -2.6, True),
])
def test_the_pulse_reaches_2_67_tiles_from_his_centre_to_theirs(dx, dy, caught):
    env, override = _duel(bot1=(10.0 + dx, 10.0 + dy))
    _stomp(env, override)
    assert (float(env.state.ent_slow_t[0, 1]) == 1.0) == caught


def test_he_walks_and_may_attack_through_the_wind_up_and_the_pulse_lands_where_he_stands():
    """Bull taps at (6, 10) walking +x and is 6 x 0.134 = 0.804 further on when the pulse lands on
    tick 7, before that tick's walk. Bot 1, 3.1 tiles ahead of the tap, is 2.296 from the pulse and
    slowed; bot 2, 2.4 behind it, is 3.204 away and not. Only the pulse waits: his attack
    stays open through the wind-up, unlike the charge's."""
    env, override = _duel(bull_at=(6.0, 10.0), bot1=(9.1, 10.0), bot2=(3.6, 10.0))
    st = env.state
    _step(env, override, _GADGET, move=_EAST)
    for tick in range(2, 7):
        assert bool(_attack_mask(env)[1]), f"the attack masked in the wind-up, tick {tick}"
        _step(env, override, move=_EAST)
    assert float(st.ent_pos[0, 0, 0]) == pytest.approx(6.804, abs=1e-4)
    _step(env, override, move=_EAST)
    assert float(st.ent_slow_t[0, 1]) == 1.0
    assert float(st.ent_slow_t[0, 2]) == 0


def test_the_stomp_hurts_nothing_crates_included():
    env, override = _duel(bot1=(11.0, 10.0))
    st = env.state
    st.box_alive[0, 0] = True
    st.box_pos[0, 0] = torch.tensor([10.5, 11.5])
    st.box_max_hp[0, 0] = 4000.0
    st.box_hp[0, 0] = 4000.0
    hp = st.ent_hp[0].clone()
    _stomp(env, override)
    assert float(st.ent_slow_t[0, 1]) == 1.0
    assert torch.equal(st.ent_hp[0], hp)
    assert float(st.box_hp[0, 0]) == 4000.0 and bool(st.box_alive[0, 0])


def test_the_tap_arms_the_stomp_instead_of_throwing_a_spinner(monkeypatch):
    """`projectiles.spawn_gadget` runs every tick, and on the tap tick it must see no firing row.
    Bull's spinner numbers are all 0, so a spinner thrown beside the stomp would go off unseen on
    its own tick: only the projectile slot it took would show."""
    env, override = _duel(bot1=(11.0, 10.0))
    rows = []
    real = projectiles.spawn_gadget

    def spy(state, fire_mask, *args, **kwargs):
        rows.append(int(fire_mask.sum()))
        return real(state, fire_mask, *args, **kwargs)

    monkeypatch.setattr(projectiles, "spawn_gadget", spy)
    _step(env, override, _GADGET)
    assert rows == [0]
    assert float(env.state.ent_stomp_t[0, 0]) == pytest.approx(0.30)


@pytest.mark.parametrize("bot1, bot2, airborne, hits", [
    ((11.0, 10.0), (3.0, 17.0), False, 1),     # one player caught
    ((11.0, 10.0), (9.0, 10.0), False, 1),     # two caught: still one landing
    ((3.0, 3.0), (3.0, 17.0), False, 0),       # nobody within reach
    ((11.0, 10.0), (3.0, 17.0), True, 0),      # only a player in the air, who is skipped
])
def test_gadget_hit_pays_once_for_a_stomp_that_slowed_or_stunned_a_player(bot1, bot2, airborne, hits):
    """D9 (user, 2026-10-23): `info["gadget_hit_tick"]`, the reward's gadget_hit count, is 1 on
    the tick a stomp lands on at least one player, and 0 on every other."""
    env, override = _duel(bot1=bot1, bot2=bot2)
    if airborne:
        env.state.ent_airborne_t[0, 1] = 1.0
    counts = []
    for tick in range(1, 10):
        *_, info = _step(env, override, _GADGET if tick == 1 else 0)
        counts.append(int(info["gadget_hit_tick"][0]))
    assert counts == [0] * 6 + [hits] + [0] * 2


def test_at_the_shipped_decision_rate_the_stomp_lands_on_the_next_decisions_second_tick():
    """At action_repeat 5 the tap goes on decision 1's first tick and lands 6 ticks on, the second
    of decision 2, before that tick's attack phase. The reward's gadget_hit is paid on decision 2,
    and bot 1's slow has 1.0 - 3 x 0.05 left at its end."""
    env, override = _duel(bot1=(11.0, 10.0), action_repeat=5)
    st = env.state
    *_, info = _step(env, override, _GADGET)
    assert int(info["gadget_hit_tick"][0]) == 0
    assert float(st.ent_stomp_t[0, 0]) == pytest.approx(0.10, abs=1e-6)
    assert float(st.ent_slow_t[0, 1]) == 0
    *_, info = _step(env, override)
    assert int(info["gadget_hit_tick"][0]) == 1
    assert float(st.ent_slow_t[0, 1]) == pytest.approx(0.85, abs=1e-5)


def test_the_cooldown_is_400_ticks_to_the_tick():
    """20.0 s at one tick a step: tapped on tick 1, masked after tick 400 (399 countdowns), open
    after tick 401 (400). Plain float32 subtraction leaves 1.8e-5 s after the 400th countdown and
    would hold the gadget one tick more; hero.tick_timers snaps it as status.count_down does."""
    env, override = _duel(debug=False)
    st = env.state
    _step(env, override, _GADGET)
    for _ in range(399):
        _step(env, override)
    assert not bool(_attack_mask(env)[3]), "open after 399 countdowns"
    _step(env, override)
    assert float(st.ent_gadget_cd[0, 0]) == 0
    assert bool(_attack_mask(env)[3])
    _step(env, override, _GADGET)
    assert float(st.ent_gadget_cd[0, 0]) == 20.0
    assert float(st.ent_stomp_t[0, 0]) == pytest.approx(0.30)


def test_at_the_shipped_decision_rate_the_stomp_rearms_on_the_81st_observation():
    """As test_gadget.py pins Mortis's 73rd. The tap lands on decision 1's first tick, so the
    observation after decision k reads 19.8 - 0.25 (k - 1): 0.05 (still masked) after decision 80
    and 0 after decision 81. The earliest second stomp is the decision after it, 405 ticks =
    20.25 s after the first: the figure a decision-rate mirror (the live shadow) must reproduce,
    not 20.0."""
    env, override = _duel(action_repeat=5, debug=False)
    st = env.state
    obs, *_ = _step(env, override, _GADGET)
    legal = [bool(obs["action_mask"]["attack"][0, 3])]
    for _ in range(80):
        obs, *_ = _step(env, override)
        legal.append(bool(obs["action_mask"]["attack"][0, 3]))
    assert legal[:80] == [False] * 80
    assert legal[80] is True
    assert int(st.step_count[0]) == 405
    _step(env, override, _GADGET)                          # ...and legal means it really goes
    assert float(st.ent_gadget_cd[0, 0]) == pytest.approx(19.8, abs=1e-4)
    assert float(st.ent_stomp_t[0, 0]) == pytest.approx(0.10, abs=1e-6)


def test_the_charge_holds_the_gadget_closed_from_its_tap_until_he_lands():
    """Q5 (user, 2026-10-23): no Stomper mid-charge. Charged at spawn, the gadget is open until the
    super; then closed in the mask and in obs `hero.gadget_ready` alike through the 4 standing
    ticks and the rush, a tap on tick 10 doing nothing; and open again on tick 37, the landing,
    after which a tap arms it."""
    env, override = _duel(bull_at=(4.0, 10.0))
    st = env.state
    assert bool(_attack_mask(env)[3]), "not charged at spawn"
    st.ent_super_charge[0, 0] = 9
    obs, *_ = _step(env, override, _SUPER, move=_EAST)
    for tick in range(1, 38):
        if tick > 1:
            obs, *_ = _step(env, override, _GADGET if tick == 10 else 0)
        legal = bool(obs["action_mask"]["attack"][0, 3])
        assert bool(obs["hero"]["gadget_ready"][0]) == legal, f"obs and mask differ, tick {tick}"
        assert legal == (tick == 37), f"tick {tick}"
    assert float(st.ent_gadget_cd[0, 0]) == 0 and float(st.ent_stomp_t[0, 0]) == 0
    _step(env, override, _GADGET)
    assert float(st.ent_stomp_t[0, 0]) == pytest.approx(0.30)


def test_a_reset_disarms_a_pending_stomp_and_recharges_the_gadget():
    env, override = _duel()
    _step(env, override, _GADGET)
    assert float(env.state.ent_stomp_t[0, 0]) == pytest.approx(0.30)
    env.reset()
    assert float(env.state.ent_stomp_t.abs().sum()) == 0
    assert float(env.state.ent_gadget_cd[0, 0]) == 0


# ---- straight from hero.advance_stomp and hero.tick_timers ---------------------------------

def _cfg_and_params(hero_overrides=BULL, spec_overrides=None, world=True):
    overrides = _deep_merge({"world": {"map_h": 30, "map_w": 30}} if world else {}, hero_overrides)
    cfg = load_config(CONFIGS / "default.yaml", overrides=overrides)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    spec = _spec(_deep_merge(overrides, spec_overrides or {}))
    return cfg, build_params(cfg, n_envs=1, device="cpu", gen=gen, spec=spec)


def _direct(victims=()):
    """Bull at (15, 15) on a 30x30 world, every unit at 10000 of 10000 HP, every bot a sniper parked
    along y = 3 far from him, except the `victims` [(slot, (x, y))] placed by hand."""
    cfg, params = _cfg_and_params()
    state = allocate(cfg, n_envs=1, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[0, 0] = int(Kind.HERO)
    state.ent_kind[0, 1:] = int(Kind.BOT_SNIPER)
    state.ent_max_hp.fill_(10000.0)
    state.ent_hp.fill_(10000.0)
    for e in range(1, cfg.n_entities):
        state.ent_pos[0, e] = torch.tensor([2.0 + 2.0 * e, 3.0])
    state.ent_pos[0, 0] = torch.tensor([15.0, 15.0])
    for slot, xy in victims:
        state.ent_pos[0, slot] = torch.tensor(xy)
    return cfg, params, state


def test_the_pulse_skips_the_dead_and_a_stomper_dead_in_the_wind_up_lands_nothing():
    """Slot 1 alive and slot 2 dead, each a tile from him: only slot 1 is caught. Then Bull dies
    with the stomp one tick from landing: it is spent and catches nobody."""
    cfg, params, state = _direct(victims=[(1, (16.0, 15.0)), (2, (14.0, 15.0))])
    state.ent_alive[0, 2] = False
    state.ent_hp[0, 2] = 0.0
    state.ent_stomp_t[0, 0] = 0.05
    caught, lands = hero.advance_stomp(state, params, cfg)
    assert caught[0, 0].nonzero().flatten().tolist() == [1]
    assert lands[0].tolist()[:3] == [True, False, False]
    assert state.ent_slow_t[0, :3].tolist() == [0.0, 1.0, 0.0]
    assert float(state.ent_stomp_t[0, 0]) == 0

    cfg, params, state = _direct(victims=[(1, (16.0, 15.0))])
    state.ent_alive[0, 0] = False
    state.ent_hp[0, 0] = 0.0
    state.ent_stomp_t[0, 0] = 0.05
    caught, lands = hero.advance_stomp(state, params, cfg)
    assert not bool(caught.any())
    assert not bool(lands.any()), "a dead stomper's stomp landed (and would score a gadget_miss)"
    assert float(state.ent_slow_t.abs().sum()) == 0
    assert float(state.ent_stomp_t[0, 0]) == 0, "a dead stomper's stomp stayed armed"


def test_apply_stun_keeps_the_longer_stun_on_the_rows_hit_and_leaves_a_slow_running():
    """As apply_airborne: the longer of what is left and the new stun, nothing on a row not hit."""
    cfg, params, state = _direct()
    state.ent_stun_t[0, 1] = 2.0
    state.ent_slow_t[0, 2] = 0.5
    state.ent_slow_frac[0, 2] = 0.3
    hit = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    hit[0, 1:3] = True
    status.apply_stun(state, hit, torch.full((1, cfg.n_entities), 1.0))
    assert state.ent_stun_t[0, :4].tolist() == [0.0, 2.0, 1.0, 0.0]
    assert float(state.ent_slow_t[0, 2]) == 0.5
    assert float(state.ent_slow_frac[0, 2]) == pytest.approx(0.3)


def _reload_gain(state, params, cfg, hp, max_hp=10000.0, slot=0):
    """One tick of reload from an empty clip at `hp` of `max_hp`."""
    state.ent_hp[0, slot] = hp
    state.ent_max_hp[0, slot] = max_hp
    state.ent_ammo[0, slot] = 0.0
    state.ent_attack_cd[0, slot] = 0.0
    hero.tick_timers(state, params, cfg)
    return float(state.ent_ammo[0, slot])


@pytest.mark.parametrize("hp, max_hp, gain", [
    (5999.0, 10000.0, 0.0625),   # below 60 %: twice 0.05 / 1.6
    (6000.0, 10000.0, 0.03125),  # at it: "below" is strict
    (1.0, 10000.0, 0.0625),
    (10000.0, 10000.0, 0.03125),
    (7100.0, 12000.0, 0.0625),   # five 400-HP cubes: 60 % is 7200, so 7100 is below it,
    (7300.0, 12000.0, 0.03125),  # though both are above 60 % of his base 10000
])
def test_berserker_doubles_the_reload_below_60_percent_of_max_hp_cubes_included(hp, max_hp, gain):
    """The strict edge is pinned at 10000 only: float32's 0.6 times 12000 rounds to 7200.0005, so
    exactly 7200 of 12000 reads a hair below it."""
    cfg, params, state = _direct()
    assert _reload_gain(state, params, cfg, hp, max_hp) == pytest.approx(gain, abs=1e-7)


def test_berserker_crosses_60_percent_both_ways_with_no_latch():
    """Healing back to 60 % ends it on the next tick and falling below starts it again: the
    boost reads this tick's HP and remembers nothing."""
    cfg, params, state = _direct()
    gains = [_reload_gain(state, params, cfg, hp) for hp in (5000.0, 6000.0, 5999.0, 6500.0, 100.0)]
    assert gains == pytest.approx([0.0625, 0.03125, 0.0625, 0.03125, 0.0625], abs=1e-7)


def test_berserker_still_waits_out_the_attack_cooldown_and_is_bulls_alone():
    """The cooldown pauses the reload, boosted or not. A sniper bot at 1 HP reloads at its own
    0.05 / 2.1 a tick, and no kind in a Mortis spec has a threshold."""
    cfg, params, state = _direct()
    state.ent_hp[0, 0] = 1000.0
    state.ent_ammo[0, 0] = 0.0
    state.ent_attack_cd[0, 0] = 0.25
    hero.tick_timers(state, params, cfg)
    assert float(state.ent_ammo[0, 0]) == 0.0
    assert _reload_gain(state, params, cfg, 1.0, slot=1) == pytest.approx(0.05 / 2.1, abs=1e-7)
    _, mortis = _cfg_and_params(MORTIS)
    assert float(mortis.reload_boost_below_hp_frac.abs().max()) == 0


# ---- the switches, validate and the invariant -----------------------------------------------

def test_a_mortis_spec_has_no_stomp_and_skips_the_phase():
    """The stomp is behind a Python switch on stomp_radius_tiles, the widest STOMP in the spec:
    2.67 with Bull, 0 with Mortis, whose gadget is a SPINNER (0) in every column."""
    _, bull = _cfg_and_params(BULL)
    assert bull.stomp_radius_tiles == pytest.approx(2.67)
    assert int(bull.gadget_type[0, int(Kind.HERO)]) == 1
    _, mortis = _cfg_and_params(MORTIS)
    assert mortis.stomp_radius_tiles == 0
    assert int(mortis.gadget_type.max()) == 0

    over = _deep_merge(TINY, MORTIS)
    cfg = load_config(CONFIGS / "default.yaml", overrides=over)
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, spec=_spec(over), verbose=False)
    env.reset()
    assert env._stomp_phase() is None


@pytest.mark.parametrize("hero_name, block, match", [
    (BULL, {"hero_bull": {"stomp_radius": 0}}, "STOMP but stomp_radius is 0"),
    (BULL, {"hero_bull": {"stomp_windup_seconds": 0}}, "STOMP but stomp_windup_seconds is 0"),
    (BULL, {"hero_bull": {"stomp_slow_seconds": 0, "stomp_stun_seconds": 0}}, "change nothing"),
    (BULL, {"hero_bull": {"stomp_stun_seconds": -1.0}}, "stomp_stun_seconds must be >= 0"),
    (BULL, {"hero_bull": {"stomp_slow_frac": 1.5}}, r"stomp_slow_frac must be in \[0, 1\]"),
    (MORTIS, {"hero_mortis": {"stomp_radius": 2.0}}, "stomp_radius is set but gadget_type is SPINNER"),
    (BULL, {"hero_bull": {"reload_boost_below_hp_frac": 1.5}},
     r"reload_boost_below_hp_frac must be in \[0, 1\]"),
    (BULL, {"hero_bull": {"reload_boost_mult": 0}}, "reload_boost_mult is 0"),
])
def test_validate_refuses_a_stomp_or_a_berserker_that_cannot_work(hero_name, block, match):
    cfg, params = _cfg_and_params(hero_name, block, world=False)
    with pytest.raises(ValueError, match=match):
        validate(cfg, params)


@pytest.mark.parametrize("block", [
    {},
    # Milestone M1, the attack alone: no super, no gadget, no star power.
    {"hero_bull": {"super_charge_hits": 0, "gadget_cooldown": 0, "reload_boost_below_hp_frac": 0}},
    # A STOMP kind with no gadget needs no stomp numbers.
    {"hero_bull": {"gadget_cooldown": 0, "stomp_radius": 0, "stomp_windup_seconds": 0}},
])
def test_validate_takes_the_shipped_kit_and_one_with_no_gadget(block):
    cfg, params = _cfg_and_params(BULL, block, world=False)
    validate(cfg, params)


def test_check_invariants_bounds_the_stomp_timer_by_each_kinds_wind_up():
    """[0, 0.30] for Bull; a bot kind has no wind-up, so any armed bot stomp is refused."""
    env, _ = _duel()
    st, cfg, params = env.state, env.cfg, env.params
    st.ent_stomp_t[0, 0] = 0.30
    check_invariants(st, cfg, params)
    for slot, value in ((0, 0.35), (0, -0.05), (1, 0.05)):
        st.ent_stomp_t.zero_()
        st.ent_stomp_t[0, slot] = value
        with pytest.raises(ValueError, match="ent_stomp_t"):
            check_invariants(st, cfg, params)
