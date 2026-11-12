"""Status effects (BULL_HERO_PLAN.md §5.3, D4): slow, stun and airborne as per-entity countdowns
(core/status.py), and the gates they close. Nothing in the sim applies them yet -- Bull's charge
and Stomper will, in Phases 4-5 -- so every test here sets a timer by hand. That a run with every
timer at zero (all of Mortis's) is unchanged is the whole-rollout digest's job
(scripts/probes/hero_seam_digest.py), not this file's.

A timer set before a step reads as one a hit applied on the tick before: a k-tick effect holds
through k - 1 more ticks and is gone on the k-th.

Literals, not params, so no test passes by reading back the constant the code uses: dt is 0.05 s;
bot_rifle walks 2.40 tiles/s and fires five pellets; Edgar's combo lands 1080 on ticks 0 and 5;
Mortis dashes 2.67 tiles in 0.30 s (0.445 a tick) with radius 0.70 for 2000, reaches 3.77 tiles
for the in-reach flag, throws a 2000 spinner with a 1.0 blast on an 18.0 s cooldown, and needs 5
charge for his super; Bull's pellet is 880 with a 0.5 rad fan. Move value 1 is +x; attack values
are 1 attack, 2 super, 3 gadget, 4 auto-aimed.
"""
import math
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import _deep_merge, load_config
from brawl_sim.constants import Kind
from brawl_sim.core import hero, status
from brawl_sim.core.state import check_invariants
from brawl_sim.env import BrawlVecEnv

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
TINY = yaml.safe_load((CONFIGS / "presets" / "debug_tiny.yaml").read_text())
BULL = {"entities": {"hero": "bull"}}

_EAST = 1
_ATTACK, _SUPER, _GADGET, _AUTO = 1, 2, 3, 4
_FAR = [3.0, 3.0]
_FAR_2 = [3.0, 17.0]
_STOPS = ("ent_stun_t", "ent_airborne_t")   # the two timers that stop a unit


def _spec(overrides) -> dict:
    spec = {
        **yaml.safe_load((CONFIGS / "default.yaml").read_text()),
        **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text()),
    }
    return _deep_merge(spec, overrides)


def _env(enemies=("rifle", "rifle"), overrides=None):
    """One env on the blank 20x20 map, one tick per step, `check_invariants` after every step,
    and the two bots of the kinds named (slots 1 and 2)."""
    merged = _deep_merge(TINY, {
        "sim": {"action_repeat": 1},
        "engine": {"debug_checks": True},
        "entities": {"randomize_enemy_types": False, "fixed_enemy_types": list(enemies)},
    })
    merged = _deep_merge(merged, overrides or {})
    cfg = load_config(CONFIGS / "default.yaml", overrides=merged)
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, spec=_spec(merged), verbose=False)
    env.reset()
    return env


def _stage(env, enemy_at, other_at=_FAR, facing=0.0):
    """Hero at (10,10) with `facing`, bot 1 at `enemy_at`, bot 2 at `other_at`, every other facing
    0 (+x), every crate dead. Returns the override that freezes both bots (idle, no fire) and lets
    the hero's own action through (its row's -1)."""
    st = env.state
    assert env.cfg.n_entities == 3
    st.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    st.ent_pos[0, 1] = torch.tensor(enemy_at)
    st.ent_pos[0, 2] = torch.tensor(other_at)
    st.ent_facing.zero_()
    st.ent_facing[0, 0] = facing
    st.box_alive.fill_(False)
    override = torch.zeros(1, 3, 2, dtype=torch.int64)
    override[0, 0, 0] = -1
    return override


def _step(env, override, attack=0, move=0):
    return env.step(torch.tensor([[move, attack]], dtype=torch.int64), override)


def _walking_east(override):
    walk = override.clone()
    walk[0, 1] = torch.tensor([_EAST, 0])
    return walk


# ---- the countdown ---------------------------------------------------------------------------

@pytest.mark.parametrize("seconds, ticks", [
    (0.25, 5), (0.3, 6), (0.5, 10), (0.7, 14), (1.0, 20), (1.5, 30), (2.0, 40), (2.5, 50),
    (3.0, 60),
])
def test_an_effect_lasts_its_whole_ticks_and_not_one_more(seconds, ticks):
    """Plain float32 subtraction leaves ~1e-8 s after the last full tick of 0.25, 0.3, 1.5, 2.0,
    2.5 and 3.0 s, which would hold the effect a tick longer; the half-tick snap removes it."""
    t = torch.tensor([seconds], dtype=torch.float32)
    for _ in range(ticks - 1):
        t = status.count_down(t, 0.05)
    assert float(t) > 0, f"{seconds} s ran out early"
    t = status.count_down(t, 0.05)
    assert float(t) == 0.0, f"{seconds} s left {float(t)!r} after {ticks} ticks"


def test_tick_runs_each_timer_on_its_own_and_clears_the_slow_with_its_timer():
    env = _env()
    st, cfg = env.state, env.cfg
    assert cfg.dt == pytest.approx(0.05)
    st.ent_slow_t[0, 1], st.ent_slow_frac[0, 1] = 0.25, 0.4
    st.ent_stun_t[0, 2] = 0.3
    st.ent_airborne_t[0, 0] = 0.5
    for _ in range(4):
        status.tick(st, cfg)
    assert float(st.ent_slow_frac[0, 1]) == pytest.approx(0.4), "the fraction holds while the slow does"
    status.tick(st, cfg)
    assert float(st.ent_slow_t[0, 1]) == 0.0
    assert float(st.ent_slow_frac[0, 1]) == 0.0, "the fraction goes on the timer's last tick"
    assert float(st.ent_stun_t[0, 2]) > 0 and float(st.ent_airborne_t[0, 0]) > 0
    status.tick(st, cfg)
    assert float(st.ent_stun_t[0, 2]) == 0.0
    for _ in range(3):
        status.tick(st, cfg)
    assert float(st.ent_airborne_t[0, 0]) > 0
    status.tick(st, cfg)
    for name in ("ent_slow_t", "ent_slow_frac", "ent_stun_t", "ent_airborne_t"):
        assert bool((getattr(st, name) == 0).all()), name


# ---- walking and turning ---------------------------------------------------------------------

@pytest.mark.parametrize("timer, seconds, still", [
    ("ent_stun_t", 0.25, 4), ("ent_airborne_t", 0.5, 9),
])
def test_a_stopped_bot_walks_again_on_the_tick_its_timer_runs_out(timer, seconds, still):
    """Through `env.step`, so the countdown is the one phase 2 runs: 5 ticks of stun hold bot 1
    for 4 and let it walk (2.40 x 0.05 = 0.12 tiles) on the 5th; 10 of airborne, 9 and the 10th."""
    env = _env()
    walk = _walking_east(_stage(env, [12.0, 10.0]))
    st = env.state
    getattr(st, timer)[0, 1] = seconds
    start = st.ent_pos[0, 1].clone()
    for t in range(still):
        _step(env, walk)
        assert torch.equal(st.ent_pos[0, 1], start), f"walked on tick {t + 1} of {still}"
    _step(env, walk)
    assert float(st.ent_pos[0, 1, 0] - start[0]) == pytest.approx(0.12, abs=1e-5)


@pytest.mark.parametrize("timer", [None, *_STOPS])
def test_a_stunned_or_airborne_bot_neither_walks_nor_turns(timer):
    env = _env()
    walk = _walking_east(_stage(env, [12.0, 10.0]))
    st = env.state
    st.ent_facing[0, 1] = math.pi / 2
    if timer:
        getattr(st, timer)[0, 1] = 1.0
    start = st.ent_pos[0, 1].clone()
    for _ in range(3):
        _step(env, walk)
    if timer is None:
        assert float(st.ent_pos[0, 1, 0] - start[0]) == pytest.approx(0.36, abs=1e-5)
        assert float(st.ent_facing[0, 1]) == 0.0
    else:
        assert torch.equal(st.ent_pos[0, 1], start)
        assert float(st.ent_facing[0, 1]) == pytest.approx(math.pi / 2)
        assert bool((st.ent_vel[0, 1] == 0).all())


def test_a_slowed_bot_walks_at_the_reduced_speed_and_still_turns():
    """0.48 tiles in 4 ticks at 2.40 tiles/s, 0.288 under a 0.4 slow, and both end facing +x."""
    def walk(slow):
        env = _env()
        walk_east = _walking_east(_stage(env, [12.0, 10.0]))
        st = env.state
        st.ent_facing[0, 1] = math.pi / 2
        if slow:
            st.ent_slow_t[0, 1], st.ent_slow_frac[0, 1] = 1.0, 0.4
        x0 = float(st.ent_pos[0, 1, 0])
        for _ in range(4):
            _step(env, walk_east)
        return float(st.ent_pos[0, 1, 0]) - x0, float(st.ent_facing[0, 1])

    assert walk(False) == (pytest.approx(0.48, abs=1e-5), 0.0)
    assert walk(True) == (pytest.approx(0.288, abs=1e-5), 0.0)


def test_a_slow_does_not_shorten_mortiss_dash():
    """The slow is a walking speed. A dash covers its 2.67 tiles under it and lands on the same
    float as without it."""
    def dash(slow):
        env = _env()
        override = _stage(env, _FAR, other_at=_FAR_2)
        st = env.state
        if slow:
            st.ent_slow_t[0, 0], st.ent_slow_frac[0, 0] = 1.0, 0.4
        _step(env, override, _ATTACK, move=_EAST)
        for _ in range(7):
            _step(env, override)
        return env.state.ent_pos[0, 0].clone()

    plain, slowed = dash(False), dash(True)
    assert float(plain[0]) == pytest.approx(12.67, abs=1e-4)
    assert torch.equal(plain, slowed)


# ---- attacking -------------------------------------------------------------------------------

@pytest.mark.parametrize("timer", [None, *_STOPS])
def test_a_stunned_or_airborne_bot_cannot_fire_even_on_an_override(timer):
    """An override's fire has no mask in front of it, so `_attack_phase`'s `can_attack` is the
    only gate: free, bot_rifle fires its five pellets and spends one ammo; stopped, nothing."""
    env = _env()
    fire = _stage(env, [15.0, 10.0]).clone()
    fire[0, 1] = torch.tensor([0, 1])
    st = env.state
    st.ent_ammo[0, 1], st.ent_attack_cd[0, 1] = 3.0, 0.0
    if timer:
        getattr(st, timer)[0, 1] = 1.0
    _step(env, fire)
    pellets = int((st.prj_alive[0] & (st.prj_owner[0] == 1)).sum())
    shots = int(st.ent_shots_fired[0, 1])
    if timer is None:
        assert (pellets, shots, float(st.ent_ammo[0, 1])) == (5, 1, pytest.approx(2.0))
    else:
        assert (pellets, shots, float(st.ent_ammo[0, 1])) == (0, 0, 3.0)


@pytest.mark.parametrize("timer", _STOPS)
def test_a_stunned_or_airborne_hero_may_only_hold_fire_and_every_move_stays_legal(timer):
    env = _env()
    _stage(env, _FAR, other_at=_FAR_2)
    st = env.state
    st.ent_super_charge[0, 0] = 5
    free = hero.action_mask(st, env.params, env.cfg)
    assert free["attack"][0].tolist() == [True, True, True, True], "the control: all four open"
    getattr(st, timer)[0, 0] = 0.5
    stopped = hero.action_mask(st, env.params, env.cfg)
    assert stopped["attack"][0].tolist() == [True, False, False, False]
    assert bool(stopped["move"].all())


@pytest.mark.parametrize("attack", [_ATTACK, _SUPER, _GADGET])
@pytest.mark.parametrize("stunned", [False, True])
def test_a_stunned_heros_attack_super_and_gadget_do_nothing(attack, stunned):
    """Through `env.step`, moving east. Free, Mortis dashes (one ammo), fires his super (the
    charge spent, a bolt out) or throws his gadget (its 18.0 s cooldown, a spinner out). Stunned,
    none of it happens and he does not walk either."""
    env = _env()
    override = _stage(env, _FAR, other_at=_FAR_2)
    st = env.state
    st.ent_super_charge[0, 0] = 5
    if stunned:
        st.ent_stun_t[0, 0] = 0.5
    start = st.ent_pos[0, 0].clone()
    _step(env, override, attack, move=_EAST)
    acted = {
        _ATTACK: float(st.ent_ammo[0, 0]) == pytest.approx(2.0) and float(st.ent_dash_t[0, 0]) > 0,
        _SUPER: int(st.ent_super_charge[0, 0]) == 0,
        _GADGET: float(st.ent_gadget_cd[0, 0]) == 18.0,
    }[attack]
    thrown = int((st.prj_alive[0] & (st.prj_owner[0] == 0)).sum())
    if stunned:
        assert not acted and thrown == 0
        assert torch.equal(st.ent_pos[0, 0], start), "a stunned hero walked"
    else:
        assert acted and thrown == (0 if attack == _ATTACK else 1)


@pytest.mark.parametrize("which", ["fire", "super", "gadget"])
def test_the_attack_phase_refuses_a_stunned_heros_request_from_any_source(which):
    """`_attack_phase` is the gate behind every source, overrides included, so it is driven here
    with the request already past the mask: free, each one goes; stunned, none does."""
    for stunned in (False, True):
        env = _env()
        _stage(env, _FAR, other_at=_FAR_2)
        st = env.state
        st.ent_super_charge[0, 0] = 5
        if stunned:
            st.ent_stun_t[0, 0] = 0.5
        no = torch.zeros(1, 3, dtype=torch.bool)
        ask = no.clone()
        ask[0, 0] = True
        move = torch.zeros(1, 3, 2)
        move[0, 0, 0] = 1.0
        env._attack_phase(
            move, ask if which == "fire" else no, ask if which == "super" else no,
            torch.zeros(1, 3, 2), st.ent_pos.clone(), ask if which == "gadget" else no,
            torch.ones(1, 3, 3, dtype=torch.bool))
        acted = {"fire": float(st.ent_dash_t[0, 0]) > 0,
                 "super": int(st.ent_super_charge[0, 0]) == 0,
                 "gadget": float(st.ent_gadget_cd[0, 0]) > 0}[which]
        assert acted != stunned, f"stunned={stunned}"


def _edgar_scene(env, edgar_at):
    """Edgar is bot 1, at `edgar_at` facing +x with a loaded clip. Returns (fire, hold): the
    override that starts his combo on tick 0, and the one that holds both bots after it."""
    hold = _stage(env, edgar_at)
    st = env.state
    assert int(st.ent_kind[0, 1]) == int(Kind.BOT_EDGAR)
    st.ent_ammo[0, 1], st.ent_attack_cd[0, 1] = 3.0, 0.0
    fire = hold.clone()
    fire[0, 1] = torch.tensor([0, 1])
    return fire, hold


@pytest.mark.parametrize("stun, landed", [(0.0, [0, 5]), (0.5, [0]), (0.1, [0, 5])])
def test_a_stun_drops_the_sub_swings_inside_it_and_not_the_ones_after(stun, landed):
    """Edgar's combo, 1.2 tiles from the hero, lands on ticks 0 and 5. A 0.5 s stun from the
    first hit covers tick 5 and drops the second swing; a 0.1 s one is over by tick 2, and the
    second still lands on tick 5: the schedule runs off `ent_attack_cd`, which a stun does not
    pause."""
    env = _env(enemies=("edgar", "rifle"))
    fire, hold = _edgar_scene(env, [8.8, 10.0])
    st = env.state
    hits = []
    for t in range(16):
        before = float(st.ent_hp[0, 0])
        _step(env, fire if t == 0 else hold)
        if t == 0 and stun:
            st.ent_stun_t[0, 1] = stun
        lost = before - float(st.ent_hp[0, 0])
        if lost > 1.0:
            assert lost == pytest.approx(1080.0)
            hits.append(t)
    assert hits == landed


# ---- airborne: nothing lands on it -----------------------------------------------------------

@pytest.mark.parametrize("airborne", [False, True])
def test_a_pellet_flies_through_an_airborne_unit_to_the_one_behind(airborne):
    """Bull's volley at two bots in a row, 2 and 4 tiles along his facing. All five pellets stop
    in the first (5 x 880); with it in the air they pass it, and the middle three of the 0.5 rad
    fan land on the second (3 x 880). Both sit 0.05 off the exact facing ray: `melee_hitscan` runs
    for every non-dash attack, and a zero `attack_arc_rad` still catches a body exactly on it."""
    env = _env(overrides=BULL)
    override = _stage(env, [12.0, 10.05], other_at=[14.0, 10.05])
    st = env.state
    if airborne:
        st.ent_airborne_t[0, 1] = 2.0
    hp = st.ent_hp[0].clone()
    _step(env, override, _ATTACK)
    for _ in range(19):
        _step(env, override)
    lost = (hp - st.ent_hp[0]).tolist()
    assert lost == ([0.0, 0.0, 3 * 880.0] if airborne else [0.0, 5 * 880.0, 0.0])


@pytest.mark.parametrize("airborne", [False, True])
def test_a_blast_skips_an_airborne_unit_and_hits_the_one_beside_it(airborne):
    """Mortis's spinner lands on bot 1, 1.5 tiles out, on the 4th tick, for 2000 in a 1.0 blast,
    with bot 2 0.7 from the landing. It is still thrown at an airborne bot (`gadget_target` does
    not skip one), and the blast hits only the bot on the ground."""
    env = _env()
    override = _stage(env, [11.5, 10.0], other_at=[11.5, 10.7])
    st = env.state
    if airborne:
        st.ent_airborne_t[0, 1] = 1.0
    hp = st.ent_hp[0].clone()
    _step(env, override, _GADGET)
    for _ in range(4):
        _step(env, override)
    lost = (hp - st.ent_hp[0]).tolist()
    assert lost == ([0.0, 0.0, 2000.0] if airborne else [0.0, 2000.0, 2000.0])


@pytest.mark.parametrize("airborne", [False, True])
def test_a_melee_cone_skips_an_airborne_target(airborne):
    """Edgar's whole combo at the hero, 1.2 tiles away: 2 x 1080, or nothing while he is in the
    air."""
    env = _env(enemies=("edgar", "rifle"))
    fire, hold = _edgar_scene(env, [8.8, 10.0])
    st = env.state
    if airborne:
        st.ent_airborne_t[0, 0] = 1.0
    hp = float(st.ent_hp[0, 0])
    for t in range(12):
        _step(env, fire if t == 0 else hold)
    assert hp - float(st.ent_hp[0, 0]) == (0.0 if airborne else 2 * 1080.0)


@pytest.mark.parametrize("airborne, hit_on", [(0.0, 1), (0.15, 2), (1.0, None)])
def test_a_dash_passes_an_airborne_unit_and_hits_it_if_it_lands_inside(airborne, hit_on):
    """Mortis dashes east through bot 1, 1.5 tiles out. The capsule first holds it on the dash's
    second tick (1); in the air through that tick, it is hit on the next while still inside; in
    the air for the whole dash, never. One hit of 2000 at most: a dash hits a unit once."""
    env = _env()
    override = _stage(env, [11.5, 10.0])
    st = env.state
    if airborne:
        st.ent_airborne_t[0, 1] = airborne
    hits = []
    for t in range(8):
        before = float(st.ent_hp[0, 1])
        _step(env, override, _ATTACK if t == 0 else 0, move=_EAST if t == 0 else 0)
        lost = before - float(st.ent_hp[0, 1])
        if lost > 1.0:
            hits.append((t, lost))
    assert hits == ([] if hit_on is None else [(hit_on, 2000.0)])


# ---- airborne: nothing picks it --------------------------------------------------------------

def _volley_dir(env):
    """The fan is symmetric, so the summed velocity of the hero's five pellets points down the aim."""
    st = env.state
    mine = st.prj_alive[0] & (st.prj_owner[0] == 0)
    assert int(mine.sum()) == 5, "one volley of five pellets"
    v = st.prj_vel[0][mine].sum(dim=0)
    return v / v.norm()


@pytest.mark.parametrize("airborne", [False, True])
def test_auto_aim_skips_an_airborne_enemy(airborne):
    """Bull's value 4 with bot 1 two tiles out at 60 degrees goes at it; with it in the air, down
    the move bin (+x), as with nothing in reach."""
    env = _env(overrides=_deep_merge(BULL, {"action": {"auto_aim": True}}))
    at = [10.0 + 2.0 * math.cos(math.pi / 3), 10.0 + 2.0 * math.sin(math.pi / 3)]
    override = _stage(env, at)
    if airborne:
        env.state.ent_airborne_t[0, 1] = 1.0
    _step(env, override, _AUTO, move=_EAST)
    want = [1.0, 0.0] if airborne else [0.5, math.sqrt(3.0) / 2]
    assert torch.allclose(_volley_dir(env), torch.tensor(want), atol=1e-4)


@pytest.mark.parametrize("airborne", [False, True])
def test_an_airborne_enemy_is_not_in_reach(airborne):
    """Mortis attacking with bot 1 three tiles out (inside his 3.77) earns the in-reach flag,
    unless bot 1 is in the air. The attack itself goes either way."""
    env = _env()
    override = _stage(env, [13.0, 10.0])
    if airborne:
        env.state.ent_airborne_t[0, 1] = 1.0
    *_, info = _step(env, override, _ATTACK, move=_EAST)
    assert int(env.state.ent_shots_fired[0, 0]) == 1
    assert int(info["attack_in_reach_tick"][0]) == (0 if airborne else 1)


@pytest.mark.parametrize("airborne", [False, True])
def test_the_idle_super_tap_skips_an_airborne_enemy(airborne):
    """Mortis facing +x with bot 1 three tiles north: his idle super goes at it, or along his
    facing while it is in the air. Bot 2 sits past the bolt's 11.1-tile reach."""
    env = _env()
    override = _stage(env, [10.0, 13.0], other_at=[1.5, 1.5])
    st = env.state
    st.ent_super_charge[0, 0] = 5
    if airborne:
        st.ent_airborne_t[0, 1] = 1.0
    _step(env, override, _SUPER)
    mine = st.prj_alive[0] & (st.prj_owner[0] == 0)
    assert int(mine.sum()) == 1, "one bolt"
    v = st.prj_vel[0][mine][0]
    want = [1.0, 0.0] if airborne else [0.0, 1.0]
    assert torch.allclose(v / v.norm(), torch.tensor(want), atol=1e-5)


# ---- state contract --------------------------------------------------------------------------

@pytest.mark.parametrize("fields, match", [
    ({"ent_stun_t": -0.05}, "ent_stun_t is negative"),
    ({"ent_airborne_t": -0.05}, "ent_airborne_t is negative"),
    ({"ent_slow_t": -0.05}, "ent_slow_t is negative"),
    ({"ent_slow_t": 1.0, "ent_slow_frac": 1.5}, r"ent_slow_frac out of \[0, 1\]"),
    ({"ent_slow_frac": 0.4}, "ent_slow_frac set with no ent_slow_t"),
    ({"ent_slow_window_t": -0.05}, r"ent_slow_window_t out of \[0, super_slow_seconds\]"),
    # A bot has no CHARGE super, so no landing slow and no window either.
    ({"ent_slow_window_t": 0.05}, r"ent_slow_window_t out of \[0, super_slow_seconds\]"),
])
def test_check_invariants_rejects_a_broken_status_timer(fields, match):
    env = _env()
    st = env.state
    check_invariants(st, env.cfg, env.params)
    for name, value in fields.items():
        getattr(st, name)[0, 1] = value
    with pytest.raises(ValueError, match=match):
        check_invariants(st, env.cfg, env.params)


def test_an_episode_that_ends_mid_effect_starts_the_next_one_clear():
    """The autoreset path training runs: every timer is set long on the last step of a 3-step
    episode, and the fresh episode starts with all four at zero."""
    env = _env(overrides={"sim": {"max_episode_steps": 3}})
    override = _stage(env, _FAR, other_at=_FAR_2)
    st = env.state
    for _ in range(2):
        _step(env, override)
    for name in ("ent_slow_t", "ent_stun_t", "ent_airborne_t"):
        getattr(st, name).fill_(3.0)
    st.ent_slow_frac.fill_(0.4)
    _obs, _reward, terminated, truncated, _info = _step(env, override)
    assert bool(terminated[0] | truncated[0]), "the episode ended on its third step"
    for name in ("ent_slow_t", "ent_slow_frac", "ent_stun_t", "ent_airborne_t"):
        assert bool((getattr(st, name) == 0).all()), name
