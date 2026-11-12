"""The two miss costs (user decision 2026-11-02, BULL_HERO_PLAN.md §5.8): `attack_miss`, a volley of
Bull's whose pellets all ended without landing on a player or a crate, and `gadget_miss`, a stomp
that landed on nobody.

Literals, not params. dt is 0.05 s. Bull's volley is 5 pellets fanned over 0.5 rad, flying
9 tiles/s for 5.2 tiles: 0.45 a tick, so a pellet that touches nothing runs out on its 12th step
(11 steps cover 4.95). A shot on tick t spawns in the attack phase and steps that same tick, so its
miss is scored on tick t + 11. His attack cooldown is 0.25 s, 5 ticks. A stomp tapped on tick t
lands on tick t + 6. Move value 1 is +x and 5 is +y; attack values are 1 attack, 3 gadget.
"""
import math
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import _deep_merge, load_config
from brawl_sim.core.state import HERO_VOLLEY_RING
from brawl_sim.env import BrawlVecEnv
from brawl_sim.training.config import RewardConfig, load_train_config
from brawl_sim.training.reward import ShapedReward

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
TINY = yaml.safe_load((CONFIGS / "presets" / "debug_tiny.yaml").read_text())

_ATTACK, _GADGET = 1, 3
_EAST, _SOUTH = 1, 5


def _spec(overrides) -> dict:
    spec = {
        **yaml.safe_load((CONFIGS / "default.yaml").read_text()),
        **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text()),
    }
    return _deep_merge(spec, overrides)


def _duel(hero="bull", bot1=(3.0, 3.0), bot2=(3.0, 17.0), action_repeat=1):
    """The hero at (10, 10) on the 20x20 blank map, two rifle bots frozen by the override (the
    hero row's -1 lets his action through), no crates, regen off, invariants checked every step."""
    over = _deep_merge(TINY, {
        "entities": {"hero": hero, "randomize_enemy_types": False, "fixed_enemy_types": ["rifle", "rifle"]},
        "sim": {"action_repeat": action_repeat, "max_episode_steps": 2000},
        "regen": {"enabled": False},
        "engine": {"debug_checks": True},
    })
    cfg = load_config(CONFIGS / "default.yaml", overrides=over)
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, spec=_spec(over), verbose=False)
    env.reset()
    st = env.state
    for slot, xy in enumerate(((10.0, 10.0), bot1, bot2)):
        st.ent_pos[0, slot] = torch.tensor(xy)
    st.ent_facing.zero_()
    st.box_alive.fill_(False)
    override = torch.zeros(1, env.cfg.n_entities, 2, dtype=torch.int64)
    override[:, 0, 0] = -1
    return env, override


def _run(env, override, script, ticks):
    """Steps `ticks` times, `script[tick]` = (move, attack) on that 1-based tick and idle
    otherwise. Returns the per-step attack_miss_tick, gadget_miss_tick and gadget_hit_tick."""
    out = {"attack_miss": [], "gadget_miss": [], "gadget_hit": []}
    for tick in range(1, ticks + 1):
        move, attack = script.get(tick, (0, 0))
        *_, info = env.step(torch.tensor([[move, attack]], dtype=torch.int64), override)
        for key in out:
            out[key].append(int(info[f"{key}_tick"][0]))
    return out


def _hero_pellets(env) -> int:
    st = env.state
    return int((st.prj_alive & (st.prj_owner == 0)).sum())


# ---- attack_miss --------------------------------------------------------------------------------

def test_a_volley_into_nothing_is_one_miss_on_the_tick_its_last_pellet_runs_out():
    env, override = _duel()
    got = _run(env, override, {1: (_EAST, _ATTACK)}, 16)
    assert got["attack_miss"] == [0] * 11 + [1] + [0] * 4
    assert _hero_pellets(env) == 0
    assert not bool(env.state.hero_volley_hit.any()), "a closed volley left its column set"


def test_a_volley_that_lands_one_pellet_on_a_player_is_no_miss():
    """Bot 1 three tiles out on the centre line eats the middle three pellets on tick 6; the
    outer two fly past and run out on tick 12, and the volley is still not charged."""
    env, override = _duel(bot1=(13.0, 10.0))
    hp = float(env.state.ent_hp[0, 1])
    got = _run(env, override, {1: (_EAST, _ATTACK)}, 16)
    assert float(env.state.ent_hp[0, 1]) < hp
    assert _hero_pellets(env) == 0
    assert got["attack_miss"] == [0] * 16
    assert not bool(env.state.hero_volley_hit.any())


def test_a_volley_that_lands_only_on_a_crate_is_no_miss():
    """The user's "(or power box)": a crate on the centre line counts as a landing."""
    env, override = _duel()
    st = env.state
    st.box_pos[0, 0] = torch.tensor([12.5, 10.0])
    st.box_alive[0, 0] = True
    st.box_hp[0, 0] = st.box_max_hp[0, 0] = 3000.0
    got = _run(env, override, {1: (_EAST, _ATTACK)}, 16)
    assert float(st.box_hp[0, 0]) < 3000.0
    assert got["attack_miss"] == [0] * 16


def test_two_volleys_in_flight_together_are_scored_apart():
    """Volley A (tick 1, east) lands its middle pellets on bot 1, four tiles out, on tick 7 and
    its outer two fly on to tick 12. Volley B (tick 6, south) leaves while all of A is still in
    the air and hits nothing: B alone is charged, on tick 6 + 11. A shared hit flag would have
    excused B; a shared pellet count would have scored neither until both were down."""
    env, override = _duel(bot1=(14.0, 10.0))
    got = {"attack_miss": []}
    for tick in range(1, 21):
        before = _hero_pellets(env)
        move, attack = {1: (_EAST, _ATTACK), 6: (_SOUTH, _ATTACK)}.get(tick, (0, 0))
        *_, info = env.step(torch.tensor([[move, attack]], dtype=torch.int64), override)
        got["attack_miss"].append(int(info["attack_miss_tick"][0]))
        if tick == 6:
            assert before > 0 and _hero_pellets(env) == before + 5, "B did not fire beside A"
    assert got["attack_miss"] == [0] * 16 + [1] + [0] * 3


def test_at_the_shipped_decision_rate_the_miss_is_paid_on_the_third_decision():
    """At action_repeat 5 the shot is on tick 1 and its last pellet runs out on tick 12, the
    second sub-tick of decision 3."""
    env, override = _duel(action_repeat=5)
    got = _run(env, override, {1: (_EAST, _ATTACK)}, 5)
    assert got["attack_miss"] == [0, 0, 1, 0, 0]


def test_mortis_never_scores_a_miss():
    """His attack is a dash and his gadget a spinner: neither fires a pellet or lands a stomp, so
    both counts stay 0 when both whiff."""
    env, override = _duel(hero="mortis")
    got = _run(env, override, {1: (_EAST, _ATTACK), 12: (0, _GADGET)}, 40)
    assert got["attack_miss"] == [0] * 40
    assert got["gadget_miss"] == [0] * 40


# ---- gadget_miss --------------------------------------------------------------------------------

@pytest.mark.parametrize("bot1, airborne, hit, miss", [
    ((11.0, 10.0), False, 1, 0),     # a player caught: a hit
    ((3.0, 3.0), False, 0, 1),       # nobody within reach: a miss
    ((11.0, 10.0), True, 0, 1),      # only a player in the air, who is skipped: a miss
])
def test_a_stomp_that_lands_on_nobody_is_one_miss_on_its_landing_tick(bot1, airborne, hit, miss):
    env, override = _duel(bot1=bot1)
    if airborne:
        env.state.ent_airborne_t[0, 1] = 1.0
    got = _run(env, override, {1: (0, _GADGET)}, 9)
    assert got["gadget_hit"] == [0] * 6 + [hit] + [0] * 2
    assert got["gadget_miss"] == [0] * 6 + [miss] + [0] * 2


def test_a_stomp_is_no_attack_miss_and_a_volley_is_no_gadget_miss():
    env, override = _duel()
    got = _run(env, override, {1: (0, _GADGET), 2: (_EAST, _ATTACK)}, 16)
    assert sum(got["attack_miss"]) == 1
    assert sum(got["gadget_miss"]) == 1


# ---- the reward terms ---------------------------------------------------------------------------

_ONLY_MISSES = dict(win_bonus=0.0, death_penalty=0.0, rank_bonus=0.0, damage_dealt=0.0,
                    damage_taken=0.0, hp_healed=0.0, kill=0.0, cube_pickup=0.0,
                    survive_per_step=0.0, in_zone_per_step=0.0, attack_in_reach=0.0,
                    gadget_hit=0.0, move_reversal=0.0, attack_miss=-0.02, gadget_miss=-0.1)


def _info(attack_misses, gadget_misses):
    n = len(attack_misses)
    return {
        "terminated": torch.zeros(n, dtype=torch.bool), "truncated": torch.zeros(n, dtype=torch.bool),
        "hero_rank": torch.ones(n, dtype=torch.int64), "hero_alive": torch.ones(n, dtype=torch.bool),
        "attack_miss_tick": torch.tensor(attack_misses, dtype=torch.int32),
        "gadget_miss_tick": torch.tensor(gadget_misses, dtype=torch.int32),
    }


def test_each_term_charges_its_weight_per_miss_and_reports_its_mean():
    fn = ShapedReward(RewardConfig(**_ONLY_MISSES), track_terms=True)
    reward = fn({}, _info([0, 1, 2], [1, 0, 1]), None)
    assert reward.tolist() == pytest.approx([-0.1, -0.02, -0.14])
    assert fn.term_means() == pytest.approx({"attack_miss": -0.06 / 3, "gadget_miss": -0.2 / 3})


def test_off_by_default_and_then_never_read():
    assert RewardConfig().attack_miss == 0.0 and RewardConfig().gadget_miss == 0.0
    fn = ShapedReward(RewardConfig(**{**_ONLY_MISSES, "attack_miss": 0.0, "gadget_miss": 0.0}))
    info = _info([1], [1])
    del info["attack_miss_tick"], info["gadget_miss_tick"]
    assert fn({}, info, None).tolist() == [0.0]


def test_train_yaml_charges_a_stomp_more_than_a_volley():
    reward = load_train_config(CONFIGS / "train.yaml").reward
    assert reward.gadget_miss < reward.attack_miss < 0.0


def test_the_volley_ring_outlasts_every_hero_pellet():
    """`hero_volley_hit` keys a volley by its spawn tick mod HERO_VOLLEY_RING; a pellet still
    flying when its column comes round again would merge two volleys."""
    dt = yaml.safe_load((CONFIGS / "default.yaml").read_text())["sim"]["dt"]
    blocks = yaml.safe_load((CONFIGS / "brawlers.yaml").read_text())
    heroes = {name: b for name, b in blocks.items()
              if name.startswith("hero_") and isinstance(b, dict) and b.get("proj_count", 0) > 0}
    assert "hero_bull" in heroes
    for name, b in heroes.items():
        flight_ticks = math.ceil(b["attack_range"] / (b["proj_speed"] * dt))
        assert flight_ticks < HERO_VOLLEY_RING, name
