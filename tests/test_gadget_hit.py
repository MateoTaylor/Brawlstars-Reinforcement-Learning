"""BRAWL_SIM_DESIGN.md §10 (user decision, 2026-09-30): the `gadget_hit` shaping term.

One count per landing of the hero's gadget spinner that hurt at least one PLAYER, paid by
training/reward.py. A spinner that lands only on crates, or on nothing, counts nothing, and so
does every other kind of hero damage. It is counted where the spinner lands, not where it is
thrown: 0.2 s of flight is 4 ticks, the throw tick included.

The first half drives the count through `BrawlVecEnv.step`; the second prices it through
`ShapedReward`. Numbers are pinned as literals from the shipped Mortis block (`gadget_range 2.0`,
`gadget_flight_seconds 0.2`, `gadget_damage 2000`, `gadget_radius 1.0`, `super_charge_hits 5`)
and from configs/train.yaml's `0.3`; a test that derived them from `params` could not fail.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.training.config import RewardConfig, load_train_config
from brawl_sim.training.reward import ShapedReward

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

_ATTACK = 1      # attack-column values, as literals: importing them would hide a renumbering
_SUPER = 2
_GADGET = 3
_SPINNER = 7     # Proj.GADGET_SPINNER
_FAR = [3.0, 3.0]


def _env(action_repeat=1, max_episode_steps=2000, autoreset=True, reward_fn=None):
    from brawl_sim.env import BrawlVecEnv

    tiny = yaml.safe_load((CONFIGS / "presets" / "debug_tiny.yaml").read_text())
    sim = {**tiny["sim"], "action_repeat": action_repeat, "max_episode_steps": max_episode_steps}
    cfg = load_config(CONFIGS / "default.yaml", overrides={
        **tiny, "sim": sim, "engine": {"debug_checks": True, "compile": False},
    })
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, verbose=False,
                      autoreset=autoreset, reward_fn=reward_fn)
    env.reset()
    return env


def _duel(enemy_at, other_at=None, **env_kwargs):
    """Hero at (10,10) facing +x, bot 1 at `enemy_at`, bot 2 at `other_at` (far away by default),
    every crate dead, both bots frozen by the override. Returns `(env, override, hp_before (E,))`;
    HP is compared as a delta, since the tick clamps it to the kind's effective max."""
    env = _env(**env_kwargs)
    st = env.state
    assert env.cfg.n_entities == 3
    st.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    st.ent_pos[0, 1] = torch.tensor(enemy_at)
    st.ent_pos[0, 2] = torch.tensor(other_at if other_at is not None else _FAR)
    st.ent_facing.zero_()
    st.box_alive.fill_(False)
    override = torch.zeros(1, env.cfg.n_entities, 2, dtype=torch.int64)
    override[:, 0, 0] = -1
    hp_before = st.ent_hp[0].clone()
    assert bool((hp_before > 2000.0).all()), "a 2000 hit must not be clipped by a kill"
    return env, override, hp_before


def _act(env, override, attack, move=0):
    return env.step(torch.tensor([[move, attack]], dtype=torch.int64), override)


def _throw_and_land(env, override):
    """Gadget on step 1, idle after. Returns three per-step lists: the `gadget_hit_tick` counts,
    the rewards, and the spinners still in flight after the step."""
    counts, rewards, flying = [], [], []
    for step in range(5):
        _obs, reward, _terminated, _truncated, info = _act(env, override,
                                                           _GADGET if step == 0 else 0)
        counts.append(int(info["gadget_hit_tick"][0]))
        rewards.append(float(reward[0]))
        st = env.state
        flying.append(int((st.prj_alive[0] & (st.prj_kind[0] == _SPINNER)).sum()))
    return counts, rewards, flying


def _hp_lost(env, hp_before):
    return (hp_before - env.state.ent_hp[0]).tolist()


# ---- the count, through env.step ----------------------------------------------------------

def test_a_spinner_that_lands_on_an_enemy_counts_once_on_its_landing_step():
    env, override, hp_before = _duel([11.5, 10.0])
    counts, _rewards, flying = _throw_and_land(env, override)
    assert flying == [1, 1, 1, 0, 0]            # detonated on step 4...
    assert counts == [0, 0, 0, 1, 0]            # ...which is where it counts, once
    assert _hp_lost(env, hp_before) == [0.0, 2000.0, 0.0]


def test_two_enemies_in_one_blast_count_once():
    """Per landing, not per player: `damage_dealt` already pays for the second one."""
    env, override, hp_before = _duel([11.5, 10.0], other_at=[11.5, 10.7])
    counts, _rewards, _flying = _throw_and_land(env, override)
    assert counts == [0, 0, 0, 1, 0]
    assert _hp_lost(env, hp_before) == [0.0, 2000.0, 2000.0]


def _crate_scene(with_crate):
    """Bot 1 six tiles out on +x is the nearest enemy the hero sees, so the spinner flies its
    2.0-tile range toward him and lands at (12, 10), four tiles short of him. A crate, if any,
    sits 0.6 from that point, inside the 1.0 blast."""
    env, override, hp_before = _duel([16.0, 10.0])
    if with_crate:
        st = env.state
        st.box_pos[0, 0] = torch.tensor([12.0, 10.6])
        st.box_alive[0, 0] = True
        st.box_hp[0, 0] = 3000.0
    return env, override, hp_before


def test_a_spinner_that_lands_only_on_a_crate_counts_nothing():
    env, override, hp_before = _crate_scene(with_crate=True)
    counts, _rewards, flying = _throw_and_land(env, override)
    assert flying == [1, 1, 1, 0, 0]
    assert env.state.box_hp[0, 0].item() == 1000.0          # the crate took the blast...
    assert _hp_lost(env, hp_before) == [0.0, 0.0, 0.0]      # ...and no player did
    assert counts == [0, 0, 0, 0, 0]


def test_a_spinner_that_lands_on_nothing_counts_nothing():
    env, override, hp_before = _crate_scene(with_crate=False)
    counts, _rewards, flying = _throw_and_land(env, override)
    assert flying == [1, 1, 1, 0, 0]
    assert _hp_lost(env, hp_before) == [0.0, 0.0, 0.0]
    assert counts == [0, 0, 0, 0, 0]


@pytest.mark.parametrize("attack", [_ATTACK, _SUPER])
def test_the_heros_other_hits_are_not_gadget_hits(attack):
    """The attack's dash deals its damage outside the projectile phase. The super's bolt is a
    projectile, but its hits charge the super, which is what tells them apart from the
    spinner's."""
    env, override, hp_before = _duel([11.5, 10.0])
    env.state.ent_super_charge[0, 0] = 5                    # super_charge_hits 5: a full meter
    counts = []
    for step in range(3):
        *_, info = _act(env, override, attack if step == 0 else 0)
        counts.append(int(info["gadget_hit_tick"][0]))
    assert _hp_lost(env, hp_before)[1] > 0.0                # the enemy was hit
    assert counts == [0, 0, 0]


def test_a_landing_on_a_later_sub_tick_counts_under_action_repeat():
    """One decision is 5 sub-ticks, so the throw (sub-tick 1) and the landing (sub-tick 4) share
    it: the count must be summed over the sub-ticks, not read off the first."""
    env, override, hp_before = _duel([11.5, 10.0], action_repeat=5)
    *_, info = _act(env, override, _GADGET)
    assert info["gadget_hit_tick"].dtype == torch.int32
    assert info["gadget_hit_tick"].tolist() == [1]
    assert _hp_lost(env, hp_before) == [0.0, 2000.0, 0.0]


def test_a_landing_after_the_episode_ended_mid_decision_does_not_count():
    """The tick cap is crossed on sub-tick 1. The world keeps ticking to the end of the decision
    and the spinner does land on sub-tick 4, outside the episode; the same `live` gate that stops
    damage and kills from counting there stops this count too."""
    env, override, hp_before = _duel([11.5, 10.0], action_repeat=5, autoreset=False)
    env.state.step_count.fill_(1999)
    *_, info = _act(env, override, _GADGET)
    assert bool(info["truncated"][0])
    assert int(info["n_ticks"][0]) == 1
    assert _hp_lost(env, hp_before) == [0.0, 2000.0, 0.0]
    assert info["gadget_hit_tick"].tolist() == [0]


# ---- the price, through ShapedReward ---------------------------------------------------------

_ONLY_GADGET = dict(win_bonus=0.0, death_penalty=0.0, rank_bonus=0.0, damage_dealt=0.0,
                    damage_taken=0.0, hp_healed=0.0, kill=0.0, cube_pickup=0.0,
                    survive_per_step=0.0, in_zone_per_step=0.0, attack_in_reach=0.0,
                    gadget_hit=0.3)


def _info(counts):
    """The keys ShapedReward reads whatever its weights, plus this term's own."""
    n = len(counts)
    return {
        "terminated": torch.zeros(n, dtype=torch.bool),
        "truncated": torch.zeros(n, dtype=torch.bool),
        "hero_rank": torch.zeros(n, dtype=torch.int64),
        "gadget_hit_tick": torch.tensor(counts, dtype=torch.int32),
    }


def test_the_term_pays_its_weight_per_counted_landing_and_reports_its_mean():
    fn = ShapedReward(RewardConfig(**_ONLY_GADGET), track_terms=True)
    reward = fn({}, _info([1, 0, 1]), SimpleNamespace(n_entities=3))
    assert reward.tolist() == pytest.approx([0.3, 0.0, 0.3])
    assert fn.term_means() == pytest.approx({"gadget_hit": 0.6 / 3})


def test_off_by_default_and_then_never_read():
    """A hand-built RewardConfig keeps its meaning, and an `info` without the key (any caller of
    compute_info that predates it) is not a KeyError."""
    assert RewardConfig().gadget_hit == 0.0
    fn = ShapedReward(RewardConfig(**{**_ONLY_GADGET, "gadget_hit": 0.0}), track_terms=True)
    info = _info([1, 1, 1])
    del info["gadget_hit_tick"]
    assert fn({}, info, SimpleNamespace(n_entities=3)).tolist() == [0.0, 0.0, 0.0]
    assert fn.term_means() == {}


def test_train_yaml_ships_it_at_0_3():
    assert load_train_config(CONFIGS / "train.yaml").reward.gadget_hit == 0.3


def test_through_the_env_the_landing_decision_pays_0_3_and_the_throw_nothing():
    fn = ShapedReward(RewardConfig(**_ONLY_GADGET), track_terms=False)
    env, override, _hp = _duel([11.5, 10.0], reward_fn=fn)
    _counts, rewards, _flying = _throw_and_land(env, override)
    assert rewards == pytest.approx([0.0, 0.0, 0.0, 0.3, 0.0])
