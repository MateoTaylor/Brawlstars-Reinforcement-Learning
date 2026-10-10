"""SIM_ISSUES_PLAN.md §5.2 (user decision, 2026-10-07): the `move_reversal` shaping term.

One count per decision whose move bin swung at least 135 degrees from the previous decision's,
both non-idle, paid by training/reward.py. The previous decision is read off the history ring
before `env.step` pushes this one, so the first decision of an episode never counts.

The first part checks the rule against angles computed independently of the bin arithmetic; the
second drives it through `BrawlVecEnv.step`; the third prices it through `ShapedReward`. The
-0.01 is pinned as a literal from configs/train.yaml.
"""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.core import events
from brawl_sim.training.config import RewardConfig, load_train_config
from brawl_sim.training.reward import ShapedReward

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

_IDLE = 0
_E, _W = 1, 9          # bins 1 and 9 of 16: opposite directions


# ---- the rule ------------------------------------------------------------------------------

def _rule(prev, cur, n_bins, valid=True):
    state = SimpleNamespace(
        hist_valid=torch.tensor([[valid]] * len(prev)),
        hist_action=torch.tensor([[[p, 0]] for p in prev], dtype=torch.int64),
    )
    action = torch.tensor([[c, 0] for c in cur], dtype=torch.int64)
    return events.move_reversals(state, action, SimpleNamespace(n_move_bins=n_bins))


@pytest.mark.parametrize("n_bins", [8, 16])
def test_a_reversal_is_a_swing_of_at_least_135_degrees_between_two_moves(n_bins):
    """Every (previous, current) pair, against the angle between the two directions measured in
    degrees, so the expectation shares nothing with the bin arithmetic under test."""
    pairs = [(p, c) for p in range(n_bins + 1) for c in range(n_bins + 1)]
    got = _rule([p for p, _ in pairs], [c for _, c in pairs], n_bins)
    assert got.dtype == torch.int32

    def angle(k):
        return (k - 1) * 360.0 / n_bins

    for (p, c), flag in zip(pairs, got.tolist()):
        swing = abs((angle(p) - angle(c) + 180.0) % 360.0 - 180.0)
        expect = p != _IDLE and c != _IDLE and swing >= 135.0 - 1e-9
        assert flag == int(expect), (p, c, swing)


def test_the_edges_at_16_bins():
    """135 degrees counts and 112.5 does not, around the wrap at bin 16 as anywhere else."""
    prev = [1, 1, 1, 16, 16, 2]
    cur = [9, 7, 6, 6, 5, 12]
    assert _rule(prev, cur, 16).tolist() == [1, 1, 0, 1, 0, 1]


def test_nothing_counts_without_a_previous_decision():
    assert _rule([_E], [_W], 16, valid=False).tolist() == [0]


# ---- the count, through env.step ----------------------------------------------------------

def _env(action_repeat=1, reward_fn=None):
    from brawl_sim.env import BrawlVecEnv

    tiny = yaml.safe_load((CONFIGS / "presets" / "debug_tiny.yaml").read_text())
    sim = {**tiny["sim"], "action_repeat": action_repeat, "max_episode_steps": 2000}
    cfg = load_config(CONFIGS / "default.yaml", overrides={
        **tiny, "sim": sim, "engine": {"debug_checks": True, "compile": False},
    })
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, verbose=False, reward_fn=reward_fn)
    env.reset()
    return env


def _walk(env, moves, before=None):
    """One decision per move bin, bots held idle by the override. `before(env, step)` runs ahead
    of each step. Returns the per-decision `move_reversal_tick` flags and rewards."""
    override = torch.zeros(1, env.cfg.n_entities, 2, dtype=torch.int64)
    override[:, 0, 0] = -1
    flags, rewards = [], []
    for step, move in enumerate(moves):
        if before is not None:
            before(env, step)
        _obs, reward, _terminated, _truncated, info = env.step(
            torch.tensor([[move, 0]], dtype=torch.int64), override)
        flags.append(int(info["move_reversal_tick"][0]))
        rewards.append(float(reward[0]))
    return flags, rewards


def test_each_decision_is_compared_with_the_one_before_it():
    """Read before the history push: read after it, every decision would compare with itself
    and nothing would ever count."""
    flags, _ = _walk(_env(), [_E, _W, _W, 3, _IDLE, _W, _E])
    assert flags == [0, 1, 0, 1, 0, 0, 1]


def test_once_per_decision_under_action_repeat():
    flags, _ = _walk(_env(action_repeat=5), [_E, _W, _E])
    assert flags == [0, 1, 1]


def test_the_first_decision_after_a_reset_counts_nothing():
    """The second decision ends the episode at the tick cap. The third is the new episode's first:
    it reverses the second's move, but the reset cleared the ring that remembered it."""
    def cap_on_step_1(env, step):
        if step == 1:
            env.state.step_count.fill_(env.cfg.max_episode_steps - 1)

    flags, _ = _walk(_env(), [_E, _W, _E, _W], before=cap_on_step_1)
    assert flags == [0, 1, 0, 1]


# ---- the price, through ShapedReward ---------------------------------------------------------

_ONLY_REVERSAL = dict(win_bonus=0.0, death_penalty=0.0, rank_bonus=0.0, damage_dealt=0.0,
                      damage_taken=0.0, hp_healed=0.0, kill=0.0, cube_pickup=0.0,
                      survive_per_step=0.0, in_zone_per_step=0.0, attack_in_reach=0.0,
                      gadget_hit=0.0, move_reversal=-0.01)


def _info(flags):
    """The keys ShapedReward reads whatever its weights, plus this term's own."""
    n = len(flags)
    return {
        "terminated": torch.zeros(n, dtype=torch.bool),
        "truncated": torch.zeros(n, dtype=torch.bool),
        "hero_rank": torch.zeros(n, dtype=torch.int64),
        "move_reversal_tick": torch.tensor(flags, dtype=torch.int32),
    }


def test_the_term_charges_its_weight_per_reversal_and_reports_its_mean():
    fn = ShapedReward(RewardConfig(**_ONLY_REVERSAL), track_terms=True)
    reward = fn({}, _info([1, 0, 1]), SimpleNamespace(n_entities=3))
    assert reward.tolist() == pytest.approx([-0.01, 0.0, -0.01])
    assert fn.term_means() == pytest.approx({"move_reversal": -0.02 / 3})


def test_off_by_default_and_then_never_read():
    """A hand-built RewardConfig keeps its meaning, and an `info` without the key (any caller of
    compute_info that predates it) is not a KeyError."""
    assert RewardConfig().move_reversal == 0.0
    fn = ShapedReward(RewardConfig(**{**_ONLY_REVERSAL, "move_reversal": 0.0}), track_terms=True)
    info = _info([1, 1, 1])
    del info["move_reversal_tick"]
    assert fn({}, info, SimpleNamespace(n_entities=3)).tolist() == [0.0, 0.0, 0.0]
    assert fn.term_means() == {}


def test_train_yaml_ships_it_at_minus_0_01():
    assert load_train_config(CONFIGS / "train.yaml").reward.move_reversal == -0.01


def test_through_the_env_only_the_reversing_decisions_are_charged():
    fn = ShapedReward(RewardConfig(**_ONLY_REVERSAL), track_terms=False)
    _flags, rewards = _walk(_env(reward_fn=fn), [_E, _W, _W, 3, _IDLE, _W])
    assert rewards == pytest.approx([0.0, -0.01, 0.0, -0.01, 0.0, 0.0])
