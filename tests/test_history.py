"""Observation history rings (SIM_OVERHAUL_PLAN.md Phase H, Step H1): core/history.push and its
wiring into env.step. What the H2 observation fields will read; nothing in the tick does.
"""
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.core import history
from brawl_sim.env import BrawlVecEnv

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


def _env(n_envs=2, seed=0, overrides=None):
    merged = {name: dict(section) if isinstance(section, dict) else section
              for name, section in CONFIGS_TINY.items()}
    for name, section in (overrides or {}).items():
        merged[name] = {**merged.get(name, {}), **section} if isinstance(section, dict) else section
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged)
    return BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=seed, verbose=False)


def _idle(n_envs):
    return torch.zeros(n_envs, 2, dtype=torch.int64)


def _valid(env) -> list:
    return env.state.hist_valid.to(torch.int64).tolist()


def test_rings_start_empty_and_fill_one_slot_per_decision():
    env = _env(n_envs=2)
    env.reset()
    assert _valid(env) == [[0, 0, 0], [0, 0, 0]]
    for expected in ([1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]):
        env.step(_idle(2))
        assert _valid(env) == [expected, expected]


def test_slot_zero_holds_the_last_action_and_the_pre_step_hero_state():
    """Slot 0 is the state the policy's observation was built from, paired with the action it
    chose against it -- so it is written BEFORE the world moves, and the previous slot 0 slides to
    slot 1 untouched."""
    env = _env(n_envs=2)
    env.reset()
    first = torch.tensor([[3, 0], [5, 0]], dtype=torch.int64)
    pos_before = env.state.ent_pos[:, 0].clone()
    hp_before = env.state.ent_hp[:, 0].clone()
    ammo_before = env.state.ent_ammo[:, 0].clone()
    all_pos_before = env.state.ent_pos.clone()
    env.step(first)
    assert torch.equal(env.state.hist_action[:, 0], first)
    assert torch.equal(env.state.hist_pos[:, 0], pos_before)
    assert torch.equal(env.state.hist_hp[:, 0], hp_before)
    assert torch.equal(env.state.hist_ammo[:, 0], ammo_before)
    assert torch.equal(env.state.hist_enemy_pos[:, 0], all_pos_before)
    assert not torch.equal(env.state.ent_pos[:, 0], pos_before), "the hero moved; the ring kept the old position"

    second = torch.tensor([[7, 0], [9, 0]], dtype=torch.int64)
    pos_mid = env.state.ent_pos[:, 0].clone()
    env.step(second)
    assert torch.equal(env.state.hist_action[:, 0], second)
    assert torch.equal(env.state.hist_action[:, 1], first)
    assert torch.equal(env.state.hist_pos[:, 0], pos_mid)
    assert torch.equal(env.state.hist_pos[:, 1], pos_before)
    assert _valid(env) == [[1, 1, 0], [1, 1, 0]]


def test_enemy_seen_is_alive_and_visible_and_never_the_hero_itself():
    env = _env(n_envs=1)
    env.reset()
    env.step(_idle(1))
    seen = env.state.hist_enemy_seen[:, 0]
    assert not seen[:, 0].any(), "the hero's own column is never 'seen'"
    assert seen[:, 1:].all(), "blank map, no bushes, everyone alive: every enemy is visible"

    env.state.ent_alive[0, 1] = False
    env.state.ent_hp[0, 1] = 0.0
    env.step(_idle(1))
    assert not env.state.hist_enemy_seen[0, 0, 1], "a dead entity is not seen"
    assert env.state.hist_enemy_seen[0, 1, 1], "...but the slot before it still says it was"


def test_history_is_per_decision_regardless_of_action_repeat():
    for repeat in (1, 5):
        env = _env(n_envs=1, overrides={"sim": {"action_repeat": repeat}})
        env.reset()
        env.step(_idle(1))
        env.step(_idle(1))
        assert _valid(env) == [[1, 1, 0]], f"action_repeat={repeat}"


def test_an_autoreset_row_comes_back_with_no_history_and_the_others_keep_theirs():
    env = _env(n_envs=2)
    env.reset()
    for _ in range(3):
        env.step(_idle(2))
    assert _valid(env) == [[1, 1, 1], [1, 1, 1]]

    # Kill env 0's hero by hand: that step terminates env 0, autoreset zeroes its row (zero_ covers
    # the rings like every other resettable field), and env 1 is untouched.
    env.state.ent_alive[0, 0] = False
    env.state.ent_hp[0, 0] = 0.0
    _, _, terminated, _, _ = env.step(_idle(2))
    assert bool(terminated[0]) and not bool(terminated[1])
    assert _valid(env) == [[0, 0, 0], [1, 1, 1]]
    assert not env.state.hist_action[0].any() and not env.state.hist_enemy_seen[0].any()


def test_push_shifts_oldest_first_so_nothing_is_read_after_being_overwritten():
    """Direct unit check of the shift on a hand-built ring: three pushes with distinct actions
    leave the newest at slot 0 and the oldest at slot 2, for a ring deeper than the plan's 3."""
    env = _env(n_envs=1, overrides={"observation": {"history_frames": 4}})
    env.reset()
    vis = torch.ones(1, env.cfg.n_entities, env.cfg.n_entities, dtype=torch.bool)
    for value in (1, 2, 3):
        history.push(env.state, torch.tensor([[value, value]]), vis)
    assert env.state.hist_action[0, :, 0].tolist() == [3, 2, 1, 0]
    assert env.state.hist_valid[0].tolist() == [True, True, True, False]
