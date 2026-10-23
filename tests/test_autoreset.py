import yaml
import torch

from brawl_sim.config import load_config
from brawl_sim.env import BrawlVecEnv
from brawl_sim.wrappers.episode_stats import EpisodeStats

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


def _tiny_env(n_envs=8, seed=0, overrides=None):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = {**merged, **overrides}
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged)
    return BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=seed)


def _idle_action(n_envs):
    return torch.zeros(n_envs, 2, dtype=torch.int64)


def _random_action(n_envs, gen=None):
    move = torch.randint(0, 17, (n_envs,), generator=gen)
    fire = torch.randint(0, 2, (n_envs,), generator=gen)
    return torch.stack([move, fire], dim=1)


# ---- per-env autoreset isolation (only the finished env resets) --------------------------

def test_only_the_finished_env_resets_others_are_untouched():
    n_envs = 9
    env = _tiny_env(n_envs=n_envs)
    env.reset()
    # kill only env 7's hero; env 8 (and everyone else) plays on normally.
    env.state.ent_hp[7, 0] = 0.0
    env.state.ent_last_hit_by[7, 0] = -1  # avoid the zero_-default-0 last_hit_by test artifact

    obs, reward, terminated, truncated, info = env.step(_idle_action(n_envs))

    assert bool(terminated[7])
    assert not bool(terminated[8])
    # step_count is in SIM TICKS and one step() is cfg.action_repeat of them (env.py
    # `_run_decision`), so a env that just ran the decision sits at action_repeat, not 1.
    assert int(env.state.step_count[7]) == 0  # env 7 autoreset in place
    assert int(env.state.step_count[8]) == env.cfg.action_repeat  # env 8 just ticked forward
    assert not bool(info["final_observation"]["hero"]["alive"][7])
    assert float(info["final_observation"]["hero"]["hp"][7]) <= 0.0


# ---- EpisodeStats: final_episode_length fires exactly once per completed episode ---------

def test_episode_stats_final_episode_length_matches_ground_truth_step_count():
    """final_episode_length must agree, on every done tick, with the finished episode's own
    step_count as captured in final_observation -- an independent cross-check that EpisodeStats'
    own parallel counter isn't drifting from the simulator's ground truth -- and with the gap,
    in decisions, since that env's previous done: the counter is reset on every done and never
    double-counts or carries a stale length into the next episode.

    The two are in DIFFERENT UNITS once action_repeat > 1: final_episode_length counts DECISIONS
    (one per step() call, SB3's convention for info["episode"]["l"]) and step_count counts SIM
    TICKS, so the conversion factor is cfg.action_repeat. It divides exactly -- every sub-tick
    runs whether or not the env finished early, so step_count is always a whole multiple.

    Two episode lengths of decisions: no episode outlasts `max_agent_steps`, so every env ends
    at least two, the second testing the reset after the first, and random actions on
    debug_tiny kill most heroes before the truncation, so both ways an episode ends reach the
    check. Until 2026-10-20 the gap was a test of its own that re-ran this loop over 64 envs x
    180 decisions."""
    n_envs = 16
    env = EpisodeStats(_tiny_env(n_envs=n_envs))
    env.reset()
    repeat = env.env.cfg.action_repeat
    gen = torch.Generator().manual_seed(0)

    last_done = torch.zeros(n_envs, dtype=torch.int64)   # 0 == "since the initial reset"
    episodes = torch.zeros(n_envs, dtype=torch.int64)
    for step_idx in range(1, 2 * env.env.cfg.max_agent_steps + 1):
        obs, reward, terminated, truncated, info = env.step(_random_action(n_envs, gen))
        done = terminated | truncated
        if bool(done.any()):
            got = info["final_episode_length"][done]
            ticks = info["final_observation"]["meta"]["step_count"][done].to(torch.int64)
            assert torch.equal(ticks % repeat, torch.zeros_like(ticks))
            assert torch.equal(got, ticks // repeat)
            assert torch.equal(got, (step_idx - last_done)[done])
            last_done = torch.where(done, torch.full_like(last_done, step_idx), last_done)
            episodes += done.to(torch.int64)
    assert int(episodes.min()) >= 2, episodes.tolist()


def _one_per_decision(obs, info, cfg):
    """A reward of 1 every decision, so a return equals its episode's length."""
    return torch.ones(info["time"].shape[0])


def test_episode_stats_return_accumulates_and_resets():
    """Under a reward of 1 per decision, a 4-decision episode returns 4.0 and the 1-decision one
    after it 1.0: the return is the running total (not the done decision's reward alone) and is
    zeroed on the done. Until 2026-10-20 the only episode here was one decision long, which
    neither property can be told from."""
    n_envs = 4
    base = _tiny_env(n_envs=n_envs)
    base.reward_fn = _one_per_decision
    env = EpisodeStats(base)
    env.reset()
    for _ in range(3):
        *_, terminated, truncated, _info = env.step(_idle_action(n_envs))
        assert not bool((terminated | truncated).any())

    for decisions in (4, 1):
        base.state.ent_hp[:, 0] = 0.0
        base.state.ent_last_hit_by[:, 0] = -1
        *_, terminated, _truncated, info = env.step(_idle_action(n_envs))
        assert torch.all(terminated)
        assert torch.equal(info["final_episode_return"], torch.full((n_envs,), float(decisions)))
        assert torch.equal(info["final_episode_length"], torch.full((n_envs,), decisions))


def test_episode_stats_reset_zeroes_counters_for_masked_envs_only():
    n_envs = 4
    base = _tiny_env(n_envs=n_envs)
    base.reward_fn = _one_per_decision
    env = EpisodeStats(base)
    env.reset()
    gen = torch.Generator().manual_seed(2)
    for _ in range(5):
        env.step(_random_action(n_envs, gen))
    before, before_return = env._length.clone(), env._return.clone()
    assert torch.all(before > 0) and torch.all(before_return > 0)

    mask = torch.tensor([True, False, True, False])
    env.reset(mask)
    assert torch.equal(env._length[mask], torch.zeros(2, dtype=torch.int64))
    assert torch.equal(env._length[~mask], before[~mask])
    assert torch.equal(env._return[mask], torch.zeros(2))
    assert torch.equal(env._return[~mask], before_return[~mask])
