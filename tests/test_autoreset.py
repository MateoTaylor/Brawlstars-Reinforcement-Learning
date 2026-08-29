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
    own parallel counter isn't drifting from the simulator's ground truth.

    The two are in DIFFERENT UNITS once action_repeat > 1: final_episode_length counts DECISIONS
    (one per step() call, SB3's convention for info["episode"]["l"]) and step_count counts SIM
    TICKS, so the conversion factor is cfg.action_repeat. It divides exactly -- every sub-tick
    runs whether or not the env finished early, so step_count is always a whole multiple."""
    n_envs = 16
    env = EpisodeStats(_tiny_env(n_envs=n_envs))
    env.reset()
    repeat = env.env.cfg.action_repeat
    gen = torch.Generator().manual_seed(0)

    for _ in range(2000 // repeat):   # same span of SIMULATED time regardless of action_repeat
        obs, reward, terminated, truncated, info = env.step(_random_action(n_envs, gen))
        done = terminated | truncated
        if bool(done.any()):
            got = info["final_episode_length"][done]
            ticks = info["final_observation"]["meta"]["step_count"][done].to(torch.int64)
            assert torch.equal(ticks % repeat, torch.zeros_like(ticks))
            assert torch.equal(got, ticks // repeat)


def test_episode_stats_fires_exactly_once_per_completed_episode_no_staleness():
    """For every env, the gap (in DECISIONS) between consecutive done events must equal exactly
    what final_episode_length reported at the later of the two -- i.e. the counter is reset
    on every done tick and never double-counts or carries over stale length from a prior
    episode. Both sides count step() calls, so this holds at any action_repeat."""
    n_envs = 64
    env = EpisodeStats(_tiny_env(n_envs=n_envs))
    env.reset()
    gen = torch.Generator().manual_seed(1)

    last_done_step = torch.zeros(n_envs, dtype=torch.int64)  # 0 == "since the initial reset"
    episodes_completed = torch.zeros(n_envs, dtype=torch.int64)

    for step_idx in range(1, 10_001 // env.env.cfg.action_repeat):
        obs, reward, terminated, truncated, info = env.step(_random_action(n_envs, gen))
        done = terminated | truncated
        if bool(done.any()):
            gap = torch.full((n_envs,), step_idx, dtype=torch.int64) - last_done_step
            assert torch.equal(info["final_episode_length"][done], gap[done])
            last_done_step = torch.where(done, torch.full_like(last_done_step, step_idx), last_done_step)
            episodes_completed += done.to(torch.int64)

    assert int(episodes_completed.sum()) > 0  # sanity: episodes actually completed in 10k steps


def test_episode_stats_return_accumulates_and_resets():
    """With a reward_fn that's zero everywhere except a +-1 terminal tick (ExampleReward,
    Step 28), the accumulated final_episode_return on a done tick must equal exactly that
    tick's own reward -- every earlier tick in the episode contributed 0."""
    from brawl_sim.core.reward import ExampleReward
    n_envs = 4
    base = _tiny_env(n_envs=n_envs)
    base.reward_fn = ExampleReward()
    env = EpisodeStats(base)
    env.reset()
    base.state.ent_hp[:, 0] = 0.0
    base.state.ent_last_hit_by[:, 0] = -1

    obs, reward, terminated, truncated, info = env.step(_idle_action(n_envs))
    assert torch.all(terminated)
    assert torch.equal(info["final_episode_return"], reward)
    assert torch.equal(info["final_episode_length"], torch.ones(n_envs, dtype=torch.int64))


def test_episode_stats_reset_zeroes_counters_for_masked_envs_only():
    n_envs = 4
    env = EpisodeStats(_tiny_env(n_envs=n_envs))
    env.reset()
    gen = torch.Generator().manual_seed(2)
    for _ in range(5):
        env.step(_random_action(n_envs, gen))
    before = env._length.clone()
    assert torch.all(before > 0)

    mask = torch.tensor([True, False, True, False])
    env.reset(mask)
    assert torch.equal(env._length[mask], torch.zeros(2, dtype=torch.int64))
    assert torch.equal(env._length[~mask], before[~mask])
