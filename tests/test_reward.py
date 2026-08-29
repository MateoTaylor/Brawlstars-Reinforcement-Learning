import torch

from brawl_sim.core.reward import ExampleReward, RewardFn, ZeroReward


def _fake_obs_info(n_envs=4, device="cpu", terminated=None, hero_alive=None):
    if terminated is None:
        terminated = torch.zeros(n_envs, dtype=torch.bool, device=device)
    if hero_alive is None:
        hero_alive = torch.ones(n_envs, dtype=torch.bool, device=device)
    obs = {"hero": {"alive": hero_alive}}
    info = {
        "time": torch.zeros(n_envs, dtype=torch.float32, device=device),
        "terminated": terminated,
        "truncated": torch.zeros(n_envs, dtype=torch.bool, device=device),
    }
    return obs, info


# ---- ZeroReward -------------------------------------------------------------------------

def test_zero_reward_returns_zeros_correct_shape_dtype():
    fn = ZeroReward()
    obs, info = _fake_obs_info(n_envs=6)
    reward = fn(obs, info, cfg=None)
    assert reward.shape == (6,)
    assert reward.dtype == torch.float32
    assert torch.all(reward == 0)


def test_zero_reward_returns_same_tensor_object_across_calls():
    fn = ZeroReward()
    obs, info = _fake_obs_info(n_envs=4)
    r1 = fn(obs, info, cfg=None)
    obs2, info2 = _fake_obs_info(n_envs=4)  # a fresh obs/info dict each "step", same shape/device
    r2 = fn(obs2, info2, cfg=None)
    assert r1 is r2


def test_zero_reward_reallocates_on_n_envs_change():
    fn = ZeroReward()
    obs, info = _fake_obs_info(n_envs=4)
    r1 = fn(obs, info, cfg=None)
    obs2, info2 = _fake_obs_info(n_envs=8)
    r2 = fn(obs2, info2, cfg=None)
    assert r1 is not r2
    assert r2.shape == (8,)
    assert torch.all(r2 == 0)


# ---- ExampleReward ------------------------------------------------------------------------

def test_example_reward_terminal_win_lose_and_nonterminal():
    fn = ExampleReward()
    terminated = torch.tensor([True, True, False, True])
    hero_alive = torch.tensor([True, False, True, True])  # env 3: terminated & alive -> win
    obs, info = _fake_obs_info(n_envs=4, terminated=terminated, hero_alive=hero_alive)

    reward = fn(obs, info, cfg=None)
    assert reward.dtype == torch.float32
    assert reward.tolist() == [1.0, -1.0, 0.0, 1.0]


def test_example_reward_truncated_without_termination_is_zero():
    fn = ExampleReward()
    terminated = torch.zeros(3, dtype=torch.bool)
    hero_alive = torch.ones(3, dtype=torch.bool)
    obs, info = _fake_obs_info(n_envs=3, terminated=terminated, hero_alive=hero_alive)
    info["truncated"] = torch.tensor([True, False, True])  # timeout, hero alive but not alone

    reward = fn(obs, info, cfg=None)
    assert torch.all(reward == 0)


# ---- RewardFn Protocol: structural typing, no inheritance required -----------------------

def test_plain_function_satisfies_reward_fn_protocol():
    def my_reward(obs, info, cfg):
        return torch.ones(info["time"].shape[0], dtype=torch.float32)

    assert isinstance(my_reward, RewardFn)
    assert isinstance(ZeroReward(), RewardFn)
    assert isinstance(ExampleReward(), RewardFn)


def test_custom_reward_fn_needs_no_special_registration():
    class DistanceBasedReward:
        """A custom reward reading obs directly -- proof a RewardFn never needs to touch the
        simulator or subclass anything from core/reward.py."""

        def __call__(self, obs, info, cfg):
            return -obs["hero"]["dummy_dist"].to(torch.float32)

    obs, info = _fake_obs_info(n_envs=3)
    obs["hero"]["dummy_dist"] = torch.tensor([1.0, 2.0, 3.0])

    fn = DistanceBasedReward()
    reward = fn(obs, info, cfg=None)
    assert reward.tolist() == [-1.0, -2.0, -3.0]


# ---- batched / device-agnostic smoke -------------------------------------------------------

def test_batched_smoke_large_n():
    fn = ZeroReward()
    n_envs = 4096
    obs, info = _fake_obs_info(n_envs=n_envs)
    reward = fn(obs, info, cfg=None)
    assert reward.shape == (n_envs,)
    assert not torch.any(torch.isnan(reward))
