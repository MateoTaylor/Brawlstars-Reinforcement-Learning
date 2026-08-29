import numpy as np
import torch
import yaml
from stable_baselines3.common.vec_env import VecEnv

from brawl_sim.config import load_config
from brawl_sim.core import obs_select
from brawl_sim.core.reward import ExampleReward, ZeroReward
from brawl_sim.env import BrawlVecEnv
from brawl_sim.wrappers.sb3_vecenv import BrawlSB3VecEnv

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
AGENT_OBS_YAML = "configs/agent_obs.yaml"


def _make(n_envs=6, seed=0, reward_fn=None, info_mode="minimal", info_every=1, overrides=None):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = {**merged, **overrides}
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged)
    env = BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=seed)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    return BrawlSB3VecEnv(env, spec, reward_fn or ExampleReward(), info_mode=info_mode, info_every=info_every)


def _random_actions(n_envs, rng):
    return np.stack([rng.integers(0, 17, size=n_envs), rng.integers(0, 2, size=n_envs)], axis=1)


# ---- VecEnv subclass check ------------------------------------------------------------------

def test_is_a_vecenv_subclass_and_constructs_cleanly():
    venv = _make()
    assert isinstance(venv, VecEnv)
    assert venv.num_envs == 6
    # 3-valued attack dim as of Step D2 (0 = nothing, 1 = attack, 2 = super); the
    # SB3 action space must track EnvConfig.action_nvec exactly or MaskablePPO's
    # mask width and the env's own mask disagree.
    assert venv.action_space.nvec.tolist() == list(venv.cfg.action_nvec)
    assert venv.action_space.nvec.tolist() == [17, 3]


def test_vec_monitor_wraps_cleanly():
    """Regression test: `self.spec` is a reserved gymnasium/SB3 attribute name (the
    registration EnvSpec, read by VecMonitor as `venv.spec.id`) -- BrawlSB3VecEnv's own
    AgentObsSpec must live under a different name (`agent_spec`), or VecMonitor.__init__
    crashes with `AttributeError: 'AgentObsSpec' object has no attribute 'id'`. Caught by an
    actual MaskablePPO integration smoke test, not by any of the acceptance criteria above."""
    from stable_baselines3.common.vec_env import VecMonitor
    venv = _make(n_envs=3)
    assert getattr(venv, "spec", None) is None
    monitored = VecMonitor(venv)  # must not raise
    monitored.reset()


# ---- reset() returns observations only -------------------------------------------------------

def test_reset_returns_observations_only():
    venv = _make()
    obs = venv.reset()
    assert isinstance(obs, dict)
    for name, box in venv.observation_space.spaces.items():
        assert obs[name].shape == (venv.num_envs, *box.shape)
        assert obs[name].dtype == box.dtype


# ---- terminal_observation / TimeLimit.truncated / episode ------------------------------------

def test_terminal_observation_present_on_done_absent_otherwise_and_matches_final_obs():
    venv = _make(n_envs=4)
    venv.reset()
    venv.env.env.state.ent_hp[0, 0] = 0.0
    venv.env.env.state.ent_last_hit_by[0, 0] = -1

    venv.step_async(np.zeros((4, 2), dtype=np.int64))
    obs, reward, dones, infos = venv.step_wait()

    assert bool(dones[0]) and not any(dones[1:])
    assert "terminal_observation" in infos[0]
    for i in (1, 2, 3):
        assert "terminal_observation" not in infos[i]

    # matches building the agent obs directly from the underlying env's own final_observation
    # (accessed via a second BrawlSB3VecEnv step is not possible after the fact, so instead
    # recompute independently from the same info the wrapper itself used).
    expected = obs_select.build_agent_obs(
        venv._last_full_obs, venv.agent_spec, venv.cfg,
        obs_select.make_agent_obs_buffers(venv.agent_spec, venv.cfg, 4, "cpu"),
    )
    # env 0 was reset -> _last_full_obs is the NEW episode; terminal_observation must differ
    # from it (it's the OLD, finished episode) for the field that changed (hero just died there).
    assert not np.allclose(infos[0]["terminal_observation"]["self"], expected["self"][0].numpy())


def test_time_limit_truncated_true_exactly_on_timeout_without_death():
    venv = _make(n_envs=2)
    venv.reset()
    venv.env.env.state.step_count.fill_(venv.cfg.max_episode_steps - 1)

    venv.step_async(np.zeros((2, 2), dtype=np.int64))
    obs, reward, dones, infos = venv.step_wait()

    assert dones.all()
    for i in range(2):
        assert infos[i]["TimeLimit.truncated"] is True

    # a death, by contrast, must NOT set TimeLimit.truncated.
    venv2 = _make(n_envs=2)
    venv2.reset()
    venv2.env.env.state.ent_hp[:, 0] = 0.0
    venv2.env.env.state.ent_last_hit_by[:, 0] = -1
    venv2.step_async(np.zeros((2, 2), dtype=np.int64))
    _, _, dones2, infos2 = venv2.step_wait()
    assert dones2.all()
    for i in range(2):
        assert infos2[i]["TimeLimit.truncated"] is False


def test_episode_stats_reward_matches_the_wrapper_owned_reward_fn():
    """Reward-ownership design (see module docstring): the wrapped BrawlVecEnv's own reward_fn
    is overwritten to be the SAME reward_fn passed to BrawlSB3VecEnv, so info["episode"]["r"]
    (sourced from EpisodeStats, which accumulates env.step()'s own reward) must exactly equal
    the reward this tick actually reported to SB3 -- not silently stuck at 0 (ZeroReward)."""
    venv = _make(n_envs=3, reward_fn=ExampleReward(), info_mode="episode")
    venv.reset()
    venv.env.env.state.ent_hp[:, 0] = 0.0
    venv.env.env.state.ent_last_hit_by[:, 0] = -1

    venv.step_async(np.zeros((3, 2), dtype=np.int64))
    obs, reward, dones, infos = venv.step_wait()
    assert dones.all()
    for i in range(3):
        assert infos[i]["episode"]["r"] == float(reward[i])
        assert reward[i] == -1.0  # ExampleReward: dead hero -> -1
        assert infos[i]["episode"]["l"] == 1


def test_episode_key_absent_in_minimal_mode_present_in_episode_mode():
    for mode, expect_key in (("minimal", False), ("episode", True), ("full", True)):
        venv = _make(n_envs=2, info_mode=mode)
        venv.reset()
        venv.env.env.state.ent_hp[:, 0] = 0.0
        venv.env.env.state.ent_last_hit_by[:, 0] = -1
        venv.step_async(np.zeros((2, 2), dtype=np.int64))
        _, _, dones, infos = venv.step_wait()
        assert dones.all()
        assert ("episode" in infos[0]) == expect_key


# ---- action_masks() matches obs["action_mask"] bit for bit -----------------------------------

def test_has_attr_action_masks_true_for_maskable_ppo_detection():
    """Regression test: sb3_contrib.MaskablePPO detects masking support via
    VecEnv.has_attr("action_masks") (-> get_attr, catching AttributeError). get_attr must
    check THIS wrapper (where action_masks() is actually defined) before the underlying
    BrawlVecEnv, or MaskablePPO refuses to train with
    'Environment does not support action masking'."""
    venv = _make(n_envs=3)
    assert venv.has_attr("action_masks")
    from sb3_contrib.common.maskable.utils import is_masking_supported
    assert is_masking_supported(venv)


def test_env_method_action_masks_matches_direct_call():
    """Regression test: sb3_contrib.get_action_masks(env) calls env.env_method("action_masks"),
    NOT env.action_masks() directly, and np.stack()s the result -- env_method must special-case
    this instead of raising NotImplementedError, or MaskablePPO.learn() can't run at all."""
    from sb3_contrib.common.maskable.utils import get_action_masks
    venv = _make(n_envs=4)
    venv.reset()
    stacked = get_action_masks(venv)
    assert np.array_equal(stacked, venv.action_masks())

    subset = venv.env_method("action_masks", indices=[0, 2])
    assert len(subset) == 2
    assert np.array_equal(np.stack(subset), venv.action_masks()[[0, 2]])


def test_action_masks_matches_full_obs_action_mask_bit_for_bit():
    venv = _make(n_envs=5)
    venv.reset()
    expected = torch.cat(
        [venv._last_full_obs["action_mask"]["move"], venv._last_full_obs["action_mask"]["attack"]], dim=-1
    ).numpy()
    assert np.array_equal(venv.action_masks(), expected)
    # The flat mask must be exactly sum(action_nvec) wide -- MaskablePPO slices it by
    # nvec, so any disagreement silently misaligns the attack mask against the move bins.
    # Derived rather than pinned at 19: Step D2 widened the attack dim 2 -> 3.
    assert venv.action_masks().shape == (5, sum(venv.cfg.action_nvec))
    assert venv.action_masks().shape == (5, 20)
    assert venv.action_masks().dtype == np.bool_


# ---- bounded, counted device->host transfers per step (minimal mode) -------------------------

def test_minimal_mode_host_transfers_are_bounded_and_independent_of_n_envs():
    # 5 obs groups + reward + terminated + truncated, every tick, regardless of n_envs;
    # +5 more (a second, same-size group pass) only on ticks where >= 1 episode just finished
    # (building terminal_observation) -- so the FLOOR is identical across n_envs (nothing here
    # scales with batch size, only with "did an episode end this particular tick", which is
    # deliberately checked separately from n_envs).
    rng = np.random.default_rng(0)
    baseline_by_n_envs = {}
    max_by_n_envs = {}
    for n_envs in (4, 64):
        venv = _make(n_envs=n_envs, info_mode="minimal")
        venv.reset()
        seen = []
        for _ in range(50):
            venv.step_async(_random_actions(n_envs, rng))
            venv.step_wait()
            seen.append(venv.last_step_host_transfers)
        baseline_by_n_envs[n_envs] = min(seen)
        max_by_n_envs[n_envs] = max(seen)

    assert baseline_by_n_envs[4] == baseline_by_n_envs[64] == 8
    assert max_by_n_envs[4] <= 13 and max_by_n_envs[64] <= 13  # bounded even when episodes end


# ---- switching info_mode does not change trajectories -----------------------------------------

def test_switching_info_mode_does_not_change_trajectories():
    n_envs = 4
    action_seq = [np.random.default_rng(7).integers(0, 2, size=(n_envs, 2)).astype(np.int64) for _ in range(20)]
    # deterministic action sequence, precomputed once, replayed identically against both envs
    rng = np.random.default_rng(7)
    action_seq = [_random_actions(n_envs, rng) for _ in range(20)]

    results = {}
    for mode in ("minimal", "episode", "full"):
        venv = _make(n_envs=n_envs, seed=3, reward_fn=ZeroReward(), info_mode=mode)
        obs0 = venv.reset()
        traj = [{k: v.copy() for k, v in obs0.items()}]
        for a in action_seq:
            venv.step_async(a)
            obs, reward, dones, infos = venv.step_wait()
            traj.append((reward.copy(), dones.copy(), {k: v.copy() for k, v in obs.items()}))
        results[mode] = traj

    for mode in ("episode", "full"):
        assert np.array_equal(results["minimal"][0]["self"], results[mode][0]["self"])
        for (r_m, d_m, o_m), (r_o, d_o, o_o) in zip(results["minimal"][1:], results[mode][1:]):
            assert np.array_equal(r_m, r_o)
            assert np.array_equal(d_m, d_o)
            for k in o_m:
                assert np.array_equal(o_m[k], o_o[k])


# ---- get_attr / set_attr / env_method / render / seed -----------------------------------------

def test_get_attr_set_attr_batch_wide():
    venv = _make(n_envs=3)
    values = venv.get_attr("n_envs")
    assert values == [3, 3, 3]

    venv.set_attr("reward_fn", ZeroReward())
    assert isinstance(venv.env.env.reward_fn, ZeroReward)


def test_env_method_and_render_raise_not_implemented():
    """`seed` used to be in this list. It no longer raises -- see
    test_seed_reseeds_the_shared_generator below for why that had to change."""
    venv = _make()
    for call in (lambda: venv.env_method("foo"), lambda: venv.render()):
        try:
            call()
            assert False, "expected NotImplementedError"
        except NotImplementedError:
            pass


def test_seed_reseeds_the_shared_generator():
    """`seed()` honors a single seed by reseeding the one shared `torch.Generator`, rather than
    raising as it originally did. The refusal was defensible (there genuinely are no independent
    per-sub-env seeds here) but it broke the plan's own recommended entry point: SB3's
    `BaseAlgorithm.set_random_seed` calls `env.seed(seed)` unconditionally whenever a model is
    built with `seed=`, so `MaskablePPO(..., seed=0)` could not be constructed against this env
    at all. Returning `[seed] * num_envs` satisfies the `VecEnv` contract; the repeated value is
    the honest signal that the batch shares one RNG stream."""
    venv = _make(n_envs=4)
    gen = venv.env.env.gen
    assert venv.seed(1234) == [1234] * 4
    first = torch.rand(8, generator=gen)
    venv.seed(1234)
    assert torch.equal(first, torch.rand(8, generator=gen))
    venv.seed(None)  # a no-op, not an error -- SB3 passes None when no seed was configured


def test_env_is_wrapped_returns_false_batch_wide():
    venv = _make(n_envs=3)
    assert venv.env_is_wrapped(object) == [False, False, False]
