"""The privileged critic (archive/CRITIC_AND_MEMORY_PLAN.md part B): a spec's `critic:` groups reach
the value head and never the policy.

Built the way `scripts/train.py` builds a run, from configs/train.yaml with the tiny env on the
CPU, so the policy class, the extractor's critic keys and every width are the run's own. The
guarantee under test is the one live play rests on: the deployment zero-fills the critic groups
(`build_agent_obs(critic=False)`), which is harmless only if no actor output moves when they do.
"""
import numpy as np
import pytest
import torch
import yaml

from brawl_sim.training.builder import build_run, no_image_transpose
from brawl_sim.training.config import load_train_config
from brawl_sim.wrappers.sb3_policy import PrivilegedCriticPolicy, SplitMlpExtractor

TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
CRITIC_KEYS = ("critic_enemies", "critic_world")


def _overrides(**run) -> dict:
    return {"run": {"device": "cpu", "n_envs": 4, "tensorboard": False, "env_overrides": TINY,
                    **run},
            "ppo": {"n_steps": 8, "batch_size": 16},
            "eval": {"holdout_maps": None}}


def _tcfg(**run):
    """train.yaml as `scripts/train.py --set` would load it: its own env overrides (the 12-miss
    coast, auto-aim) with the tiny world merged in, deploy6, the narrow widths."""
    return load_train_config("configs/train.yaml", overrides=_overrides(**run))


@pytest.fixture(scope="module")
def built():
    model, venv, parts = build_run(_tcfg())
    yield model, venv, parts
    venv.close()


def _logits(policy, obs_t) -> torch.Tensor:
    return torch.cat([d.logits for d in policy.get_distribution(obs_t).distributions], dim=1)


def _with_critic(obs_t: dict, fill) -> dict:
    return {k: (fill(v) if k in CRITIC_KEYS else v) for k, v in obs_t.items()}


def test_the_critic_inputs_move_the_value_and_never_the_policy(built):
    """Every actor path -- the distribution, `forward`, `evaluate_actions` -- is bit-identical
    with the critic groups replaced by noise, and the value is not."""
    model, venv, _ = built
    pol = model.policy
    fe = pol.features_extractor
    assert type(pol) is PrivilegedCriticPolicy and isinstance(pol.mlp_extractor, SplitMlpExtractor)
    assert fe.critic_keys == CRITIC_KEYS
    critic_floats = sum(int(np.prod(venv.observation_space[k].shape)) for k in CRITIC_KEYS)
    assert pol.mlp_extractor.policy_net[0].in_features == fe.actor_dim
    assert pol.mlp_extractor.value_net[0].in_features == fe.features_dim == fe.actor_dim + critic_floats

    obs_t, _ = pol.obs_to_tensor(venv.reset())
    assert all(obs_t[k].any() for k in CRITIC_KEYS), "the sim fills them, so the noise is a change"
    gen = torch.Generator().manual_seed(0)
    noisy = _with_critic(obs_t, lambda v: v + 3.0 * torch.randn(v.shape, generator=gen))
    with torch.no_grad():
        assert torch.equal(_logits(pol, obs_t), _logits(pol, noisy))
        (a1, v1, lp1), (a2, v2, lp2) = pol(obs_t, deterministic=True), pol(noisy, deterministic=True)
        assert torch.equal(a1, a2) and torch.equal(lp1, lp2) and not torch.equal(v1, v2)
        (ev1, elp1, ent1), (ev2, elp2, ent2) = (pol.evaluate_actions(obs_t, a1),
                                                pol.evaluate_actions(noisy, a1))
        assert torch.equal(elp1, elp2) and torch.equal(ent1, ent2) and not torch.equal(ev1, ev2)


def test_the_deployed_policy_decides_the_same_on_the_zeros_live_play_feeds_it(built):
    from brawl_deployment.policy import DeployedPolicy

    model, venv, parts = built
    deployed = DeployedPolicy(model, parts["agent_spec"], parts["env_cfg"])
    legal = (True,) * int(parts["env_cfg"].action_nvec[1])
    obs = venv.reset()
    for i in range(len(next(iter(obs.values())))):
        one = {k: v[i] for k, v in obs.items()}
        zeroed = {k: (np.zeros_like(v) if k in CRITIC_KEYS else v) for k, v in one.items()}
        assert deployed.act(one, legal) == deployed.act(zeroed, legal)


def test_a_saved_checkpoint_loads_back_as_the_same_split_policy(built, tmp_path):
    """`MaskablePPO.save` pickles the policy class by its import path, so this is also the test
    that wrappers/sb3_policy.py is still where checkpoints expect it."""
    from sb3_contrib import MaskablePPO

    model, venv, _ = built
    model.save(tmp_path / "m.zip")
    loaded = MaskablePPO.load(tmp_path / "m.zip", device="cpu")
    assert type(loaded.policy) is PrivilegedCriticPolicy
    assert isinstance(loaded.policy.mlp_extractor, SplitMlpExtractor)
    assert loaded.policy.features_extractor.critic_keys == CRITIC_KEYS
    obs_t, _ = model.policy.obs_to_tensor(venv.reset())
    with torch.no_grad():
        assert torch.equal(_logits(model.policy, obs_t), _logits(loaded.policy, obs_t))
        assert torch.equal(model.policy.predict_values(obs_t), loaded.policy.predict_values(obs_t))


def test_a_rollout_and_an_update_train_both_heads():
    model, venv, _ = build_run(_tcfg())
    nets = model.policy.mlp_extractor
    before = {name: [p.detach().clone() for p in net.parameters()]
              for name, net in (("pi", nets.policy_net), ("vf", nets.value_net))}
    model.learn(total_timesteps=4 * 8)  # one rollout of n_steps x n_envs, then one update
    venv.close()
    assert model.num_timesteps >= 32
    for name, net in (("pi", nets.policy_net), ("vf", nets.value_net)):
        assert any(not torch.equal(a, b) for a, b in zip(before[name], net.parameters())), name


def test_critic_groups_need_the_maskable_policy(built):
    """build_model refuses before it touches a venv, so the module's built run serves."""
    from brawl_sim.training.builder import build_model
    _, _, parts = built
    with pytest.raises(ValueError, match="maskable_ppo"):
        build_model(_tcfg(algo="ppo"), None, parts["agent_spec"], parts["env_cfg"])


def _run_dir(path, model):
    """A run directory `DeployedPolicy.from_run` accepts: a train.yaml naming deploy6 under the
    fixture's own env overrides, and `model` as best_model.zip."""
    tcfg = _tcfg()
    path.mkdir(parents=True, exist_ok=True)
    (path / "train.yaml").write_text(yaml.safe_dump({
        "run": {"algo": "maskable_ppo", "env_config": tcfg.run.env_config,
                "agent_obs": tcfg.run.agent_obs, "env_overrides": tcfg.run.env_overrides},
        "normalize": {"obs": False, "reward": True},
        "eval": {"enabled": False},
    }))
    model.save(path / "best_model.zip")
    return path


def test_from_run_loads_a_split_policy_and_refuses_one_that_would_act_on_zeros(built, tmp_path):
    """A deploy6 checkpoint whose actor could read the critic groups would act on the zeros live
    play feeds it. Two ways to build one: the extractor sets them aside but the stock
    `MlpExtractor` hands the policy net every column, or the extractor never set them aside.
    Both are refused at load, before a decision is made."""
    from sb3_contrib import MaskablePPO

    from brawl_deployment.policy import DeployedPolicy
    from brawl_sim.wrappers.sb3_features import default_policy_kwargs

    model, venv, parts = built
    good = DeployedPolicy.from_run(_run_dir(tmp_path / "good", model))
    assert type(good.model.policy) is PrivilegedCriticPolicy

    kwargs = default_policy_kwargs(parts["agent_spec"], parts["env_cfg"])
    for name, policy_kwargs in (("stock_mlp", kwargs),
                                ("no_critic_keys", {**kwargs, "features_extractor_kwargs": {}})):
        bad = MaskablePPO("MultiInputPolicy", no_image_transpose(venv), policy_kwargs=policy_kwargs,
                          n_steps=8, batch_size=16, device="cpu")
        with pytest.raises(ValueError, match="actor head"):
            DeployedPolicy.from_run(_run_dir(tmp_path / name, bad))
