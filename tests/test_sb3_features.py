import numpy as np
import torch
import yaml
from gymnasium import spaces
from stable_baselines3 import PPO

from brawl_sim.config import load_config
from brawl_sim.core import obs_select
from brawl_sim.wrappers.sb3_features import BrawlFeaturesExtractor, default_policy_kwargs, float_group_names

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
AGENT_OBS_YAML = "configs/agent_obs.yaml"


def _space(overrides=None):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = {**merged, **overrides}
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    return obs_select.agent_space(spec, cfg), spec, cfg


def _sample_batch(space: spaces.Dict, batch_size: int) -> dict:
    return {
        name: torch.stack([torch.as_tensor(box.sample()).float() for _ in range(batch_size)])
        for name, box in space.spaces.items()
    }


# ---- forward pass produces the declared features_dim, no shape errors ---------------------

def test_forward_pass_produces_declared_features_dim():
    space, spec, cfg = _space()
    fe = BrawlFeaturesExtractor(space)
    batch = _sample_batch(space, batch_size=5)
    out = fe(batch)
    assert out.shape == (5, fe.features_dim)
    assert torch.isfinite(out).all()


def test_forward_pass_works_at_the_real_default_config_grid_size():
    """The real (non-tiny) view size, the one NatureCNN's stack cannot handle.

    The grid shape is read from the config rather than written as a literal. It used to say
    (10, 20, 40), which was only ever true while the view was a guess -- measuring it against the
    real camera (Terrain_Perception_Build_Plan.md Phase K) made it 13x21 and this test failed on a
    hardcoded number rather than on anything about the feature extractor.

    What DOES need asserting is the thing the smaller view puts at risk: the conv stack is
    stride 1/2/2, so a 13x21 grid comes out 4x6, and a further shrink would eventually collapse a
    spatial dimension to zero and take the forward pass with it.
    """
    space, spec, cfg = _space(overrides=None)
    cfg_full = load_config(CONFIGS_DEFAULT)
    spec_full = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg_full)
    space_full = obs_select.agent_space(spec_full, cfg_full)
    c, h, w = space_full.spaces["grid"].shape
    assert (h, w) == (cfg_full.view_h, cfg_full.view_w)
    assert h >= 8 and w >= 8, (
        f"view {w}x{h} leaves the stride-1/2/2 conv stack {w // 4}x{h // 4} to work with"
    )
    fe = BrawlFeaturesExtractor(space_full)
    batch = _sample_batch(space_full, batch_size=3)
    out = fe(batch)
    assert out.shape == (3, fe.features_dim)
    assert torch.isfinite(out).all()


# ---- generic to grid-only / vector-only observation spaces (not hardcoded to 5 groups) -----

def test_handles_grid_only_and_vector_only_spaces():
    grid_only = spaces.Dict({"grid": spaces.Box(low=0, high=255, shape=(10, 10, 14), dtype=np.uint8)})
    fe = BrawlFeaturesExtractor(grid_only)
    out = fe({"grid": torch.zeros(2, 10, 10, 14)})
    assert out.shape == (2, fe.features_dim)

    vector_only = spaces.Dict({"self": spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)})
    fe2 = BrawlFeaturesExtractor(vector_only)
    out2 = fe2({"self": torch.zeros(2, 21)})
    assert out2.shape == (2, fe2.features_dim)


def test_rejects_two_uint8_groups():
    bad = spaces.Dict({
        "grid1": spaces.Box(low=0, high=255, shape=(4, 10, 10), dtype=np.uint8),
        "grid2": spaces.Box(low=0, high=255, shape=(4, 10, 10), dtype=np.uint8),
    })
    try:
        BrawlFeaturesExtractor(bad)
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "grid1" in str(e) and "grid2" in str(e)


# ---- default_policy_kwargs plugs into PPO("MultiInputPolicy", ...) unmodified --------------

def test_default_policy_kwargs_plugs_into_ppo_unmodified():
    from brawl_sim.env import BrawlVecEnv
    from brawl_sim.core.reward import ExampleReward
    from brawl_sim.wrappers.sb3_vecenv import BrawlSB3VecEnv

    cfg = load_config(CONFIGS_DEFAULT, overrides=CONFIGS_TINY)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    env = BrawlVecEnv(cfg, n_envs=4, device="cpu", seed=0)
    venv = BrawlSB3VecEnv(env, spec, ExampleReward())

    model = PPO(
        "MultiInputPolicy", venv, policy_kwargs=default_policy_kwargs(spec, cfg),
        n_steps=16, batch_size=16, verbose=0,
    )
    model.learn(total_timesteps=64)  # must not raise


def test_default_policy_kwargs_has_normalize_images_false():
    _space_ignored, spec, cfg = _space()
    kwargs = default_policy_kwargs(spec, cfg)
    assert kwargs["normalize_images"] is False
    assert kwargs["features_extractor_class"] is BrawlFeaturesExtractor


def test_normalize_images_false_is_load_bearing():
    """Regression/documentation test: without normalize_images=False, SB3's own preprocess_obs
    would divide the grid's raw occupancy counts (0-255, not pixel intensities) by 255 before
    this extractor ever sees them -- exactly the corruption the plan's text warns about."""
    from stable_baselines3.common.preprocessing import preprocess_obs
    space, spec, cfg = _space()
    grid_space = space.spaces["grid"]
    obs = torch.full((1, *grid_space.shape), 200, dtype=torch.uint8)
    assert preprocess_obs(obs, grid_space, normalize_images=False).flatten()[0].item() == 200.0
    assert preprocess_obs(obs, grid_space, normalize_images=True).flatten()[0].item() != 200.0


# ---- float_group_names -----------------------------------------------------------------------

def test_float_group_names_excludes_the_uint8_grid_group():
    _space_ignored, spec, cfg = _space()
    names = float_group_names(spec)
    assert "grid" not in names
    assert set(names) == {g.name for g in spec.groups if g.name != "grid"}
