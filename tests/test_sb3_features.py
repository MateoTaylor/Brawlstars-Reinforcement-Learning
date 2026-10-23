import numpy as np
import pytest
import torch
import yaml
from gymnasium import spaces

from brawl_sim.config import _deep_merge, load_config
from brawl_sim.core import obs_select
from brawl_sim.wrappers.sb3_features import BrawlFeaturesExtractor, default_policy_kwargs, float_group_names

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
AGENT_OBS_YAML = "configs/agent_obs.yaml"


def _space(overrides=None):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = _deep_merge(merged, overrides)  # the preset's other keys in a section stay
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


def test_default_policy_kwargs_has_normalize_images_false():
    _space_ignored, spec, cfg = _space()
    kwargs = default_policy_kwargs(spec, cfg)
    assert kwargs["normalize_images"] is False
    assert kwargs["features_extractor_class"] is BrawlFeaturesExtractor


def test_normalize_images_false_is_load_bearing():
    """Regression/documentation test: without normalize_images=False, SB3's own preprocess_obs
    would divide the grid's raw occupancy counts (0-255, not pixel intensities) by 255 before
    this extractor ever sees them."""
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


# ---- the grid's channel axis, pinned --------------------------------------------------------

# A checkpoint's first conv is sized from the grid's channel count, so a change to it is a
# from-scratch retrain, never a checkpoint swap. deploy6's own input widths are pinned in
# test_configs_files.py; the five deploy specs before it are in archive/configs/.
DEPLOY6 = "configs/agent_obs_deploy6.yaml"
HISTORY_PLANES = ("enemy_hist1", "enemy_hist2", "enemy_hist3")


def test_the_extractor_sizes_its_first_conv_from_the_grid_channel_axis(tmp_path):
    """deploy6's grid is 13 x 13 x 21, so its channel count equals the view height, and an
    extractor that read the wrong axis would still build the 13 input channels
    test_configs_files.py pins for it. The same spec without its three history planes is
    10 x 13 x 21, and its 10 is what tells the two axes apart, as deploy3's ten planes did before
    it went to archive/configs/. Dropping grid planes leaves every float group alone, so the MLP
    keeps deploy6's 272.
    """
    doc = yaml.safe_load(open(DEPLOY6, encoding="utf-8").read())
    grid = next(g for g in doc["groups"] if g["name"] == "grid")
    grid["view_channels"] = [c for c in grid["view_channels"] if c not in HISTORY_PLANES]
    variant = tmp_path / "agent_obs_variant.yaml"
    variant.write_text(yaml.safe_dump(doc), encoding="utf-8")
    cfg = load_config(CONFIGS_DEFAULT)
    spec = obs_select.load_agent_spec(variant, cfg)
    space = obs_select.agent_space(spec, cfg)
    kwargs = default_policy_kwargs(spec, cfg)
    fe = BrawlFeaturesExtractor(space, **kwargs["features_extractor_kwargs"])

    assert space.spaces["grid"].shape == (10, 13, 21)
    assert fe.cnn[0].in_channels == 10
    assert fe.mlp[0].in_features == 272
    out = fe(_sample_batch(space, batch_size=2))
    assert out.shape == (2, fe.features_dim)
    assert torch.isfinite(out).all()


# ---- critic_keys (archive/CRITIC_AND_MEMORY_PLAN.md part B) --------------------------------------

def test_critic_keys_bypass_both_actor_layers_and_are_appended_as_given():
    """The output is the actor features, then each critic group flattened, untouched: the MLP is
    sized without them and the tail is the input itself. An unknown key is refused."""
    cfg = load_config(CONFIGS_DEFAULT, overrides=CONFIGS_TINY)
    spec = obs_select.load_agent_spec(DEPLOY6, cfg)
    space = obs_select.agent_space(spec, cfg)
    keys = ("critic_enemies", "critic_world")
    fe = BrawlFeaturesExtractor(space, critic_keys=keys)
    actor_floats = sum(int(np.prod(s.shape)) for k, s in space.spaces.items()
                       if s.dtype == np.float32 and k not in keys)
    assert fe.mlp[0].in_features == actor_floats
    batch = _sample_batch(space, batch_size=3)
    out = fe(batch)
    assert torch.equal(out[:, fe.actor_dim:], torch.cat([batch[k].flatten(1) for k in keys], 1))
    with pytest.raises(ValueError, match="critic_keys"):
        BrawlFeaturesExtractor(space, critic_keys=("critic_nope",))
