"""BrawlFeaturesExtractor: the minimum model-side code needed for SB3 to *accept* this
observation. See BRAWL_SIM_BUILD_PLAN.md Step 35.

**Not a tuned architecture -- a working default you will replace.**

**Why this exists (concrete gotcha).** SB3's default `MultiInputPolicy` uses `CombinedExtractor`,
which routes any 3D `uint8` subspace to `NatureCNN` -- whose fixed 8/4-then-4/2-then-3/1
kernel/stride stack (no padding) collapses `configs/agent_obs.yaml`'s `grid` group (`(10, 13,
21)` by default, `(10, 10, 14)` under `debug_tiny`) to an invalid size and raises. Confirmed by
hand in Step 33's own integration smoke test, not just anticipated from this step's text.
`BrawlFeaturesExtractor` replaces `CombinedExtractor` wholesale: a small padded 3x3-kernel CNN
for the one `uint8` grid group (padding is what keeps this stack alive on small grids where
`NatureCNN`'s stack dies), and a flatten+`Linear`+`ReLU` MLP for every other (float32) group
concatenated together, concatenated with the CNN's output.

**Nothing here is sized by hand.** `n_flatten` is measured by pushing one sample of the real
observation space through the stack, so the view size can change without touching this file --
which it did: `Terrain_Perception_Build_Plan.md` Phase K measured the real camera and the default
view went from 20x40 to 13x21, taking the CNN's output from 5x10 to 4x6 and `cnn_linear`'s input
from 3200 to 1536. That resizing is also the reason a checkpoint trained at the old view cannot be
loaded at the new one: the observation shape and the first linear layer both changed.

**This module hard-requires `stable_baselines3`** (no `try`/`except ImportError` guard) -- the
same choice `wrappers/sb3_vecenv.py` (Step 33) already made, kept consistent here rather than
introducing a different import-guarding convention for one file in the same package. Both are
under the `sb3` optional dependency group (`pyproject.toml`); nothing under `core/`/`bots/`/
`env.py` imports this module or requires it to be installed.

**`normalize_images=False` is required, and is NOT something `BrawlFeaturesExtractor` itself
can enforce** -- it's read by the POLICY (`BaseModel.extract_features`, via
`preprocess_obs(obs, observation_space, normalize_images=self.normalize_images)`), one level
above where any features extractor runs, before `BrawlFeaturesExtractor.forward` ever sees the
data. `default_policy_kwargs` returns it as a top-level key specifically so
`PPO(..., policy_kwargs=default_policy_kwargs(spec, cfg))` wires it into the POLICY constructor,
not into `features_extractor_kwargs`. Without it, SB3 would divide the grid's small occupancy
counts (0-255, but NOT pixel intensities) by 255 before this extractor ever runs.

**If you wrap the vec env in `VecNormalize`,** pass `norm_obs_keys=float_group_names(spec)` to
exclude the `uint8` grid group -- `VecNormalize` tracks a running mean/std per key and applying
that to occupancy counts (rather than continuous vector features) would corrupt them. This
module can't enforce that choice either (it happens entirely outside this file, at `VecNormalize`
construction time); `float_group_names` exists so you don't have to hand-transcribe your
`AgentObsSpec`'s group names/dtypes to get it right.
"""
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ..core import obs_select


class BrawlFeaturesExtractor(BaseFeaturesExtractor):
    """Small CNN sized for the one `uint8` grid group (`(C, view_h, view_w)`) + MLP for every
    other (float32) group, concatenated. Grid conv stack: 3x3 s1 -> 3x3 s2 -> 3x3 s2 -> flatten
    -> Linear -> ReLU (all convs padded, unlike `NatureCNN` -- see module docstring for why).
    Vector groups: flatten each, concatenate, -> `len(mlp_hidden_dims)` stacked Linear->ReLU
    layers (a single layer by default). Works with any `AgentObsSpec`-derived `observation_space`,
    not just the default `configs/agent_obs.yaml` layout -- zero, one uint8 group and any number
    of float32 groups (including zero of either, as long as not both) are all handled."""

    def __init__(
        self, observation_space: spaces.Dict,
        cnn_output_dim: int = 128, mlp_hidden_dims: tuple[int, ...] = (256,),
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(f"BrawlFeaturesExtractor requires a spaces.Dict, got {type(observation_space).__name__}")
        if not mlp_hidden_dims:
            raise ValueError("mlp_hidden_dims must have at least one layer width")

        grid_keys = [k for k, s in observation_space.spaces.items() if s.dtype == np.uint8]
        if len(grid_keys) > 1:
            raise ValueError(f"BrawlFeaturesExtractor supports at most one uint8 grid group, got {grid_keys}")
        vector_keys = [k for k in observation_space.spaces if k not in grid_keys]

        has_grid = bool(grid_keys)
        has_vector = bool(vector_keys)
        if not has_grid and not has_vector:
            raise ValueError("observation_space has no groups at all")
        features_dim = (cnn_output_dim if has_grid else 0) + (mlp_hidden_dims[-1] if has_vector else 0)
        super().__init__(observation_space, features_dim=features_dim)

        self._grid_key = grid_keys[0] if has_grid else None
        self._vector_keys = vector_keys

        if has_grid:
            c_in = observation_space.spaces[self._grid_key].shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(c_in, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.SiLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.as_tensor(observation_space.spaces[self._grid_key].sample()[None]).float()
                n_flatten = self.cnn(dummy).shape[1]
            self.cnn_linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.SiLU())
        else:
            self.cnn = None
            self.cnn_linear = None

        if has_vector:
            vector_dim = sum(int(np.prod(observation_space.spaces[k].shape)) for k in vector_keys)
            layers = []
            in_dim = vector_dim
            for width in mlp_hidden_dims:
                layers += [nn.Linear(in_dim, width), nn.SiLU()]
                in_dim = width
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = None

    def forward(self, observations: dict) -> torch.Tensor:
        parts = []
        if self.cnn is not None:
            parts.append(self.cnn_linear(self.cnn(observations[self._grid_key])))
        if self.mlp is not None:
            vec = torch.cat([torch.flatten(observations[k], start_dim=1) for k in self._vector_keys], dim=1)
            parts.append(self.mlp(vec))
        return torch.cat(parts, dim=1)


def default_policy_kwargs(spec: obs_select.AgentObsSpec, cfg) -> dict:
    """`{"features_extractor_class": BrawlFeaturesExtractor, "features_extractor_kwargs": {},
    "normalize_images": False}` -- plug straight into `PPO`/`MaskablePPO`'s `policy_kwargs`.
    Eagerly constructs a throwaway `BrawlFeaturesExtractor` against `obs_select.agent_space(spec,
    cfg)` so a shape-incompatible spec (e.g. a grid too small for the conv stack, or two uint8
    groups) fails HERE with a clear message, rather than lazily inside SB3's own policy
    construction."""
    space = obs_select.agent_space(spec, cfg)
    BrawlFeaturesExtractor(space)
    return {
        "features_extractor_class": BrawlFeaturesExtractor,
        "features_extractor_kwargs": {},
        "normalize_images": False,
    }


def float_group_names(spec: obs_select.AgentObsSpec) -> list:
    """Group names with `dtype == "float32"` -- exactly `VecNormalize(venv,
    norm_obs_keys=float_group_names(spec))`'s intended argument (see module docstring)."""
    return [g.name for g in spec.groups if g.dtype == "float32"]
