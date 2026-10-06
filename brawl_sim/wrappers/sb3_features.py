"""BrawlFeaturesExtractor: the features extractor SB3 needs to accept this observation.

A working default, not a tuned architecture. SB3's `CombinedExtractor` routes any 3D `uint8`
subspace to `NatureCNN`, whose unpadded 8/4 -> 4/2 -> 3/1 stack collapses a 13 x 21 view to an
invalid size and raises. This replaces it wholesale: a small padded 3x3 CNN for the one `uint8`
grid group, and a flatten + Linear + SiLU MLP over every other (float32) group, concatenated.

`n_flatten` is measured by pushing one sample of the real observation space through the stack,
so the view size can change without touching this file; a checkpoint still loads only at its own
view size, since the observation shape and the first linear layer both depend on it.

Hard-requires `stable_baselines3` (the `sb3` optional dependency group), like
`wrappers/sb3_vecenv.py`; nothing under `core/`/`bots/`/`env.py` imports it.

**`normalize_images=False` is required, and this extractor cannot enforce it**: the POLICY reads
it (`preprocess_obs`) before any features extractor sees the data. `default_policy_kwargs`
returns it as a top-level key so it reaches the policy constructor; without it SB3 would divide
the grid's occupancy counts by 255.

**Under `VecNormalize`,** pass `norm_obs_keys=float_group_names(spec)` to exclude the `uint8`
grid group: a running mean/std would corrupt occupancy counts.
"""
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ..core import obs_select


class BrawlFeaturesExtractor(BaseFeaturesExtractor):
    """Small CNN for the one `uint8` grid group (`(C, view_h, view_w)`) + MLP for every other
    (float32) group, concatenated. Grid: 3x3 s1 -> 3x3 s2 -> 3x3 s2 (all padded, unlike
    `NatureCNN`) -> flatten -> Linear -> SiLU. Vector groups: flatten, concatenate, then
    `len(mlp_hidden_dims)` stacked Linear -> SiLU layers. Handles zero or one uint8 group and any
    number of float32 groups, as long as there is at least one group."""

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
