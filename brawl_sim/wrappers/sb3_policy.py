"""PrivilegedCriticPolicy: MaskablePPO's dict-observation policy with an asymmetric value head.

A spec's `critic:` groups (obs_select) are the sim's TRUE state: every bot, seen or not. The value
function may read them, because it only shapes the advantage during training and never acts; the
policy must not, because live play has no true state and zero-fills those buffers
(`build_agent_obs(critic=False)`). `BrawlFeaturesExtractor(critic_keys=...)` lays its output out
as `[actor features (actor_dim) | critic inputs]`; this module's extractor hands the policy MLP
the first `actor_dim` columns and the value MLP all of them.
archive/CRITIC_AND_MEMORY_PLAN.md part B.

Only `_build_mlp_extractor` is overridden, SB3's documented hook. Every path to the actor
(`forward`, `evaluate_actions`, `get_distribution`, `predict`) goes through
`mlp_extractor.forward_actor`, so the slice below is the whole guarantee that a critic input
cannot move the policy.

**The import path is a contract.** `MaskablePPO.save` pickles the policy class by reference and
`MaskablePPO.load` imports it again, so moving or renaming this module or the class breaks every
checkpoint trained with it.

Hard-requires `stable_baselines3` and `sb3_contrib`, like `wrappers/sb3_features.py`.
"""
import torch
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor


class SplitMlpExtractor(MlpExtractor):
    """SB3's `MlpExtractor`, with the policy net over the first `actor_dim` feature columns only.
    The value net reads all `feature_dim` of them."""

    def __init__(self, actor_dim: int, feature_dim: int, net_arch, activation_fn, device="auto"):
        super().__init__(feature_dim, net_arch=net_arch, activation_fn=activation_fn, device=device)
        actor = MlpExtractor(actor_dim, net_arch=net_arch, activation_fn=activation_fn, device=device)
        self.actor_dim = actor_dim
        self.policy_net = actor.policy_net
        self.latent_dim_pi = actor.latent_dim_pi

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features[:, :self.actor_dim])


class PrivilegedCriticPolicy(MaskableMultiInputActorCriticPolicy):
    """The maskable dict policy with a `SplitMlpExtractor`; needs `BrawlFeaturesExtractor`."""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SplitMlpExtractor(
            self.features_extractor.actor_dim, self.features_dim, net_arch=self.net_arch,
            activation_fn=self.activation_fn, device=self.device)
