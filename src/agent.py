"""
Agent implementation.
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py
"""

import collections
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn as nn

from stable_baselines3.common.distributions import *
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.a2c import A2C
from stable_baselines3 import PPO
from network import GridCNN, TwoHeadedNet


def lr_(x):
    return x


class MaskedBernoulli(BernoulliDistribution):
    """
    Bernoulli distribution for MultiBinary action spaces with mask.
    :param action_dim: (int) Number of binary actions
    """

    def __init__(self, action_dim: int, subaction_dim: int, mask: th.Tensor):
        super(MaskedBernoulli, self).__init__(action_dim)
        self.distribution = None
        self.action_dims = action_dim
        self.subaction_dim = subaction_dim
        self.list_rows = [i for i in range(action_dim)]
        self.mask = mask

    def proba_distribution_net(self, action_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution.
        """
        subaction_logits = nn.Sequential(
            nn.Linear(action_dim, self.subaction_dim),
            nn.Sigmoid()
        )
        return subaction_logits

    def proba_distribution(self, action_logits: th.Tensor, subaction_logits: th.Tensor):
        # Get prob of each action
        probs = action_logits.detach().numpy().reshape(-1,)
        ix = np.random.choice(self.list_rows, p = probs)
        # Construct the action vector
        subaction_logits_masked = subaction_logits * self.mask[ix]
        self.distribution = Bernoulli(logits=subaction_logits_masked)
        return self


class CustomGridPolicy(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms. Used by A2C, PPO and the likes.
    :param env: (gym.Env) Gym environment
    :param lr_schedule: (Callable) Learning rate schedule (could be constant)
    :param masking:
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        mask: np.array,
        num_substations: int = 36,
        **kwargs
    ):
        super(CustomGridPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GridCNN,
            normalize_images=False,
            features_extractor_kwargs = kwargs)

        # Environment information
        n_buses = action_space.n
        # Action distribution
        self.action_dist = MaskedBernoulli(num_substations, n_buses, th.as_tensor(mask))
        self._build_custom(lr_schedule)

    def _build_custom(self, lr_schedule: Callable[[float], float]) -> None:
        """
        Create the networks and the optimizer.
        :param lr_schedule: (Callable) Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        # Policy and value networks
        self.mlp_extractor = TwoHeadedNet(self.features_dim)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = self.action_dist.proba_distribution_net(latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: (th.Tensor) Latent code for the actor
        :param latent_sde: (Optional[th.Tensor]) Latent code for the gSDE exploration function
        :return: (Distribution) Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(latent_pi, mean_actions)


if __name__ == '__main__':
    from environment import GridEnv
    env = GridEnv()
    agent = CustomGridPolicy(env.observation_space, env.action_space, lr_, mask = env.mask)
    model = A2C(agent, env)
    model.learn(10)