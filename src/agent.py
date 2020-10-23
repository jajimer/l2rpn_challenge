"""
Agent implementation.
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py
"""


import gym
import numpy as np
import torch as th
from torch import nn as nn
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union

from stable_baselines3.common.distributions import *
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from network import CustomNetwork, GridCNN
from environment import GridEnv


class MaskedBernoulli(BernoulliDistribution):
    """
    Bernoulli distribution for MultiBinary action spaces with mask.
    :param action_dim: (int) Number of binary actions
    """

    def __init__(self, action_dim: int, subaction_dim: int, mask):
        super(MaskedBernoulli, self).__init__(subaction_dim)
        self.distribution = None
        self.action_dims = action_dim
        self.subaction_dim = subaction_dim
        self.list_rows = [i for i in range(action_dim)]
        if th.cuda.is_available():
            self.mask = mask.cuda()
        else:
            self.mask = mask

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution.
        """
        subaction_logits = nn.Sequential(
            nn.Linear(latent_dim, self.subaction_dim, bias = True),
            nn.Sigmoid()
        )
        return subaction_logits

    def proba_distribution(self, action_logits: th.Tensor, subaction_logits: th.Tensor):
        # Select substations
        action_logits = (action_logits + 1.0)/2.0
        indexes = th.multinomial(action_logits, 1, replacement=True).view(-1)
        print(indexes)
        # Construct the action vector
        subaction_logits_masked = subaction_logits * self.mask[indexes,:]
        self.distribution = Bernoulli(probs=subaction_logits_masked)
        return self


class CustomGridPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = GridCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(CustomGridPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        # Se ejecuta en el __init__ de super()
        self.mlp_extractor = CustomNetwork(self.features_dim)