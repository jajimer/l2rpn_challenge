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

from network import TwoHeadedNet
from environment import GridEnv


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
#        mask: np.array,
#        num_substations: int = 36,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        super(CustomGridPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs
            )

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
#        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn)
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


def benchmark_do_nothing():
	env = GridEnv()
	total_rewards = []
	total_steps = []
	a = np.zeros(177, dtype = np.bool)
	for i in range(10):
		obs = env.reset()
		reward = 0
		steps = 0
		done = False
		while not done:
			_, r, done, _ = env.step(a)
			reward += r
			steps += 1
		print(i, reward, steps)
		total_rewards.append(reward)
		total_steps.append(steps)	
	print("Mean reward: %.3f +/- %.3f" % (np.mean(total_rewards), np.std(total_rewards)))
	print("Mean steps: %.3f +/- %.3f" % (np.mean(total_steps), np.std(total_steps)))
	return total_rewards, total_steps