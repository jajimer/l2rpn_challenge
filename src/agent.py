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
        self.mask = mask

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution.
        """
        subaction_logits = nn.Sequential(
            nn.Linear(latent_dim, self.subaction_dim),
            nn.Sigmoid()
        )
        return subaction_logits

    def proba_distribution(self, action_logits: th.Tensor, subaction_logits: th.Tensor):
        # Get prob of each action
        probs = action_logits.detach().numpy()
        # Select substations
        indexes = [np.random.choice(self.list_rows, p = p) for p in probs]
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
        mask,
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
        ortho_init_super = False
        super(CustomGridPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init_super,
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
        # Action distribution
        self.action_dist = MaskedBernoulli(36, 177, th.as_tensor(mask))
        latent_dim = self.mlp_extractor.latent_dim_pi
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim)
        # Init weights: orthogonal initialization with small initial weight for output
        if ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        # Se ejecuta en el __init__ de super()
        self.mlp_extractor = CustomNetwork(self.features_dim)
        
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, 
            latent_sde: Optional[th.Tensor] = None) -> Distribution:
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


def benchmark_random_discrete():
	env = GridEnv(discrete = True)
	total_rewards = []
	total_steps = []
	for i in range(10):
		obs = env.reset()
		reward = 0
		steps = 0
		done = False
		while not done:
			_, r, done, _ = env.step(env.action_space.sample())
			reward += r
			steps += 1
		print(i, reward, steps)
		total_rewards.append(reward)
		total_steps.append(steps)	
	print("Mean reward: %.3f +/- %.3f" % (np.mean(total_rewards), np.std(total_rewards)))
	print("Mean steps: %.3f +/- %.3f" % (np.mean(total_steps), np.std(total_steps)))
	return total_rewards, total_steps


def benchmark_random_multibinary(model):
	env = GridEnv(discrete = False)
	total_rewards = []
	total_steps = []
	for i in range(10):
		obs = env.reset()
		reward = 0
		steps = 0
		done = False
		while not done:
			_, r, done, _ = env.step(model.predict(obs)[0])
			reward += r
			steps += 1
		print(i, reward, steps)
		total_rewards.append(reward)
		total_steps.append(steps)	
	print("Mean reward: %.3f +/- %.3f" % (np.mean(total_rewards), np.std(total_rewards)))
	print("Mean steps: %.3f +/- %.3f" % (np.mean(total_steps), np.std(total_steps)))
	return total_rewards, total_steps