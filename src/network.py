"""
CNN implementation. Takes the Graph Matrix and returns a latent vector.
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
"""

from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union
import gym
import torch as th
from torch import nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GridCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(GridCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        # Latent vector
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim))

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class TwoHeadedNet(nn.Module):
    """
    Constructs the Policy and Value heads from the feature vector.
    :param feature_dim: (int) Dimension of the feature vector (can be the output of a CNN)
    :param device: (th.device)
    """

    def __init__(self, feature_dim: int = 512):
        super(TwoHeadedNet, self).__init__()
        
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.feature_dim = feature_dim

        # Create networks
        self.policy_net = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, 32, (5, 5)),
            nn.ConvTranspose2d(32, 8, (4, 4), 2),
            nn.ConvTranspose2d(8, 1, (3, 6), (3, 1))
        ).to(device)

        self.value_net = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU()
        ).to(device)

        # Save dim, used to create the distributions
        self.latent_dim_pi, self.latent_dim_vf = self._get_final_dims()

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
        """
        input_pi = features[:, :, None, None] # Add HxW to feature map
        return self.policy_net(input_pi), self.value_net(features)

    def _get_final_dims(self):
        """
        :return: ((int, int), int) Shape of the final layers.
        """
        with th.no_grad():
            x = th.randn(1, self.feature_dim)
            pi, v = self.forward(x)
        return tuple(pi.shape[2:]), v.shape[1]


if __name__ == '__main__':
    from environment import GridEnv
    env = GridEnv()
    cnn = GridCNN(env.observation_space)
    net = TwoHeadedNet()
    x = th.as_tensor(env.observation_space.sample()[None])
    features = cnn(x)
    print(features.shape)
    pi, v = net(features)
    print(net.latent_dim_pi, net.latent_dim_vf)