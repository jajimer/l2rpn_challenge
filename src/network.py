"""
CNN implementation. Takes the Graph Matrix and returns a latent vector.
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
https://github.com/dsgiitr/graph_nets/blob/master/GCN/GCN_Blog%2BCode.ipynb
"""


from typing import Tuple
import gym
import torch as th
from torch import nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.W = nn.Parameter(th.rand(in_channels, out_channels, requires_grad=True))
    
    def forward(self, A, X):
        """Returns output and the embedding of A."""
        A_hat = self._conv(A)
        Z = th.bmm(A_hat, X)
        out = th.relu(th.bmm(Z, self.W.repeat(Z.size(0), 1, 1)))
        return out, A_hat

    def _conv(self, A):
        """A is adjacency matrix."""
        A_hat = A + th.matrix_power(A, 0)
        D = th.diag_embed(th.sum(A, -1), offset=0, dim1=-2, dim2=-1)
        D = D.inverse().sqrt()
        return th.bmm(th.bmm(D, A_hat), D)


class GridCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 1024):
        super(GridCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        input_channels = 10 #observation_space.shape[-1]
        self.conv1 = GCNConv(input_channels, 64)
        self.conv2 = GCNConv(64, 32)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=8, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            obs = th.as_tensor(observation_space.sample()[None])
            H, A_  = self.conv1(obs[:, 0], obs[:, 1, :, :10])
            H2, _ = self.conv2(A_, H)
            n_flatten = self.cnn(H2.unsqueeze(1)).float().shape[1]
            print(n_flatten)
        # Final layer
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        A = observations[:, 0]
        X = observations[:, 1, :, :10]
        H, A_  = self.conv1(A, X)
        H2, _ = self.conv2(A_, H) # Not sure if here I should put A or A_
        out = self.linear(self.cnn(H2.unsqueeze(1)))
        return out


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 512,
        last_layer_dim_vf: int = 256,
    ):
        super(CustomNetwork, self).__init__()
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi, bias = True),
            nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf, bias = True), 
            nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)