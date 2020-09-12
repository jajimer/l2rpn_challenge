"""
Neural networks.
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_dim=32, lr = 0.01):
        super(ActorCriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.lr = lr

        # 2 hidden layers
        self.body = nn.Sequential(
            nn.Linear(*self.input_dims, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        self.policy = nn.Linear(self.hidden_dim, self.n_actions)
        self.value = nn.Linear(self.hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.device)
        x = self.body(state)
        pi = self.policy(x)
        v = self.value(x)
        return pi, v


class VPGNetwork(nn.Module):
    def __init__(self, s_dim, n_actions, lr):
        super(VPGNetwork, self).__init__()
        self.input_dim = s_dim
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr)
        self.device = 'cuda:0' if T.cuda.is_available() else 'cpu'
        self.net.to(self.device)

    def forward(self, obs):
        probs = self.net(obs)
        dist = T.distributions.bernoulli.Bernoulli(probs)
        return dist