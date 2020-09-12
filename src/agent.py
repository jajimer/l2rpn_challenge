"""
Agent implementation
"""

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network import VPGNetwork

class A2C(object):
    def __init__(self, net, gamma=0.99):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor_critic = ActorCriticNetwork(alpha, input_dims, layer1_size,
                                        layer2_size, n_actions=n_actions)

    def choose_action(self, observation):
        pi, v = self.actor_critic.forward(observation)

        mu, sigma = pi
        sigma = T.exp(sigma)
        action_probs = T.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs).to(self.actor_critic.device)
        action = T.tanh(probs)

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value_ = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)

        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)
        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor_critic.optimizer.step()


class PGAgent(object):
    def __init__(self, input_dim, n_actions, lr = 3e-4, gamma = 0.99, beta = 0.01, normalize = True):
        self.rewards_memory = []
        self.log_actions_memory = []
        self.entropy_memory = []
        self.gamma = gamma
        self.beta = beta
        self.normalize = normalize

        self.policy = VPGNetwork(input_dim, n_actions, lr)

    def get_action(self, obs):
        x = T.Tensor([obs]).to(self.policy.device)
        dist = self.policy(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.log_actions_memory.append(log_prob)
        self.entropy_memory.append(dist.entropy())

        return action.item()

    
    def store_rewards(self, r):
        self.rewards_memory.append(r)
    
    def learn(self):

        def get_returns(rewards, gamma):
            G = np.zeros_like(rewards, dtype=np.float64)
            for t in range(len(rewards)):
                G_sum = 0
                discount = 1
                for k in range(t, len(rewards)):
                    G_sum += rewards[k] * discount
                    discount *= gamma
                G[t] = G_sum
            return G

        self.policy.optimizer.zero_grad()
        G = get_returns(self.rewards_memory, self.gamma)
        if self.normalize:
            G = (G - G.mean()) / (G.std() + 1e-9)
        gradients = [-log * g - self.beta * H for log, g, H in zip(self.log_actions_memory, G, self.entropy_memory)]
        loss = T.stack(gradients).sum().to(self.policy.device)
        loss.backward()
        self.policy.optimizer.step()

        self.rewards_memory = []
        self.log_actions_memory = []