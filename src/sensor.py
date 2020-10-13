"""
Transform the observation from the env into a useful state.
"""

import numpy as np
from gym import spaces


class Sensor(object):
    def __init__(self, env_grid2op):
        # Attributes of the grid
        self.dim_topo = env_grid2op.dim_topo
        self.reward_range = env_grid2op.reward_range
        # IDs of the objects
        self.object_types = {}
        for i, x in enumerate(env_grid2op.grid_objects_types):
            if x[1] >= 0:
                self.object_types[i] = {'id': x[1], 'type': 'load'}
            if x[2] >= 0:
                self.object_types[i] = {'id': x[2], 'type': 'gen'}
            if x[3] >= 0:
                self.object_types[i] = {'id': x[3], 'type': 'line_ex'}
            if x[4] >= 0:
                self.object_types[i] = {'id': x[4], 'type': 'line_or'}

    def get_observation_space(self):
        return spaces.Box(low=-5.0, high=5.0, 
            shape=(2, self.dim_topo, self.dim_topo), dtype=np.float32)

    def process_obs(self, obs):
        """Return (2, 177, 177) array"""
        # Adjacency matrix
        A = obs.connectivity_matrix()
        # Create feature matrix
        X = np.zeros_like(A)
        features = []
        # Indicator of gen, load or line
        features.append(np.array([1 * (v == 'line_ex') for k,v in self.object_types.items()]))
        features.append(np.array([1 * (v == 'line_or') for k,v in self.object_types.items()]))
        features.append(np.array([1 * (v == 'gen') for k,v in self.object_types.items()]))
        features.append(np.array([1 * (v == 'load') for k,v in self.object_types.items()]))
        # Indicator of bus 1 (0) or 2 (1) for each object
        features.append(obs.topo_vect - 1)
        # Resistance, generation and consumption
        P = np.zeros(A.shape[0])
        G = np.zeros(A.shape[0])
        L = np.zeros(A.shape[0])
        for k, v in self.object_types.items():
            if v['type'] in ['line_ex', 'line_or']:
                status = obs.line_status[v['id']]
                P[k] = obs.rho[v['id']] if status else -1
            elif v['type'] == 'gen':
                prods = obs.prod_p / obs.gen_pmax
                G[k] = prods[v['id']]
            elif v['type'] == 'load':
                loads = obs.load_p / obs.prod_p.sum()
                L[k] = loads[v['id']]
        features.append(P)
        features.append(G)
        features.append(L)
        # Combine all features in z
        z = np.stack(features, axis = 1)
        # X matrix must have same shape as A
        X[:z.shape[0], :z.shape[1]] = z
        return np.array([A, X])

    def process_reward(self, reward, min_ = -1., max_ = 1.):
        R_ = (reward - self.reward_range[0]) / (self.reward_range[1] - self.reward_range[0])
        R = R_ * (max_ - min_) + min_
        return R