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
        self.lines_or_id = []
        self.lines_ex_id = []
        self.gens_id = []
        self.loads_id = []
        for i, x in enumerate(env_grid2op.grid_objects_types):
            if x[1] >= 0:
                self.loads_id.append((i, x[1]))
            if x[2] >= 0:
                self.gens_id.append((i, x[2]))
            if x[3] >= 0:
                self.lines_ex_id.append((i, x[3]))
            if x[4] >= 0:
                self.lines_or_id.append((i, x[4]))

    def get_observation_space(self):
        return spaces.Box(low=-10.0, high=10.0, 
            shape=(5, self.dim_topo, self.dim_topo), dtype=np.float32)

    def process_obs(self, obs):
        """Return (5, 177, 177) array"""
        # Connectivity matrix
        C = obs.connectivity_matrix()
        # Line resistance matrix
        P = np.zeros_like(C)
        for o in self.lines_or_id:
            status = obs.line_status[o[1]]
            for e in self.lines_ex_id:
                if o[1] == e[1]:
                    P[o[1], o[0]] = obs.rho[o[1]] if status else -1
        # Generation matrix
        G = np.zeros_like(C)
        prods = obs.prod_p / obs.gen_pmax
        for g in self.gens_id:
            G[g[0], g[0]] = prods[g[1]]
        # Consumption matrix
        L = np.zeros_like(C)
        loads = obs.load_p / obs.prod_p.sum()
        for l in self.loads_id:
            L[l[0], l[0]] = loads[l[1]]
        # Bus configuration matrix
        B = np.zeros_like(C)
        topo = obs.topo_vect
        for i, t in enumerate(topo):
            B[i, i] = t
        return np.array([C, P, G, L, B])

    def process_reward(self, reward, min_ = -1., max_ = 1.):
        R_ = (reward - self.reward_range[0]) / (self.reward_range[1] - self.reward_range[0])
        R = R_ * (max_ - min_) + min_
        return R