"""
Transform the observation from the env into a useful state.
"""

import numpy as np
from gym import spaces
from time import time


class Sensor(object):
    def __init__(self, env_grid2op, N = 10):
        self.env = env_grid2op
        self.action_space = env_grid2op.action_space
        # Attributes of the grid
        self.dim_topo = env_grid2op.dim_topo
        self.reward_range = env_grid2op.reward_range
        self.max_ts_line_overflow = env_grid2op.nb_timestep_overflow_allowed
        self.ts_sub_cooldown = env_grid2op.parameters.NB_TIMESTEP_COOLDOWN_SUB
        # IDs of the objects
        self.object_types = {}
        for i, x in enumerate(env_grid2op.grid_objects_types):
            if x[1] >= 0:
                self.object_types[i] = {'id': x[1], 'type': 'load', 'sub': x[0]}
            elif x[2] >= 0:
                self.object_types[i] = {'id': x[2], 'type': 'gen', 'sub': x[0]}
            elif x[3] >= 0:
                self.object_types[i] = {'id': x[3], 'type': 'line_ex', 'sub': x[0]}
            elif x[4] >= 0:
                self.object_types[i] = {'id': x[4], 'type': 'line_or', 'sub': x[0]}
        self.num_features = N

    def get_observation_space(self):
        return spaces.Box(low=-5.0, high=5.0, 
            shape=(2, self.dim_topo, self.dim_topo), dtype=np.float32)

    def process_obs(self, obs):
        """Return (2, 177, 177) array"""
        t0 = time()
        # Adjacency matrix
        A = obs.connectivity_matrix()
        # Create feature matrix
        X = np.zeros_like(A)
        features = []
        # Indicator of gen, load or line
        features.append(np.array([1 * (v['type'] == 'line_ex') for k,v in self.object_types.items()]))
        features.append(np.array([1 * (v['type'] == 'line_or') for k,v in self.object_types.items()]))
        features.append(np.array([1 * (v['type'] == 'gen') for k,v in self.object_types.items()]))
        features.append(np.array([1 * (v['type'] == 'load') for k,v in self.object_types.items()]))
        # Indicator of bus 1 (0) or 2 (1) for each object
        features.append(obs.topo_vect - 1)
        # Resistance, generation and consumption
        P = np.zeros(A.shape[0])
        G = np.zeros(A.shape[0])
        L = np.zeros(A.shape[0])
        status = obs.line_status
        rho = obs.rho
        prods = obs.prod_p / obs.gen_pmax
        loads = obs.load_p / obs.prod_p.sum()
        for k, v in self.object_types.items():
            if v['type'] in ['line_ex', 'line_or']:
                P[k] = rho[v['id']] if status[v['id']] else -1
            elif v['type'] == 'gen':
                G[k] = prods[v['id']]
            elif v['type'] == 'load':
                L[k] = loads[v['id']]
        features.append(P)
        features.append(G)
        features.append(L)
        # Cooldowns in subs, overflows and maintenances
        C = np.zeros(A.shape[0])
        cooldown = obs.time_before_cooldown_sub
        for k, v in self.object_types.items():
            sub_id = v['sub']
            C[k] = cooldown[sub_id] / self.ts_sub_cooldown
        O = np.zeros(A.shape[0])
        overflow = obs.timestep_overflow
        for k, v in self.object_types.items():
            if v['type'] in ['line_ex', 'line_or']:
                O[k] = overflow[v['id']] / self.max_ts_line_overflow[v['id']]
        M = np.ones(A.shape[0])
        next_main = obs.time_next_maintenance
        duration_next_main = obs.duration_next_maintenance
        W_before = 1000 #timesteps
        W_during = 100 #timesteps
        for k, v in self.object_types.items():
            if v['type'] in ['line_ex', 'line_or']:
                id_ = v['id']
                t = next_main[id_]
                if t > 0:
                    T = 1.0 if t > W_before else t / W_before
                elif t == 0:
                    d = duration_next_main[id_]
                    T = -1.0 if d > W_during else -d / W_during 
                else:
                    T = 1.0
                M[k] = T
        features.append(C)
        features.append(O)
        features.append(M)
        # Combine all features in z
        z = np.stack(features, axis = 1)
        # X matrix must have same shape as A
        X[:z.shape[0], :z.shape[1]] = z
        t1 = time()
        return np.array([A, X])

    def process_reward(self, reward, action, new_range = (-1,1), 
                       do_something_penalty = 0.8, illegal_penalty = 0.2):
        # Normalize reward
        R_ = (reward - self.reward_range[0]) / (self.reward_range[1] - self.reward_range[0])
        R = R_ * (new_range[1] - new_range[0]) + new_range[0]
        # Apply penalty for doing something
        if action != self.action_space():
            R = R*do_something_penalty if R > new_range[0] else R
        # Apply penalty for illegal or ambiguous action
        if action.is_ambiguous()[0] or not self.action_space._is_legal(action, self.env)[0]:
            R -= illegal_penalty
        return R


if __name__ == "__main__":
    from environment import GridEnv
    env = GridEnv()
    for i in range(3):
        obs = env.reset()
        done = False
        while not done:
            next_obs, r, done, info = env.step(env.action_space.sample())
            print(i, r)