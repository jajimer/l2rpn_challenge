"""
Transform the observation of the environment to a useful state for the Agent.
"""

import numpy as np
import grid2op
import gym
from gym import spaces
from grid2op.Reward import L2RPNReward
from stable_baselines3.common.env_checker import check_env


class Actuator(object):
    def __init__(self, env_grid2op):
        #TODO Include change_status actions
        # Grid info
        self.sub_info = env_grid2op.sub_info
        self.n_subs = len(self.sub_info)
        self.max_connections = max(self.sub_info)
        self.num_actions = sum(self.sub_info)
        # Mask to limit action space
        self.mask = np.zeros((self.n_subs, self.num_actions),
            dtype = np.bool)
        c = 0
        for i, n in enumerate(self.sub_info):
            self.mask[i, c:c+n] = 1
            c += n
        # Grid2op action space
        self._action_space = env_grid2op.action_space
        self._do_nothing = self._action_space()
        

    def get_action_space(self):
        return spaces.MultiBinary(self.num_actions)

    def process_action(self, action):
        change_bus = action.astype('bool')
        new_action = self._action_space({'change_bus': change_bus})
        return new_action


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


class GridEnv(gym.Env):
    def __init__(self, env_name="l2rpn_neurips_2020_track1_small", env_config=None):
        super(GridEnv, self).__init__()
        # Instance of the grid2op env
        self.env_ = grid2op.make(env_name, reward_class=L2RPNReward)
        self.env_.seed(42)
        # Actuator and Sensor
        self.actuator = Actuator(self.env_)
        self.sensor = Sensor(self.env_)
        # Action space
        self.action_space = self.actuator.get_action_space()
        # Observation space
        self.observation_space = self.sensor.get_observation_space()
        # Number of substations
        self.num_substations = self.actuator.n_subs
        # Mask
        self.mask = self.actuator.mask

    def reset(self):
        obs_ = self.env_.reset()
        obs = self.sensor.process_obs(obs_)
        return obs

    def step(self, action):
        action_ = self.actuator.process_action(action)
        obs_, reward_, done, info = self.env_.step(action_)
        obs = self.sensor.process_obs(obs_)
        reward = self.sensor.process_reward(reward_)
        return obs, reward, done, info

    def render(self):
        pass
    
    def close(self):
        pass


if __name__ == '__main__':
    env_name = "l2rpn_neurips_2020_track1_small"
    env = GridEnv(env_name)
    check_env(env)
    # Doing nothing
    for i in range(3):
        r = 0
        done = False
        obs = env.reset()
        t = 0
        while not done:
            a = env.action_space.sample() * 0
            _, reward, done, info = env.step(a)
            t += 1
            r += reward
        print(i, t, r)



