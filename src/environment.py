"""
L2RPN environment.
"""


import gym

import grid2op
from grid2op.Reward import L2RPNReward

from stable_baselines3.common.env_checker import check_env

from sensor import Sensor
from actuator import Actuator


class GridEnv(gym.Env):
    def __init__(self, env_name = "l2rpn_neurips_2020_track1_small",
                 seed = 42, use_backend = False, env_config = None):
        super(GridEnv, self).__init__()
        # Instance of the grid2op env
        if use_backend:
            from lightsim2grid.LightSimBackend import LightSimBackend
            backend = LightSimBackend()
            self.env_ = grid2op.make(env_name, reward_class=L2RPNReward, backend=backend)
        else:
            self.env_ = grid2op.make(env_name, reward_class=L2RPNReward)
        self.env_.seed(seed)
        # Actuator and Sensor
        self.actuator = Actuator(self.env_)
        self.sensor = Sensor(self.env_)
        # Action space
        self.action_space = self.actuator.get_action_space()
        # Observation space
        self.observation_space = self.sensor.get_observation_space()

    def reset(self):
        obs_ = self.env_.reset()
        obs = self.sensor.process_obs(obs_)
        return obs

    def step(self, action):
        try:
            action_ = self.actuator.process_action(action)
            obs_, reward_, done, info = self.env_.step(action_)
        except:
            print('This action cause an error:')
            print(action)
            obs_, reward_, done, info = self.env_.step(self.actuator._do_nothing)
        obs = self.sensor.process_obs(obs_)
        reward = self.sensor.process_reward(reward_)
        return obs, reward, done, info

    def render(self):
        pass
    
    def close(self):
        pass


if __name__ == '__main__':
    env = GridEnv()
    try:
        check_env(env)
        print('Environment ready!')
    except:
        print('Something wrong')



