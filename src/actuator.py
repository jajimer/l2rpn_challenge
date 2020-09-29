"""
Actions to perform --> Bus change and/or line status change
"""

import numpy as np
from gym import spaces


class Actuator(object):
    def __init__(self, env_grid2op, discrete_env):
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
        self.is_discrete = discrete_env
        

    def get_action_space(self):
        if self.is_discrete:
            return spaces.Discrete(self.num_actions + 1) # N objects + do nothing
        else:
            return spaces.MultiBinary(self.num_actions)

    def process_action(self, action):
        if self.is_discrete:
            # action is an int from 0 to self.num_actions - 1
            change_bus = np.zeros(self.num_actions, dtype = np.bool)
            if action != self.num_actions:
                # Do something
                change_bus[action] = True
        else:
            # action is a (177,) binary array
            change_bus = action.astype('bool')  
        new_action = self._action_space({'change_bus': change_bus})
        return new_action


