"""
Actions to perform --> Bus change and/or line status change
"""

import numpy as np

def get_action_dict(action_space, list_keys = ['change_bus', 'change_line_status']):
    """Return the # of actions and the actions of each type"""
    d_actions = {}
    A = 0

    if 'change_bus' in list_keys:
        dim_topo = action_space.dim_topo
        d_actions['change_bus'] = dim_topo
        A += dim_topo
    if 'change_line_status' in list_keys:
        n_lines = action_space.n_line
        d_actions['change_line_status'] = n_lines
        A += n_lines

    return A, d_actions


class Actuator(object):
    def __init__(self, action_space, included_actions = ['change_bus', 'change_line_status']):
        self.action_space = action_space
        self.dim_topo = action_space.dim_topo
        self.n_lines = action_space.n_line
        self.actions = {}
        self.dim_actions = []
        self.num_actions = 0
        for key in included_actions:
            if key == 'change_bus':
                self.actions[key] = np.zeros(self.dim_topo)
                self.dim_actions.append(self.dim_topo)
                self.num_actions += self.dim_topo
            if key == 'change_line_status':
                self.actions[key] = np.zeros(self.n_lines)
                self.dim_actions.append(self.n_lines)
                self.num_actions += self.n_lines

    def act_policy(self, policy):
        assert len(policy) == self.num_actions
        act = self.actions.copy()
        for k, n in zip(act, self.dim_actions):
            act[k] = policy


