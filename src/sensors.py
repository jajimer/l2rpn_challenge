"""
Transform the observation of the environment to a useful state for the Agent.
"""

import numpy as np


def obs_to_state(obs):
    d_obs = obs.to_dict()
    state = np.hstack([
        d_obs['topo_vect'], 
        d_obs['line_status'],
        d_obs['rho'],
    ]).astype('float32')
    return state