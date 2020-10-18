"""
Actions to perform --> Bus change and/or line status change
"""

import numpy as np
from gym import spaces


def create_action_array(env, N=100, ratios = {'default': 0.0}):
    """
    Based on: 
    https://github.com/ZM-Learn/L2RPN_WCCI_a_Solution/blob/master/Data_structure_process.py
    """
    action_space = env.action_space
    no_action_vector = env.action_space({}).to_vect()
    # All topology actions
    topology = action_space.get_all_unitary_topologies_set(env.action_space)
    # Create array
    actions_array = no_action_vector
    print('Getting topology actions...')
    # Initialize target and current ratios
    ratios['otros'] = 1.0 - sum(ratios.values())
    current_ratios = {k: 0.0 for k in ratios.keys()}
    n = 0
    it = 0
    while n < N:
        # Random selection
        ix = np.random.randint(0, len(topology))
        action = topology.pop(ix)
        # Get substation id
        sub_id = int(action.as_dict().get("set_bus_vect").get("modif_subs_id")[0])
        sub_id = sub_id if sub_id in ratios else 'otros'
        # Ratios
        r = current_ratios[sub_id]
        max_r = ratios[sub_id]
        if r <= max_r:
            actions_array = np.column_stack((action.to_vect(), actions_array))
            current_ratios[sub_id] += 1/N
            n += 1
        it += 1
        if it % 10 == 0:
            print('### %d/%d with %d actions still available###' % (n, N, len(topology)))
            print(current_ratios)
    print('Done! %d actions selected' % actions_array.shape[1])
    return actions_array


class Actuator(object):
    def __init__(self, env_grid2op, file_actions_array = 'actions_topo_array.npz'):
        # Grid lines
        self.num_lines = env_grid2op.n_line
        # Grid2op action space
        self._action_space = env_grid2op.action_space
        self._do_nothing = self._action_space()
        # Load action array
        loaded = np.load(file_actions_array)
        self.actions_array = np.transpose(loaded['actions_array'])

    def get_action_space(self):
            return spaces.Discrete(self.actions_array.shape[0] + self.num_lines)

    def process_action(self, id_action):
        # id_action is an int
        if id_action < self.num_lines:
            # This first N actions are line actions
            change_status = np.zeros(self.num_lines, dtype = np.bool)
            change_status[id_action] = True
            new_action = self._action_space({'change_line_status': change_status})
        else:
            # Topology actions
            action_vector = self.actions_array[self.num_lines - id_action]
            new_action = self._action_space.from_vect(action_vector)
        return new_action


if __name__ == "__main__":
    import grid2op
    env_name = "l2rpn_neurips_2020_track1_small"
    env = grid2op.make(env_name)
    ratios = {16: 0.2, 26: 0.1, 23: 0.1, 21: 0.1}
    N = 90
    actions = create_action_array(env, N, ratios)
    np.savez_compressed('actions_topo_array.npz', actions_array=actions)
