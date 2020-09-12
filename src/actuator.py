"""
Actions to perform --> Bus change and/or line status change
"""

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