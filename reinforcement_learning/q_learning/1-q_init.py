#!/usr/bin/env python3
'''q learning'''


import numpy as np


def q_init(env):
    """doing the q table"""
    num_state = env.observation_space.n
    num_action = env.action_space.n
    q_table = np.zeros((num_state, num_action))
    return q_table
