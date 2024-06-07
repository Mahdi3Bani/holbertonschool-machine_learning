#!/usr/bin/env python3
"""q learning"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    "literally a frozen lake"
    env = gym.make('FrozenLake-v1', map_name=map_name, desc=desc,
                   is_slippery=is_slippery, render_mode='ansi')
    return env
