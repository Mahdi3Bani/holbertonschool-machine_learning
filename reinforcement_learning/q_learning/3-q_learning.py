#!/usr/bin/env python3
"""Q-learning."""
import numpy as np


epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.55):
    """
    Perform Q-learning.
    Return the update q-table and rewards per episode
    """
    total_rewards = []
    done = False
    for _ in range(episodes):
        state, _ = env.reset()
        ep_rewards = 0
        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            if done and reward == 0:
                reward = -1
            ep_rewards += reward

            old_value = Q[state, action]
            next_max = np.max(Q[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward +
                                                           gamma * next_max)

            Q[state, action] = new_value
            
            state = next_state
            if done:
                break
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        total_rewards.append(ep_rewards)
    return Q, total_rewards
