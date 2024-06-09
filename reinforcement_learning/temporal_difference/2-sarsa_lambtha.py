#!/usr/bin/env python3
import numpy as np
import tqdm

def epsilon_greedy(Q, state, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state, :])

def sarsa_lambda(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """SARSA(λ) algorithm with epsilon-greedy policy"""
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    for _ in tqdm.tqdm(range(episodes)):
        state = env.reset()
        action = epsilon_greedy(Q, state, n_actions, epsilon)
        E = np.zeros((n_states, n_actions))
        
        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, n_actions, epsilon)
            
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            E[state, action] += 1 

            Q += alpha * delta * E
            E *= gamma * lambtha 

            if done:
                break

            state, action = next_state, next_action

        if epsilon > min_epsilon:
            epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q
