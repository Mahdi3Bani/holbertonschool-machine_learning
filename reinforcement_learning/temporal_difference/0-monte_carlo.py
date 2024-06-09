#!/usr/bin/env python3
import numpy as np
import tqdm

def reward_discount(rewards, gamma):
    """Compute the discounted sum of rewards."""
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + cumulative * gamma
        discounted_rewards[i] = cumulative
    return discounted_rewards

def generate_episode(env, policy, max_steps):
    """Generates an episode following the given policy."""
    episode = []
    state = env.reset()
    for _ in range(max_steps):
        action = policy(state)
        new_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = new_state
    return episode

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm."""
    n = env.observation_space.n
    returns = {s: [] for s in range(n)}
    for _ in tqdm.tqdm(range(episodes)):
        episode = generate_episode(env, policy, max_steps)
        states, actions, rewards = zip(*episode)
        discounts = reward_discount(rewards, gamma)
        seen_state_action_pairs = set()
        for i, (state, action, reward) in enumerate(episode):
            if (state, action) not in seen_state_action_pairs:
                seen_state_action_pairs.add((state, action))
                G = discounts[i]
                returns[state].append(G)
                V[state] += alpha * (G - V[state])
    return V
