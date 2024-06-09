#!/usr/bin/env python3
import numpy as np
import tqdm

def td_lambda(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """Lambda algorithm"""
    n = env.observation_space.n
    
    for _ in tqdm.tqdm(range(episodes)):
        state = env.reset()
        E = np.zeros(n)
        for step in range(max_steps):
            action = policy(state)
            new_state, reward, done, _ = env.step(action)
            
            delta = reward + gamma * V[new_state] - V[state]
            
            E[state] += 1

            V += alpha * delta * E
            
            E = gamma * lambtha * E
            
            if done:
                break
                
            state = new_state
    
    return V
