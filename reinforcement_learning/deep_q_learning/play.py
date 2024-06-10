#!/usr/bin/env python3
"""deep q learning"""
import gym
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

env = gym.make('Breakout-v0')
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.load_weights('policy.h5')

dqn.test(env, nb_episodes=10, visualize=True)
