import gym
import random
import numpy as np

environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeroos((nb_states, nb_actions))

print("Q-table:")
print(qlabel)

action = environment.action_space.sample()

"""
sol:0
asagi:1
sag=2
yukari: 3
"""
# (S1) -> (Action 1) -> S2
new_state, reward, done, info = environment.step(action)
