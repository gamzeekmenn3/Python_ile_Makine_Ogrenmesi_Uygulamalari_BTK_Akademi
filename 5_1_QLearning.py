import gymnasium as gym
import random
import numpy as np

environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table:")
print(qtable)

action = environment.action_space.sample()

new_state, reward, terminated, truncated, info = environment.step(action)
done = terminated or truncated

# %%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table:")
print(qtable)  # Ajanın beyni

episodes = 1000  # episode
alpha = 0.5      # learning rate
gamma = 0.9      # discount rate

outcomes = [] 

# training
for _ in range(episodes):
    state, _ = environment.reset()
    done = False
    
    outcomes.append("Failure")

    while not done:  # Ajan basarili olana kadar state icerisinde hareket et (action sec ve uygula)
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()

        new_state, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated

        #update q table
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        state = new_state

        if reward:
            outcomes[-1] = "Success"

print("Qtable after training: ")
print(qtable)

plt.bar(range(episodes), outcomes)
