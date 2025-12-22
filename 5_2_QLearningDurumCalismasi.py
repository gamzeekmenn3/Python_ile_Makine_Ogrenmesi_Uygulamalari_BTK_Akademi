import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm

env = gym.make("Taxi-v3", render_mode="ansi")
env.reset()
print(env.render())
"""
0: South
1: North
2: East
3: West
4: Pickup
5: Dropoff
"""

action_space = env.action_space.n
state_space = env.observation_space.n

q_table = np.zeros((state_space, action_space))

alpha = 0.1  # Learning rate
gamma = 0.6  # Discount rate
epsilon = 0.1  # Exploration rate

for i in tqdm(range(1, 100001)):
    state, _ = env.reset()
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else: # exploit
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state

print("Training finished.")

# test
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state, _ = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
