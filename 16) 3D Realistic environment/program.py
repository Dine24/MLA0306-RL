import numpy as np
import random

# Function to simulate taking an action in the environment (grid-like environment)
def take_action(state, action):
    if action == 0:  # Move left
        return max(0, state - 1), -1  # Moving left decrements state and incurs -1 reward
    elif action == 1:  # Move right
        return min(num_states - 1, state + 1), -1  # Moving right increments state and incurs -1 reward
    elif action == 2:  # Move up
        return max(0, state - 5), -1  # Moving up decrements state by 5 and incurs -1 reward
    elif action == 3:  # Move down
        return min(num_states - 1, state + 5), -1  # Moving down increments state by 5 and incurs -1 reward

# Initialize Q-values
num_states = 21
num_actions = 4
initial_state = 0
destination_state = 20
Q = np.zeros((num_states, num_actions))

# Define Q-Learning parameters
epsilon = 0.1
alpha = 0.1
gamma = 0.9
num_episodes = 1000  # Increase the number of episodes

# Q-Learning training
for episode in range(num_episodes):
    state = initial_state
    reached_destination = False

    while not reached_destination:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(num_actions))
        else:
            action = np.argmax(Q[state, :])

        next_state, reward = take_action(state, action)

        # Q-value update
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]))

        state = next_state
        if state == destination_state:  # Check if the destination is reached
            reached_destination = True

# Path selection using Q-values
state = initial_state
optimal_path = [state]

while state != destination_state:
    action = np.argmax(Q[state, :])
    next_state, _ = take_action(state, action)
    state = next_state
    optimal_path.append(state)

print("Optimal Path:", optimal_path)
