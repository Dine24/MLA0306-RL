import numpy as np
import random

# Define the grid world
n_rows, n_cols = 6, 5

# Define actions (up, down, left, right)
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Define exploration probability (ε)
epsilon = 0.2

# Initialize the state values
state_values = np.zeros((n_rows, n_cols))

# Function to check if a state is within the grid boundaries
def within_bounds(state):
    row, col = state
    return 0 <= row < n_rows and 0 <= col < n_cols

# Function to choose an action using ε-greedy strategy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        # Exploration: Choose a random action
        return random.choice(range(len(actions)))
    else:
        valid_actions = []
        for a in actions:
            next_state = (state[0] + a[0], state[1] + a[1])
            if within_bounds(next_state):
                valid_actions.append(state_values[next_state])
            else:
                valid_actions.append(float('-inf'))  # Assign negative infinity to invalid actions
        return np.argmax(valid_actions)
num_episodes = 1000

for _ in range(num_episodes):
    current_state = (0, 0)

    while True:
        action = choose_action(current_state)
        move = actions[action]
        next_state = (current_state[0] + move[0], current_state[1] + move[1])

        # Simulated reward function (example)
        if next_state == (5, 3):
            reward = 1
        else:
            reward = 0

        if within_bounds(next_state):
            # Update the state value using Q-learning (temporal difference)
            state_values[current_state] += 0.1 * (
                reward + 0.9 * state_values[next_state] - state_values[current_state]
            )

            current_state = next_state
        else:
            break  # Break the loop if the next state is out of bounds

# Display the state values with exploration
print("State Values with Exploration:")
print(state_values)
