import numpy as np

# Define the grid world
grid_world = np.zeros((3, 3))

# Define the state transition function (up, down, left, right)
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# Define the reward for each state
rewards = {
    (0, 2): 10,  # Goal state
    (1, 2): -10,  # Penalty state
}

# Define the discount factor
gamma = 0.9

# Perform the Bellman update for state values
num_iterations = 100
for _ in range(num_iterations):
    new_grid_world = np.copy(grid_world)
    for i in range(3):
        for j in range(3):
            if (i, j) not in rewards:
                new_values = []
                for action in actions:
                    next_i, next_j = i + action[0], j + action[1]
                    if 0 <= next_i < 3 and 0 <= next_j < 3:
                        new_values.append(rewards.get((next_i, next_j), 0) + grid_world[next_i, next_j])
                if new_values:
                    new_grid_world[i, j] = max(new_values) * gamma
    grid_world = new_grid_world

# Print the final state values
print("State Values:")
print(grid_world)
