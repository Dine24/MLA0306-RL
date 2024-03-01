import numpy as np

# Define the grid world
n_rows, n_cols = 3, 4
grid_world = np.zeros((n_rows, n_cols))

# Define rewards
rewards = {
    (0, 3): 10,   # Cheese state
    (1, 3): -10,  # Penalty state
}

# Define discount factor
gamma = 0.9

# Define actions
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
action_names = ['Right', 'Left', 'Down', 'Up']

# Function to calculate the Bellman update for a state
def bellman_update(i, j, action):
    if (i, j) in rewards:
        return rewards[(i, j)]

    total_reward = 0
    for a, (di, dj) in enumerate(actions):
        next_i, next_j = i + di, j + dj
        if 0 <= next_i < n_rows and 0 <= next_j < n_cols:
            total_reward += 0.25 * (grid_world[next_i, next_j] * gamma)

    return total_reward

# Perform the Bellman update for state values
num_iterations = 100
for _ in range(num_iterations):
    new_grid_world = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            new_grid_world[i, j] = max([bellman_update(i, j, a) for a in actions])
    grid_world = new_grid_world

# Calculate the optimal policy
optimal_policy = np.empty((n_rows, n_cols), dtype=object)
for i in range(n_rows):
    for j in range(n_cols):
        if (i, j) in rewards:
            optimal_policy[i, j] = "Cheese" if rewards[(i, j)] > 0 else "Penalty"
        else:
            # Find the index of the action with the maximum expected reward
            action_index = np.argmax([bellman_update(i, j, a) for a in actions])
            optimal_policy[i, j] = action_names[action_index]

# Print the optimal policy
print("Optimal Policy:")
for i in range(n_rows):
    row_str = ""
    for j in range(n_cols):
        if (i, j) in rewards:
            row_str += f"{optimal_policy[i, j]:^8} | "
        else:
            row_str += f"{optimal_policy[i, j]:<8} | "
    print(row_str[:-2])  # Remove the extra "| " at the end
