import numpy as np

# Define the number of news articles and user states
n_articles = 10
n_states = 5

# Initialize Q-values with small random values
Q = 0.01 * np.random.randn(n_states, n_articles)

# Simulated reward function (example)
rewards = np.random.rand(n_states, n_articles)

# Define the TD(0) parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration parameter

# Simulate user interactions
num_episodes = 1000
for _ in range(num_episodes):
    state = np.random.randint(n_states)  # Random initial state

    while True:
        if np.random.rand() < epsilon or np.sum(Q[state, :]) == 0:
            action = np.random.randint(n_articles)  # Exploration: Randomly select an action
        else:
            action = np.argmax(Q[state, :])  # Exploitation: Select the action with the highest Q-value

        next_state = np.random.randint(n_states)  # Simulate user moving to a new state
        reward = rewards[state, action]

        # Update Q-value using TD(0) update rule
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

        if np.random.rand() < 0.1 or np.sum(Q[state, :]) == 0:
            break

# Determine the optimal policy
optimal_policy = np.argmax(Q, axis=1)

# Print the optimal policy
print("Optimal Policy:")
print(optimal_policy)
