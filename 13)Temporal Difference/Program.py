import numpy as np

# Define the number of news articles and user states
n_articles = 10
n_states = 5

# Initialize Q-values
Q = np.zeros((n_states, n_articles))

# Simulated reward function (example)
rewards = np.random.rand(n_states, n_articles)

# Define the TD(0) parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Simulate user interactions
num_episodes = 1000
for _ in range(num_episodes):
    state = np.random.randint(n_states)  # Random initial state

    while True:
        # Exploitation or exploration with epsilon-greedy
        if np.random.rand() < 0.1:
            action = np.random.randint(n_articles)  # Exploration: Choose a random action
        else:
            action = np.argmax(Q[state, :])  # Exploitation: Select the action with the highest Q-value

        next_state = np.random.randint(n_states)  # Simulate user moving to a new state
        reward = rewards[state, action]

        # Update Q-value using TD(0) update rule
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

        if np.random.rand() < 0.1:  # Simulate the end of an episode with 10% probability
            break

# Determine the optimal policy
optimal_policy = np.argmax(Q, axis=1)

# Print the optimal policy
print("Optimal Policy:")
print(optimal_policy)

