import numpy as np
import gym

# Define the environment
env = gym.make('FrozenLake-v1')  # You can replace this with a different environment if needed

# Function to run the training for a given algorithm
def run_algorithm(algorithm, num_episodes=100):
    total_reward = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = algorithm.select_action(state)
            next_state, reward, done, _ = env.step(action)
            algorithm.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
    return total_reward

# Define TD(0) algorithm
class TD0Algorithm:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99):
        self.values = np.zeros(num_states)
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        return np.argmax(self.values[state])

    def update(self, state, action, reward, next_state):
        td_error = reward + self.gamma * self.values[next_state] - self.values[state]
        self.values[state] += self.alpha * td_error

# Define SARSA algorithm
class SARSAAlgorithm:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.values = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.values[state]))
        else:
            return np.argmax(self.values[state])

    def update(self, state, action, reward, next_state):
        next_action = self.select_action(next_state)
        td_error = reward + self.gamma * self.values[next_state, next_action] - self.values[state, action]
        self.values[state, action] += self.alpha * td_error

# Define Q-Learning algorithm
class QLearningAlgorithm:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.values = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.values[state]))
        else:
            return np.argmax(self.values[state])

    def update(self, state, action, reward, next_state):
        td_error = reward + self.gamma * np.max(self.values[next_state]) - self.values[state, action]
        self.values[state, action] += self.alpha * td_error

# Run the algorithms and compare their performance
num_episodes = 100

td0_algorithm = TD0Algorithm(env.observation_space.n, env.action_space.n)
sarsa_algorithm = SARSAAlgorithm(env.observation_space.n, env.action_space.n)
qlearning_algorithm = QLearningAlgorithm(env.observation_space.n, env.action_space.n)

total_reward_td0 = run_algorithm(td0_algorithm, num_episodes)
total_reward_sarsa = run_algorithm(sarsa_algorithm, num_episodes)
total_reward_qlearning = run_algorithm(qlearning_algorithm, num_episodes)

print("Total reward for TD(0):", total_reward_td0)
print("Total reward for SARSA:", total_reward_sarsa)
print("Total reward for Q-Learning:", total_reward_qlearning)
