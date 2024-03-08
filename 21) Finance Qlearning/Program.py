import numpy as np

class QLearningAgent:
    def __init__(self, num_states, num_actions, epsilon=1.0, alpha=0.1, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.alpha * (target - predict)

def train_q_learning_agent(agent, num_episodes):
    for episode in range(num_episodes):
        # Simulate the environment and obtain state, action, reward, next_state
        state = 0  # Placeholder for state, replace with actual state
        # Simulate the environment and obtain next_state, reward
        next_state = 1  # Placeholder for next_state, replace with actual next_state
        reward = 0  # Placeholder for reward, replace with actual reward

        # Update Q-values based on rewards and next states
        action = agent.choose_action(state)
        agent.update_q_table(state, action, reward, next_state)

        # Decay epsilon over episodes (you can adjust the decay rate)
        agent.epsilon *= 0.99

        # Display progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode: {episode}/{num_episodes}, Epsilon: {agent.epsilon:.2f}")

# Example usage
num_states = 100  # Replace with the actual number of states in your environment
num_actions = 4  # Replace with the actual number of actions in your environment
q_agent = QLearningAgent(num_states, num_actions)

train_q_learning_agent(q_agent, num_episodes=100)
