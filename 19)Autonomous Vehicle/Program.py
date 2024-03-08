import numpy as np

# Define a simple grid world environment
grid_world = np.array([
    ['S', 'E', 'E', 'E'],
    ['E', 'O', 'E', 'O'],
    ['E', 'E', 'E', 'E'],
    ['O', 'E', 'E', 'G']
])

# Traditional Q-Learning agent
class QLearningAgent:
    def __init__(self, num_states, num_actions, epsilon=0.1, alpha=0.1, gamma=0.9):
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

# Function to convert the grid world to states
def grid_to_states(grid):
    return [s for s in grid.reshape(-1) if s != 'O' and s != 'E']

# Function to find the index of a state in the grid world
def find_state_index(grid, state):
    return np.where(grid == state)[0][0]

# Training the Q-Learning agent
def train_q_learning_agent(agent, grid, goal_state, episodes, max_iterations_per_episode=10):
    for episode in range(episodes):
        current_state = 'S'
        consecutive_obstacle_count = 0  # Track consecutive steps into obstacles
        iteration_count = 0

        while current_state != goal_state and iteration_count < max_iterations_per_episode:
            state_index = find_state_index(grid, current_state)
            action = agent.choose_action(state_index)

            # Get indices for the current state
            row, col = np.where(grid == current_state)
            row, col = row[0], col[0]

            # Update next state based on the chosen action and handle boundary conditions
            if action == 0 and row > 0:  # Move Up
                next_state = grid[row - 1, col]
            elif action == 1 and row < grid.shape[0] - 1:  # Move Down
                next_state = grid[row + 1, col]
            elif action == 2 and col > 0:  # Move Left
                next_state = grid[row, col - 1]
            elif action == 3 and col < grid.shape[1] - 1:  # Move Right
                next_state = grid[row, col + 1]
            else:
                next_state = current_state

            # Check if the next state is an obstacle
            is_obstacle = (next_state == 'O')

            # If consecutive steps into obstacles exceed a threshold, break the loop
            if is_obstacle:
                consecutive_obstacle_count += 1
                if consecutive_obstacle_count > 10:
                    break
            else:
                consecutive_obstacle_count = 0
                current_state = next_state

            reward = -1 if not is_obstacle else -100  # Negative reward for obstacles
            next_state_index = find_state_index(grid, next_state)
            agent.update_q_table(state_index, action, reward, next_state_index)

            print(f"Episode: {episode + 1}, Iteration: {iteration_count + 1}, Current State: {current_state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

            iteration_count += 1

        print(f"End of Episode {episode + 1}")
        agent.epsilon *= 0.99  # Optionally decay epsilon after each episode

# Define agent, states, and actions
states = grid_to_states(grid_world)
num_states = len(states)
num_actions = 4  # Up, Down, Left, Right
q_agent = QLearningAgent(num_states, num_actions)

# Train the Q-Learning agent with fewer episodes for faster output
train_q_learning_agent(q_agent, grid_world, 'G', episodes=10)

# Display the learned Q-Values
print("\nLearned Q-Values:")
print(q_agent.q_table)
