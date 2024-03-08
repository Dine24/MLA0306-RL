import numpy as np
import tensorflow as tf
import gym

# Define the Vanilla Policy Gradient Agent
class VPGAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.policy_network = self.build_policy_network(state_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def build_policy_network(self, state_dim, action_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(state_dim,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='linear')
        ])
        return model

    def select_action(self, state):
        action_probs = self.policy_network.predict(np.array([state]))
        action_probs = np.squeeze(action_probs)  # Ensure the array is flattened
        action_probs = np.clip(action_probs, 1e-8, 1.0 - 1e-8)  # Clip probabilities to avoid numerical instability
        action_probs /= np.sum(action_probs)  # Normalize probabilities to sum to 1
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            action_probs = self.policy_network(np.array(states))
            action_masks = tf.one_hot(actions, len(action_probs[0]))
            selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            loss = -tf.reduce_sum(tf.math.log(selected_action_probs + 1e-8) * advantages)
        grads = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))

# Define the stock market environment
class StockMarketEnv:
    def __init__(self, price_data):
        self.price_data = price_data
        self.current_step = 0
        self.initial_balance = 10000  # Initial investment balance
        self.balance = self.initial_balance
        self.stock_units = 0
        self.max_steps = len(price_data) - 1

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.stock_units = 0
        return [self.balance, self.stock_units]

    def step(self, action):
        if self.current_step >= self.max_steps:
            return [self.balance, self.stock_units], 0, True

        current_price = self.price_data[self.current_step]
        next_price = self.price_data[self.current_step + 1]

        if action == 1:  # Buy
            if self.balance >= current_price:
                self.stock_units += 1
                self.balance -= current_price
        elif action == 0:  # Sell
            if self.stock_units > 0:
                self.stock_units -= 1
                self.balance += current_price

        self.current_step += 1

        # Calculate reward based on portfolio value
        portfolio_value = self.balance + (self.stock_units * next_price)
        reward = portfolio_value - self.initial_balance
        done = (self.current_step == self.max_steps)

        return [portfolio_value, self.stock_units], reward, done


# Training function for the VPG agent
def train_vpg_agent(agent, env, num_episodes=1000):
    state_dim = 2  # State: [portfolio_value, stock_units]
    action_dim = 2  # Actions: [Buy (1), Sell (0)]

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        # Print episode number and total reward only after the episode ends
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    # Generate sample price data (replace with actual stock data)
    price_data = np.random.uniform(50, 150, size=100)
    env = StockMarketEnv(price_data)
    agent = VPGAgent(2, 2)
    train_vpg_agent(agent, env, num_episodes=5)  # Adjust the number of episodes as needed
