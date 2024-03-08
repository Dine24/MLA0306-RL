import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import gym

# Define a simple replay buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# Define the Double Deep Q-Network (DDQN) agent
class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.target_update_frequency = 100  # Update the target network every n steps

        # DQN and target DQN
        self.dqn = self.build_dqn_model()
        self.target_dqn = self.build_dqn_model()
        self.target_dqn.set_weights(self.dqn.get_weights())

        self.replay_buffer = ReplayBuffer(max_size=2000)
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor

        # Exploration parameters
        self.epsilon = 1.0  # Exploration rate
        self.min_epsilon = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate
        self.total_steps = 0

    def build_dqn_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, activation='relu', input_shape=(self.state_size,)),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.dqn.predict(np.expand_dims(state, axis=0))
        return np.argmax(q_values[0])

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        targets = self.dqn.predict(states)

        target_values = self.target_dqn.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                best_action = np.argmax(self.dqn.predict(np.expand_dims(next_states[i], axis=0))[0])
                targets[i][actions[i]] = rewards[i] + self.gamma * target_values[i][best_action]

        self.dqn.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        self.total_steps += 1
        if self.total_steps % self.target_update_frequency == 0:
            self.target_dqn.set_weights(self.dqn.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def load(self, name):
        self.dqn.load_weights(name)

    def save(self, name):
        self.dqn.save_weights(name)

# Training the DDQN agent on a simple OpenAI Gym environment
def train_ddqn_agent():
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DDQNAgent(state_size, action_size)

    episodes = 10
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.train()

        if episode % 10 == 0:
            print(f"Episode: {episode}/{episodes}, Total Steps: {agent.total_steps}, Epsilon: {agent.epsilon:.2}")

    agent.save("ddqn_model.h5")

if __name__ == "__main__":
    train_ddqn_agent()
