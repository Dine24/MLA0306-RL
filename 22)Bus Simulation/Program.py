import tensorflow as tf
import numpy as np
import gym
from collections import deque
import random

# Define the Actor and Critic neural networks
class Actor(tf.keras.Model):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(400, activation='relu')
        self.dense2 = tf.keras.layers.Dense(300, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='tanh')
        self.max_action = max_action

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        actions = self.output_layer(x)
        return actions * self.max_action

class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(400, activation='relu')
        self.dense2 = tf.keras.layers.Dense(300, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, state, action):
        x = self.dense1(tf.concat([state, action], axis=-1))
        x = self.dense2(x)
        q_value = self.output_layer(x)
        return q_value

# Define the DDPG agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(action_dim, max_action)
        self.target_actor = Actor(action_dim, max_action)
        self.actor_optimizer = tf.keras.optimizers.Adam(0.001)
        self.critic = Critic()
        self.target_critic = Critic()
        self.critic_optimizer = tf.keras.optimizers.Adam(0.002)

        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.discount = 0.99
        self.tau = 0.001

    def select_action(self, state):
        return self.actor(np.expand_dims(state, axis=0))

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0, 0

        # Sample a random mini-batch from the replay buffer
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*batch))

        target_actions = self.target_actor(next_state_batch)
        target_q_values = self.target_critic(next_state_batch, target_actions)
        target_q_values = reward_batch + self.discount * target_q_values * (1 - done_batch)

        with tf.GradientTape() as tape:
            q_values = self.critic(state_batch, action_batch)
            critic_loss = tf.losses.mean_squared_error(target_q_values, q_values)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            actor_loss = -tf.reduce_mean(self.critic(state_batch, actions))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        for target, source in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
            target.assign(self.tau * source + (1 - self.tau) * target)
        for target, source in zip(self.target_actor.trainable_variables, self.actor.trainable_variables):
            target.assign(self.tau * source + (1 - self.tau) * target)

        return actor_loss, critic_loss

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

# Main training loop
def train_ddpg_agent():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    agent = DDPGAgent(state_dim, action_dim, max_action)

    num_episodes = 10
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        actor_loss, critic_loss = 0, 0
        done = False

        while not done:
            action = agent.select_action(state)
            action_array = np.squeeze(action, axis=0)  # Convert action tensor to numpy array
            next_state, reward, done, _ = env.step(action_array)
            agent.remember(state, action_array, reward, next_state, done)
            actor_loss, critic_loss = agent.train()
            total_reward += reward
            state = next_state

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

if __name__ == "__main__":
    train_ddpg_agent()

