import tensorflow as tf
import numpy as np
import gym

# Author: Dr. M. Prakash

# Reinforcement Learning - Proximal Policy Optimization (PPO)

# Define a simple policy network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_head = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        action_probs = self.action_head(x)
        return action_probs

# Define the PPO agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, num_actions):
        self.policy_network = PolicyNetwork(num_actions)
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.epochs = 10
        self.clip_epsilon = 0.2
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_actions = num_actions

    def select_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs = self.policy_network(state).numpy()
        action = np.random.choice(self.num_actions, p=action_probs[0])
        return action

    def train(self, states, actions, old_action_probs, advantages):
        for _ in range(self.epochs):
            with tf.GradientTape() as tape:
                action_probs = self.policy_network(states)
                action_masks = tf.one_hot(actions, self.num_actions)
                selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
                ratio = selected_action_probs / old_action_probs
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                surrogate_objective = tf.minimum(ratio * advantages, clipped_ratio * advantages)
                loss = -tf.reduce_mean(surrogate_objective)

            grads = tape.gradient(loss, self.policy_network.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))

# Define the environment and training loop
def train_ppo_agent():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    num_actions = env.action_space.n
    agent = PPOAgent(state_dim, action_dim, num_actions)
    num_episodes = 10
    max_steps_per_episode = 200
    gamma = 0.99
    batch_size = 32

    for episode in range(num_episodes):
        states, actions, rewards, action_probs = [], [], [], []
        state = env.reset()
        total_reward = 0

        for t in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            action_probs.append(agent.policy_network(np.expand_dims(state, axis=0)).numpy()[0, action])
            total_reward += reward
            state = next_state

            if done:
                break

        # Compute advantages
        discounted_rewards = []
        advantage = 0

        for r in rewards[::-1]:
            advantage = r + gamma * advantage
            discounted_rewards.insert(0, advantage)

        # Normalize advantages
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

        # Training
        states = np.array(states)
        actions = np.array(actions)
        old_action_probs = np.array(action_probs)
        advantages = np.array(discounted_rewards)
        indices = np.arange(len(states))

        for _ in range(len(states) // batch_size):
            batch_indices = np.random.choice(indices, batch_size, replace=False)
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_action_probs = old_action_probs[batch_indices]
            batch_advantages = advantages[batch_indices]

            agent.train(batch_states, batch_actions, batch_old_action_probs, batch_advantages)

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    train_ppo_agent()
