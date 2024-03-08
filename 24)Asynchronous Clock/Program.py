import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99  # Discount factor for future rewards
        self.actor_critic = self.build_actor_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam()

    def build_actor_critic(self):
        input_state = Input(shape=(self.state_dim,))
        dense1 = Dense(32, activation='relu')(input_state)
        dense2 = Dense(32, activation='relu')(dense1)
        action_head = Dense(self.action_dim, activation='softmax')(dense2)
        critic_head = Dense(1)(dense2)

        model = tf.keras.Model(inputs=input_state, outputs=[action_head, critic_head])
        return model

    def select_action(self, state):
        action_probs, _ = self.actor_critic.predict(state)
        action = np.random.choice(len(action_probs[0]), p=action_probs[0])
        return action

    def train(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            action_probs, values = self.actor_critic(states)
            action_masks = tf.one_hot(actions, len(action_probs[0]))
            selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            advantages = self.compute_advantages(rewards, values, dones)
            actor_loss = -tf.reduce_sum(tf.math.log(selected_action_probs) * advantages)
            critic_loss = tf.reduce_sum(tf.square(rewards - values))

            total_loss = actor_loss + critic_loss

        actor_gradients = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_critic.trainable_variables))

    def compute_advantages(self, rewards, values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0
        for t in reversed(range(len(rewards) - 1)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            advantages[t] = delta + self.gamma * last_advantage * mask
            last_advantage = advantages[t]
        return advantages

class StopwatchEnv:
    def __init__(self):
        self.time_elapsed = 0

    def reset(self):
        self.time_elapsed = 0
        return [self.time_elapsed]

    def step(self, action):
        self.time_elapsed += action
        done = False
        if self.time_elapsed >= 60:
            self.time_elapsed = 0
            done = True
        return [self.time_elapsed], 1, done

def train_a2c_agent(agent, env, num_episodes=10):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(np.array([state]))
            next_state, reward, done = env.step(action)
            agent.train(np.array([state]),
                        np.array([action]),
                        np.array([reward]),
                        np.array([next_state]),
                        np.array([done]))
            state = next_state
            total_reward += reward

        print(f'Episode {episode + 1}/{num_episodes} finished. Total reward: {total_reward}')

def main():
    env = StopwatchEnv()
    agent = A2CAgent(state_dim=1, action_dim=60)
    train_a2c_agent(agent, env)

if __name__ == '__main__':
    main()

