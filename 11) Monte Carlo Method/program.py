import numpy as np
import random

class CustomEnv:
    def __init__(self):
        self.num_states = 10
        self.actions = ['left', 'right']
        self.max_steps = 100
        self.current_state = 0
        self.reward = 0
        self.total_reward = 0
        self.steps = 0

    def reset(self):
        self.current_state = 0
        self.reward = 0
        self.total_reward = 0
        self.steps = 0
        return self.current_state

    def step(self, action):
        self.steps += 1
        if action == 'right':
            self.current_state += 1
            self.reward = 1 if self.current_state == self.num_states - 1 else 0
        else:
            self.current_state -= 1 if self.current_state > 0 else 0
            self.reward = 0

        self.total_reward += self.reward

        done = self.current_state == self.num_states - 1 or self.steps >= self.max_steps
        return self.current_state, self.reward, done, {}

# Monte Carlo Control
def monte_carlo_control(env, episodes=1000, gamma=1.0):
    returns_sum = {}
    returns_count = {}
    Q = {}
    policy = {}

    for episode in range(episodes):
        states_actions_returns = []
        state = env.reset()
        done = False

        # Generate an episode
        while not done:
            action = random.choice(env.actions)
            next_state, reward, done, _ = env.step(action)
            states_actions_returns.append((state, action, reward))
            state = next_state

        # Update Q values
        G = 0
        for i, (state, action, reward) in enumerate(reversed(states_actions_returns)):
            G = gamma * G + reward
            if (state, action) not in [(x[0], x[1]) for x in states_actions_returns[::-1][i+1:]]:
                if (state, action) in returns_sum:
                    returns_sum[(state, action)] += G
                    returns_count[(state, action)] += 1
                else:
                    returns_sum[(state, action)] = G
                    returns_count[(state, action)] = 1
                Q[(state, action)] = returns_sum[(state, action)] / returns_count[(state, action)]

        # Update policy based on Q-values
        for s in range(env.num_states):
            best_actions = [act for act in env.actions if (s, act) in Q.keys() and Q[(s, act)] == max([Q[(s, a)] for a in env.actions if (s, a) in Q.keys()])]
            policy[s] = random.choice(best_actions) if best_actions else random.choice(env.actions)

    return Q, policy

# Create an instance of the environment
env = CustomEnv()

# Monte Carlo Control for learning optimal policy
Q, optimal_policy = monte_carlo_control(env)

# Displaying the learned optimal policy
print("Learned Optimal Policy:")
for state, action in optimal_policy.items():
    print(f"State: {state}, Action: {action}")
