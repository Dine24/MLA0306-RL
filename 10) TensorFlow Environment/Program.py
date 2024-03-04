import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class CustomEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float32, minimum=0, maximum=1, name='observation')
        self._state = np.array([0.5])  # Initial state

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.array([0.5])
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        if action == 0:  # Move left
            self._state -= 0.1
        else:  # Move right
            self._state += 0.1

        if self._state <= 0:
            reward = -0.5  # Updated punishment for going too far left
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        elif self._state >= 1:
            reward = 2.0  # Updated reward for reaching the goal
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        else:
            reward = 5.0  # No reward or punishment for intermediate steps
            return ts.transition(np.array(self._state, dtype=np.float32), reward=reward)

environment = CustomEnvironment()

time_step = environment.reset()
cumulative_reward = time_step.reward

for _ in range(10):
    action = np.random.randint(2)
    time_step = environment.step(action)
    print(f"Action: {action}, Next State: {time_step.observation}, Reward: {time_step.reward}")
    cumulative_reward += time_step.reward

print(f"Cumulative Reward: {cumulative_reward}")

