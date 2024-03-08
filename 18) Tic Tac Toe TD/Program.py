import numpy as np
import random

# Define the Tic-Tac-Toe environment
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.winner = None
    def reset(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.winner = None

    def make_move(self, action):
        if self.board[action] == ' ' and not self.winner:
            self.board[action] = self.current_player
            self.check_winner()
            self.switch_player()

    def switch_player(self):
        self.current_player = 'X' if self.current_player == 'O' else 'O'

    def check_winner(self):
        winning_combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for a, b, c in winning_combinations:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                self.winner = self.board[a]

    def is_game_over(self):
        return ' ' not in self.board or self.winner

    def get_state(self):
        return tuple(self.board)

# Q-Learning agent
class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.prev_state = None
        self.prev_action = None

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            available_actions = [i for i, s in enumerate(state) if s == ' ']
            return random.choice(available_actions) if available_actions else None
        else:
            if state in self.q_table:
                available_actions = [i for i, s in enumerate(state) if s == ' ']
                if available_actions:
                    return max([(i, self.q_table[state][i]) for i in available_actions], key=lambda x: x[1])[0]
                else:
                    return None
            else:
                return None

    def update_q_table(self, state, action, reward, next_state, next_action):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 9
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 9
        if self.prev_state is not None:
            self.q_table[self.prev_state][self.prev_action] += self.alpha * (
                reward + self.gamma * self.q_table[state][action] - self.q_table[self.prev_state][self.prev_action]
            )
        self.prev_state = state
        self.prev_action = action

    def reset(self):
        self.prev_state = None
        self.prev_action = None

# Training the Q-Learning agent
def train_q_learning_agent(agent, env, episodes):
    for episode in range(episodes):
        state = env.get_state()
        agent.reset()

        while not env.is_game_over():
            action = agent.choose_action(state)
            if action is None:
                break
            env.make_move(action)
            next_state = env.get_state()

            if env.winner == 'X':
                reward = 1
            elif env.winner == 'O':
                reward = -1
            else:
                reward = 0

            next_action = agent.choose_action(next_state)
            agent.update_q_table(state, action, reward, next_state, next_action)

            state = next_state

        env.reset()

# Play against the trained agent
def play_vs_agent(agent, env):
    while not env.is_game_over():
        env.make_move(agent.choose_action(env.get_state()))
        print_board(env.board)
        if env.winner:
            print(f'Winner: {env.winner}')
            break
        player_action = int(input('Enter your move (0-8): '))
        env.make_move(player_action)
        print_board(env.board)

# Helper function to display the board
def print_board(board):
    print(board[0], '|', board[1], '|', board[2])
    print('--+---+--')
    print(board[3], '|', board[4], '|', board[5])
    print('--+---+--')
    print(board[6], '|', board[7], '|', board[8])

if __name__ == '__main__':
    agent = QLearningAgent()
    env = TicTacToe()

    # Train the Q-Learning agent
    train_q_learning_agent(agent, env, episodes=10000)

    # Play against the trained agent
    print("You are playing against the trained agent (X)")
    while True:
        play_vs_agent(agent, env)
        play_again = input("Play again? (yes/no): ").strip().lower()
        if play_again != "yes":
            break
