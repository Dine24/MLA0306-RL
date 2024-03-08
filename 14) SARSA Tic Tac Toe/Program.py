import random

# Define Tic-Tac-Toe environment
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        self.reset()

    def reset(self):
        self.board = [' '] * 9
        self.current_player = 'X'

    def is_winner(self, player):
        for combo in self.winning_combos:
            if all(self.board[i] == player for i in combo):
                return True
        return False

    def is_draw(self):
        return ' ' not in self.board

    def is_game_over(self):
        return self.is_winner('X') or self.is_winner('O') or self.is_draw()

    def available_moves(self):
        return [i for i, mark in enumerate(self.board) if mark == ' ']

    def make_move(self, move):
        self.board[move] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def print_board(self):
        for i in range(0, 9, 3):
            print(' | '.join(self.board[i:i + 3]))
            if i < 6:
                print('-' * 9)

# SARSA hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Epsilon for epsilon-greedy policy

# Initialize Q-table
Q = {}

# SARSA algorithm
def update_Q_value(state, action, reward, next_state, next_action):
    if (state, action) not in Q:
        Q[(state, action)] = 0  # Initialize Q-value for unseen state-action pair
    if (next_state, next_action) not in Q:
        Q[(next_state, next_action)] = 0

    Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

# Helper function for epsilon-greedy policy
def get_action(state):
    available_moves = env.available_moves()

    if not available_moves:
        return None  # No available moves

    if random.uniform(0, 1) < epsilon:
        return random.choice(available_moves)
    else:
        return max(available_moves, key=lambda x: Q.get((state, x), 0))

# Training loop
num_episodes = 10000
env = TicTacToe()

for episode in range(num_episodes):
    env.reset()
    state = tuple(env.board)
    action = get_action(state)

    while not env.is_game_over():
        if action is None:
            # No available moves
            break

        env.make_move(action)
        next_state = tuple(env.board)

        if env.is_winner('X'):
            reward = 1
        elif env.is_winner('O'):
            reward = -1
        else:
            reward = 0

        next_action = get_action(next_state)
        update_Q_value(state, action, reward, next_state, next_action)

        state = next_state
        action = next_action

    if action is None:
        # No available moves
        break

# After training, the system uses learned Q-values to play against the player
def play_game():
    env.reset()
    state = tuple(env.board)

    while not env.is_game_over():
        if env.current_player == 'X':
            env.print_board()
            move = int(input("Enter your move (0-8): "))
            while move not in env.available_moves():
                move = int(input("Invalid move. Enter your move (0-8): "))
            env.make_move(move)
        else:
            action = get_action(state)
            if action is None:
                print("System can't make a move. It's a draw!")
                break
            env.make_move(action)

        state = tuple(env.board)

    env.print_board()
    if env.is_winner('X'):
        print("You win!")
    elif env.is_winner('O'):
        print("System wins!")
    else:
        print("It's a draw!")

# Play the game against the system
play_game()
