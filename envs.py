"""Russell & Norvig 4x3 grid world and tic-tac-toe."""

import random


# (row, col), 1-indexed, row 3 on top:
#   . . . +1
#   . # . -1
#   S . . .

ROWS, COLS = 3, 4
WALL, HAZARD, START = (2, 2), (2, 4), (1, 1)
GOALS = {0: (3, 4), 1: (3, 1), 2: (1, 4), 3: (3, 3), 4: (1, 1)}
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]          # right, left, up, down
SIDEWAYS = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}


def cell_index(pos):
    return (pos[0] - 1) * COLS + (pos[1] - 1)


class Maze:
    def __init__(self, goal=0, step_reward=-0.04, max_steps=200, seed=None):
        self.goal = GOALS[goal]
        self.step_reward = step_reward
        self.max_steps = max_steps
        self.rng = random.Random(seed)

    def reset(self):
        self.pos = START
        self.steps = 0
        return self.pos

    def step(self, action):
        # 0.8 intended, 0.1 each side
        roll = self.rng.random()
        if roll >= 0.8:
            action = SIDEWAYS[action][0 if roll < 0.9 else 1]

        dr, dc = MOVES[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if 1 <= r <= ROWS and 1 <= c <= COLS and (r, c) != WALL:
            self.pos = (r, c)
        self.steps += 1

        if self.pos == self.goal:
            return self.pos, 1.0, True
        if self.pos == HAZARD:
            return self.pos, -1.0, True
        return self.pos, self.step_reward, self.steps >= self.max_steps


# Tic-tac-toe: cells 0-8, player 1 moves first

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


class TicTacToe:
    def __init__(self):
        self.board = [0] * 9
        self.turn = 1
        self.winner = None
        self.done = False

    def legal_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def play(self, move):
        self.board[move] = self.turn
        if self.wins(self.turn):
            self.winner, self.done = self.turn, True
        elif 0 not in self.board:
            self.done = True
        self.turn = 3 - self.turn

    def wins(self, player):
        return any(all(self.board[i] == player for i in line) for line in LINES)

    def view(self, player):
        return tuple(0 if v == 0 else (1 if v == player else -1) for v in self.board)

    def winning_moves(self, player):
        found = []
        for m in self.legal_moves():
            self.board[m] = player
            if self.wins(player):
                found.append(m)
            self.board[m] = 0
        return found

    def fork_moves(self, player):
        found = []
        for m in self.legal_moves():
            self.board[m] = player
            if len(self.winning_moves(player)) >= 2:
                found.append(m)
            self.board[m] = 0
        return found

    def open_lines(self, player, move):
        return sum(1 for line in LINES
                   if move in line and all(self.board[i] != 3 - player for i in line))


class Opponent:
    """R0-R5: each level adds one rule (Zain et al., 2020)."""

    def __init__(self, level, player=2, seed=None):
        self.level = level
        self.me, self.other = player, 3 - player
        self.rng = random.Random(seed)

    def move(self, game):
        rules = [
            lambda: game.winning_moves(self.me),        # R1 win now
            lambda: game.winning_moves(self.other),     # R2 block
            lambda: game.fork_moves(self.me),           # R3 fork
            lambda: game.fork_moves(self.other),        # R4 block a fork
            lambda: self.best_positional(game),         # R5 most open lines
        ]
        for rule in rules[:self.level]:
            options = rule()
            if options:
                return self.rng.choice(options)
        return self.rng.choice(game.legal_moves())

    def best_positional(self, game):
        scores = {m: game.open_lines(self.me, m) for m in game.legal_moves()}
        best = max(scores.values())
        return [m for m, s in scores.items() if s == best]
