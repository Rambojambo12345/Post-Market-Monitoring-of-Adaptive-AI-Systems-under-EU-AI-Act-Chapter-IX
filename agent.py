"""Tabular Q-learning (Sutton & Barto, 2018)."""

import random
from collections import defaultdict


class QAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, eps_train=0.15, eps_eval=0.05, seed=None):
        self.alpha = alpha
        self.gamma = gamma
        self.eps_train = eps_train
        self.eps_eval = eps_eval
        self.rng = random.Random(seed)
        self.q = defaultdict(lambda: [0.0] * n_actions)

    def act(self, state, legal, train):
        eps = self.eps_train if train else self.eps_eval
        if self.rng.random() < eps:
            return self.rng.choice(legal)
        values = self.q[state]
        best = max(values[a] for a in legal)
        return self.rng.choice([a for a in legal if values[a] == best])

    def learn(self, state, action, reward, next_state, next_legal):
        target = reward
        if next_legal:
            target += self.gamma * max(self.q[next_state][a] for a in next_legal)
        self.q[state][action] += self.alpha * (target - self.q[state][action])
