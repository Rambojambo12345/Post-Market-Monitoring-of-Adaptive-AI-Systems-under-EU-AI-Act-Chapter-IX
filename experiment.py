"""Train an agent, deploy it, switch goal/opponent at `change`, record detector alarms."""

from collections import namedtuple

import numpy as np

from agent import QAgent
from detectors import make_detectors
from envs import Maze, Opponent, TicTacToe, cell_index

SETTINGS = {
    "maze": dict(train=1500, baseline=200, change=450, total=900, targets=[0, 1, 2, 3, 4]),
    "tictactoe": dict(train=10_000, baseline=200, change=900, total=1500, targets=[0, 1, 2, 3, 4, 5]),
}
GRACE = 200        # alarms this soon after the change count as detections
SUPPRESSION = 50   # ignore repeat alarms for this many episodes
# target 0 keeps the baseline goal/opponent: the no-drift control

# failed: goal not reached / game not won; states, actions: histograms; traj: (state, action) pairs
Episode = namedtuple("Episode", "ret failed states actions traj")


def histogram(values, size):
    counts = np.bincount(values, minlength=size).astype(float)
    return counts / max(counts.sum(), 1)


def play_maze(agent, env, train):
    pos = env.reset()
    cells, moves, total, done = [], [], 0.0, False
    while not done:
        action = agent.act(pos, [0, 1, 2, 3], train)
        new_pos, reward, done = env.step(action)
        if train:
            agent.learn(pos, action, reward, new_pos, [] if done else [0, 1, 2, 3])
        cells.append(cell_index(pos))
        moves.append(action)
        total += reward
        pos = new_pos
    failed = int(reward != 1.0)
    return Episode(total, failed, histogram(cells, 12), histogram(moves, 4), list(zip(cells, moves)))


def play_tictactoe(agent, opponent, train):
    game = TicTacToe()
    boards, moves = [], []
    while not game.done:
        board = game.view(1)
        if train and moves:
            agent.learn(boards[-1], moves[-1], 0.0, board, game.legal_moves())
        move = agent.act(board, game.legal_moves(), train)
        boards.append(board)
        moves.append(move)
        game.play(move)
        if not game.done:
            game.play(opponent.move(game))

    reward = 0.0 if game.winner is None else (1.0 if game.winner == 1 else -1.0)
    if train:
        agent.learn(boards[-1], moves[-1], reward, None, [])
    own_cells = [i for board in boards for i in range(9) if board[i] == 1]
    return Episode(reward, int(reward != 1.0), histogram(own_cells, 9), histogram(moves, 9),
                   list(zip(boards, moves)))


def deploy(domain, target, agent_seed, before_seed, after_seed):
    s = SETTINGS[domain]
    after = s["total"] - s["change"]
    if domain == "maze":
        agent = QAgent(4, seed=agent_seed)
        env = Maze(goal=0, seed=before_seed)
        for _ in range(s["train"]):
            play_maze(agent, env, train=True)
        episodes = [play_maze(agent, env, train=False) for _ in range(s["change"])]
        env = Maze(goal=target, seed=after_seed)
        episodes += [play_maze(agent, env, train=False) for _ in range(after)]
    else:
        agent = QAgent(9, seed=agent_seed)
        opponent = Opponent(0, seed=before_seed)
        for _ in range(s["train"]):
            play_tictactoe(agent, opponent, train=True)
        episodes = [play_tictactoe(agent, opponent, train=False) for _ in range(s["change"])]
        opponent = Opponent(target, seed=after_seed)
        episodes += [play_tictactoe(agent, opponent, train=False) for _ in range(after)]
    return episodes


def run_seed(domain, target, seed):
    s = SETTINGS[domain]
    agent_seed, before_seed, after_seed, detector_seed = \
        np.random.SeedSequence(seed).generate_state(4).tolist()
    episodes = deploy(domain, target, agent_seed, before_seed, after_seed)

    alarms = {}
    for name, detector in make_detectors(detector_seed).items():
        detector.fit(episodes[:s["baseline"]])
        hits, quiet_until = [], -1
        for t in range(s["baseline"], s["total"]):
            if t == s["change"]:
                quiet_until = -1   # don't let a false alarm hide the drift
            if detector.update(episodes[t]) and t > quiet_until:
                hits.append(t)
                quiet_until = t + SUPPRESSION
        alarms[name] = hits

    returns = [e.ret for e in episodes]
    failed = [e.failed for e in episodes]
    return {
        "seed": seed,
        "return_before": float(np.mean(returns[:s["change"]])),
        "return_after": float(np.mean(returns[s["change"]:])),
        "failed_before": float(np.mean(failed[:s["change"]])),
        "failed_after": float(np.mean(failed[s["change"]:])),
        "alarms": alarms,
    }


def score(alarm_lists, change):
    """Detection rate, mean delay, mean false alarms per seed."""
    delays, false_positives = [], []
    for alarms in alarm_lists:
        false_positives.append(sum(t < change for t in alarms))
        hit = next((t for t in alarms if change <= t < change + GRACE), None)
        if hit is not None:
            delays.append(hit - change)
    return {
        "dr": len(delays) / len(alarm_lists),
        "delay": float(np.mean(delays)) if delays else None,
        "fp": float(np.mean(false_positives)),
    }
