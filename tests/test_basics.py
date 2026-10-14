import numpy as np
import pytest
from scipy.stats import ks_2samp

from detectors import KS, damerau_levenshtein, make_detectors
from envs import GOALS, HAZARD, Maze, Opponent, TicTacToe
from experiment import GRACE, Episode, score


def test_maze_goal_wall_and_hazard():
    env = Maze(goal=3)
    env.rng.random = lambda: 0.0          # no slipping
    env.reset()
    env.pos = (3, 2)
    assert env.step(0) == (GOALS[3], 1.0, True)
    env.pos = (1, 2)
    assert env.step(2)[0] == (1, 2)       # wall above
    env.pos = (1, 4)
    assert env.step(2) == (HAZARD, -1.0, True)


def test_opponent_rules():
    game = TicTacToe()
    for move in [0, 4, 1]:                # player 1 threatens 0-1-2
        game.play(move)
    assert Opponent(2, player=2).move(game) == 2      # R2 blocks
    game.play(8)
    assert Opponent(1, player=1).move(game) == 2      # R1 wins


def test_damerau_levenshtein():
    assert damerau_levenshtein("kitten", "sitting") == 3
    assert damerau_levenshtein("ab", "ba") == 1
    assert damerau_levenshtein([], [1, 2]) == 2


def test_ks_is_bonferroni_over_scipy():
    rng = np.random.default_rng(0)
    eps = [Episode(0, 0, rng.normal(size=2), None, None) for _ in range(200)]
    eps += [Episode(0, 0, rng.normal(0.3, 1, size=2), None, None) for _ in range(50)]
    ks = KS("states", window=50, stride=1)
    ks.fit(eps[:200])
    for e in eps[200:]:
        ks.recent.append(e.states)
    a, b = np.array([e.states for e in eps[:200]]), np.array([e.states for e in eps[200:]])
    expected = min(2 * min(ks_2samp(a[:, j], b[:, j]).pvalue for j in range(2)), 1.0)
    assert np.isclose(ks.p_value(), expected)


def test_score():
    assert score([[50, 110]], change=100) == {"dr": 1.0, "delay": 10.0, "fp": 1.0}
    assert score([[400]], change=100)["dr"] == 0.0


def synthetic(shift, n=700, change=400, seed=0):
    rng = np.random.default_rng(seed)
    episodes = []
    for t in range(n):
        drifted = shift and t >= change
        failed = int(rng.random() < (0.85 if drifted else 0.05))
        ret = rng.normal(-1.0, 0.2) if failed else rng.normal(0.7, 0.1)
        states = rng.dirichlet([6, 1, 1, 1] if drifted else [2, 2, 2, 2])
        actions = rng.dirichlet([1, 1, 5] if drifted else [2, 2, 2])
        path = [(0, 1), (2, 2), (3, 0)] if drifted else [(0, 0), (1, 1), (1, 1), (3, 0)]
        episodes.append(Episode(ret, failed, states, actions, path))
    return episodes


def alarms(detector, episodes, baseline=200):
    detector.fit(episodes[:baseline])
    return [t for t in range(baseline, len(episodes)) if detector.update(episodes[t])]


@pytest.mark.parametrize("name", list(make_detectors(0)))
def test_detects_an_obvious_change(name):
    hits = alarms(make_detectors(0)[name], synthetic(shift=True))
    assert any(400 <= t < 400 + GRACE for t in hits)


@pytest.mark.parametrize("name", list(make_detectors(0)))
def test_mostly_quiet_without_change(name):
    hits = alarms(make_detectors(0)[name], synthetic(shift=False))
    assert len(hits) <= 15


@pytest.mark.parametrize("name", [n for n, d in make_detectors(0).items() if hasattr(d, "detector_type")])
def test_streaming_detectors_quiet_on_a_constant_stream(name):
    hits = alarms(make_detectors(0)[name], [Episode(1.0, 0, None, None, None)] * 600)
    assert hits == []
