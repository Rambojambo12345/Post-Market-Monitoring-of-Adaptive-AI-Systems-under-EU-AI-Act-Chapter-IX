"""Drift detectors. Each has fit(baseline) and update(episode) -> True on an alarm."""

import random
import warnings
from collections import deque

import numpy as np
from frouros.detectors.concept_drift import (
    ADWIN, BOCD, CUSUM, DDM, ECDDWT, EDDM, HDDMA, HDDMW, KSWIN, RDDM, STEPD,
    GeometricMovingAverage, GeometricMovingAverageConfig, KSWINConfig, PageHinkley,
)
from frouros.detectors.data_drift import MMD as FrourosMMD, KSTest
from frouros.utils.kernels import rbf_kernel as frouros_rbf
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind

warnings.filterwarnings("ignore", message="ks_2samp: Exact calculation unsuccessful")
warnings.filterwarnings("ignore", message="divide by zero", module="frouros")


def make_detectors(seed):
    return {
        # Frouros concept drift detectors, default settings
        "CUSUM (loss)": Streaming(CUSUM, "loss"),
        "Page-Hinkley (loss)": Streaming(PageHinkley, "loss"),
        # default threshold 1.0 is unreachable on [0, 1]; 0.3 as in the Frouros docs
        "Geometric MA (loss)": Streaming(GeometricMovingAverage, "loss",
                                         GeometricMovingAverageConfig(lambda_=0.3)),
        "BOCD (loss)": Streaming(BOCD, "loss"),
        "ADWIN (loss)": Streaming(ADWIN, "loss"),
        "KSWIN (loss)": Streaming(KSWIN, "loss", KSWINConfig(seed=seed)),
        "DDM (failure)": Streaming(DDM, "failed"),
        "EDDM (failure)": Streaming(EDDM, "failed"),
        "HDDM-A (failure)": Streaming(HDDMA, "failed"),
        "HDDM-W (failure)": Streaming(HDDMW, "failed"),
        "RDDM (failure)": Streaming(RDDM, "failed"),
        "ECDD-WT (failure)": Streaming(ECDDWT, "failed"),
        "STEPD (failure)": Streaming(STEPD, "failed"),
        # data drift
        "KS (return)": KS("ret"),
        "KS (states)": KS("states"),
        "KS (actions)": KS("actions"),
        "MMD (states)": MMD("states", seed),
        "DAWIDD (states)": DAWIDD("states", seed),
        "Edit-operation": EditOperation(seed),
    }


class Streaming:
    """Frouros detector on one value per episode: loss in [0, 1] or a 0/1 failure flag."""

    def __init__(self, detector, signal, config=None):
        self.detector_type, self.signal, self.config = detector, signal, config

    def value(self, episode):
        if self.signal == "failed":
            return episode.failed
        span = self.best - self.worst
        return 0.0 if span == 0 else min(max((self.best - episode.ret) / span, 0.0), 1.0)

    def fit(self, baseline):
        returns = [e.ret for e in baseline]
        self.best, self.worst = max(returns), min(returns)
        self.detector = self.detector_type(config=self.config)
        self.seen = set()
        for episode in baseline:
            self.update(episode)

    def update(self, episode):
        value = self.value(episode)
        self.detector.update(value=value)
        self.seen.add(value)
        if not self.detector.drift:
            return False
        self.detector.reset()
        # STEPD fires on a constant stream (0/0), so ignore those
        constant, self.seen = len(self.seen) == 1, set()
        return not constant


class WindowTest:
    """Last `window` episodes against the baseline, tested every `stride` episodes."""

    def __init__(self, signal, window=50, stride=5, alpha=0.05):
        self.signal, self.window, self.stride, self.alpha = signal, window, stride, alpha

    def fit(self, baseline):
        self.reference = [getattr(e, self.signal) for e in baseline]
        self.recent = deque(maxlen=self.window)
        self.count = 0

    def update(self, episode):
        self.recent.append(getattr(episode, self.signal))
        self.count += 1
        if len(self.recent) < self.window or self.count % self.stride:
            return False
        return self.p_value() < self.alpha


class KS(WindowTest):
    """Frouros KS test per dimension, Bonferroni corrected."""

    def fit(self, baseline):
        super().fit(baseline)
        ref = np.array(self.reference).reshape(len(self.reference), -1)
        self.tests = [KSTest() for _ in range(ref.shape[1])]
        for j, test in enumerate(self.tests):
            test.fit(X=ref[:, j])

    def p_value(self):
        recent = np.array(self.recent).reshape(len(self.recent), -1)
        p = min(test.compare(X=recent[:, j])[0].p_value for j, test in enumerate(self.tests))
        return min(p * len(self.tests), 1.0)


def rbf_kernel(x):
    d2 = cdist(x, x, "sqeuclidean")
    positive = d2[d2 > 0]
    bandwidth2 = np.median(positive) if len(positive) else 1.0
    return np.exp(-d2 / (2 * bandwidth2))


class MMD(WindowTest):
    """Frouros MMD (default RBF kernel), 100-permutation p-value."""

    def __init__(self, signal, seed, permutations=100):
        super().__init__(signal)
        self.rng = np.random.default_rng(seed)
        self.permutations = permutations

    def p_value(self):
        # Frouros' permutation callback opens a process pool per call, so permute here
        x, y = np.array(self.reference), np.array(self.recent)
        data, n = np.vstack([x, y]), len(x)
        mmd = lambda a, b: FrourosMMD._mmd(X=a, Y=b, kernel=frouros_rbf)
        observed = mmd(x, y)
        null = [mmd(d[:n], d[n:]) for d in (self.rng.permutation(data) for _ in range(self.permutations))]
        return (1 + np.sum(np.array(null) >= observed)) / (1 + self.permutations)


class DAWIDD(WindowTest):
    """HSIC test of window data vs. time, permutation p-value."""

    def __init__(self, signal, seed, permutations=100):
        super().__init__(signal)
        self.rng = np.random.default_rng(seed)
        self.permutations = permutations

    def p_value(self):
        x = np.array(self.recent)
        n = len(x)
        centre = np.eye(n) - 1 / n
        kx = centre @ rbf_kernel(x) @ centre
        kt = rbf_kernel(np.arange(n, dtype=float)[:, None])
        observed = np.sum(kx * kt)
        null = [np.sum(kx * kt[np.ix_(p, p)])
                for p in (self.rng.permutation(n) for _ in range(self.permutations))]
        return (1 + np.sum(np.array(null) >= observed)) / (1 + self.permutations)


class EditOperation(WindowTest):
    """Edit distances baseline-baseline vs. baseline-window, Welch t-test."""

    def __init__(self, seed, pairs=120):
        super().__init__("traj")
        self.rng = random.Random(seed)
        self.pairs = pairs

    def fit(self, baseline):
        super().fit(baseline)
        self.base_distances = [damerau_levenshtein(*self.rng.sample(self.reference, 2))
                               for _ in range(self.pairs)]

    def p_value(self):
        recent = list(self.recent)
        distances = [damerau_levenshtein(self.rng.choice(self.reference), self.rng.choice(recent))
                     for _ in range(self.pairs)]
        if len(set(self.base_distances)) == 1 and len(set(distances)) == 1:
            return 1.0 if distances[0] == self.base_distances[0] else 0.0  # t-test undefined
        return ttest_ind(self.base_distances, distances, equal_var=False).pvalue


def damerau_levenshtein(a, b):
    """Optimal string alignment distance."""
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]
