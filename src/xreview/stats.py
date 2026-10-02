"""AUROC, bootstrap over records and a small cross-fitted logistic regression."""

import numpy as np


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    _, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=r)
    counts = np.bincount(inverse)
    return (sums / counts)[inverse]


def auroc(score: np.ndarray, wrong: np.ndarray) -> float:
    """Probability that a wrong record scores higher than a correct one (ties count one half)."""
    n_pos, n_neg = wrong.sum(), (~wrong).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks(score)[wrong].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap(stat, n: int, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([stat(rng.integers(0, n, n)) for _ in range(resamples)])


def interval(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3, steps: int = 50) -> np.ndarray:
    """Newton–Raphson for logistic regression with an intercept and a small ridge penalty."""
    a = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(a.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-a @ w))
        grad = a.T @ (p - y) + ridge * w
        hess = (a * (p * (1 - p))[:, None]).T @ a + ridge * np.eye(len(w))
        w -= np.linalg.solve(hess, grad)
    return w


def cross_fitted(x: np.ndarray, y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Out-of-fold predicted log-odds, so that every record is scored by a model not fitted on it."""
    fold = np.random.default_rng(seed).permutation(len(y)) % folds
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-12
    z = (x - mean) / std
    out = np.empty(len(y))
    for k in range(folds):
        w = fit_logistic(z[fold != k], y[fold != k].astype(float))
        out[fold == k] = np.column_stack([np.ones((fold == k).sum()), z[fold == k]]) @ w
    return out
