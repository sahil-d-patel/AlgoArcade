"""The numbers every comparison reports, from Agarwal et al. 2021, "Deep Reinforcement Learning
at the Edge of the Statistical Precipice" (arXiv 2108.13264). rliable implements them and is
archived, so they are rebuilt here on scipy.

Report a result with `describe(scores)`, one score per seed: `IQM 0.62 [0.48, 0.71] (n=5)`, the
interquartile mean and its 95% percentile bootstrap interval. To pool tasks, pass a mapping of
task to scores; the bootstrap then resamples seeds within each task. A headline needs five seeds,
and every estimate warns below four.
"""

import warnings
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import mannwhitneyu, trim_mean

type Scores = ArrayLike | Mapping[str, ArrayLike]

MIN_SEEDS = 4
BOOTSTRAP_REPS = 10_000


def _tasks(scores: Scores) -> list[NDArray[np.float64]]:
    groups = scores.values() if isinstance(scores, Mapping) else [scores]
    return [np.asarray(group, dtype=np.float64) for group in groups]


def _warn_few(seeds: int) -> None:
    if seeds < MIN_SEEDS:
        warnings.warn(
            f"{seeds} seeds is exploratory: a headline needs five", UserWarning, stacklevel=3
        )


def iqm(scores: Scores) -> float:
    """The mean of the middle half. trim_mean cuts int(0.25 * n) scores from each end, so below
    four scores it cuts none."""
    pooled = np.concatenate(_tasks(scores))
    if len(pooled) < MIN_SEEDS:
        warnings.warn(
            f"IQM of {len(pooled)} seeds trims nothing: it equals the mean",
            UserWarning,
            stacklevel=2,
        )
    return float(trim_mean(pooled, 0.25))


def interval(scores: Scores, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval of the IQM over seeds, stratified by task when pooled."""
    tasks = _tasks(scores)
    _warn_few(min(len(task) for task in tasks))
    # A fixed generator: the same scores always print the same interval.
    rng = np.random.default_rng(0)
    resamples = np.concatenate(
        [task[rng.integers(len(task), size=(BOOTSTRAP_REPS, len(task)))] for task in tasks],
        axis=1,
    )
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(trim_mean(resamples, 0.25, axis=1), [tail, 100 - tail])
    # np.mean([0.4] * 3) is 0.4000000000000001. An IQM never leaves the scores' range, so the
    # clip removes only that rounding.
    pooled = np.concatenate(tasks)
    return (
        float(np.clip(low, pooled.min(), pooled.max())),
        float(np.clip(high, pooled.min(), pooled.max())),
    )


def probability_of_improvement(x: ArrayLike, y: ArrayLike) -> float:
    """The chance that a run of x scores above a run of y, a tie counting half: Mann-Whitney U
    for x over len(x) * len(y)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    _warn_few(min(len(x), len(y)))
    return float(mannwhitneyu(x, y).statistic / (len(x) * len(y)))


def performance_profile(scores: ArrayLike, thresholds: ArrayLike) -> NDArray[np.float64]:
    """The fraction of runs scoring above each threshold."""
    runs = np.asarray(scores, dtype=np.float64)
    _warn_few(len(runs))
    return (runs[None, :] > np.asarray(thresholds, dtype=np.float64)[:, None]).mean(axis=1)


def describe(scores: Scores, level: float = 0.95) -> str:
    low, high = interval(scores, level)
    tasks = _tasks(scores)
    pooled = f", {len(tasks)} tasks" if isinstance(scores, Mapping) else ""
    n = sum(len(task) for task in tasks)
    return f"IQM {iqm(scores):.2f} [{low:.2f}, {high:.2f}] (n={n}{pooled})"


def describe_improvement(x: ArrayLike, y: ArrayLike) -> str:
    p = probability_of_improvement(x, y)
    return f"P(X > Y) {p:.2f} (n={len(np.asarray(x))}, m={len(np.asarray(y))})"
