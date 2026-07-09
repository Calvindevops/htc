"""The correlation-study analysis: does the Agent-Ready score predict real
human-graded task performance?

Pure Python (no scipy/numpy dependency) — the harness stays dependency-light
like the rest of htc-core.
"""

from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations
from typing import Callable

from .model import Grade

RHO_THRESHOLD = 0.6
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_SEED = 0


def _rank(values: list[float]) -> list[float]:
    """Rank values 1..n, averaging ranks across ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / (var_x * var_y) ** 0.5


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation, pure Python. Returns 0.0 for degenerate
    input (fewer than 2 points, or no variance in either series)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return _pearson(_rank(xs), _rank(ys))


def bootstrap_ci(
    pairs: list[tuple[float, float]],
    statistic: Callable[[list[float], list[float]], float],
    n: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """95% percentile bootstrap CI for `statistic` over resampled `pairs`.
    Deterministic given `seed` — resampling uses `random.Random(seed)`."""
    size = len(pairs)
    if size == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    boot_stats = []
    for _ in range(n):
        sample = [pairs[rng.randrange(size)] for _ in range(size)]
        xs = [p[0] for p in sample]
        ys = [p[1] for p in sample]
        boot_stats.append(statistic(xs, ys))
    boot_stats.sort()
    lo_idx = int(0.025 * n)
    hi_idx = min(int(0.975 * n), n - 1)
    return boot_stats[lo_idx], boot_stats[hi_idx]


def inter_rater_agreement(grades: list[Grade]) -> float:
    """Fraction of grader-pairs that gave an identical score to the same
    (task_id, agent_id) attempt, averaged over all doubly-graded attempts.

    LIMITATION: this is a simplified pairwise-agreement metric, not
    Krippendorff's alpha — it does not correct for chance agreement and
    treats all score mismatches equally (no credit for near misses on the
    0-4 ordinal scale). It's a documented approximation, adequate for a
    small human-graded study.

    Returns 1.0 if no attempt was graded by more than one grader (nothing
    to disagree on).
    """
    by_item: dict[tuple[str, str], list[int]] = defaultdict(list)
    for grade in grades:
        by_item[(grade.task_id, grade.agent_id)].append(grade.score)
    agreements = [
        1.0 if a == b else 0.0
        for scores in by_item.values()
        if len(scores) >= 2
        for a, b in combinations(scores, 2)
    ]
    if not agreements:
        return 1.0
    return sum(agreements) / len(agreements)


def study_verdict(
    score_by_agent: dict[str, float],
    grades: list[Grade],
    *,
    n: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Pair each agent's Agent-Ready score with its mean human grade and test
    whether the score predicts performance.

    passed: rho >= 0.6 AND the bootstrap CI lower bound is > 0 (a positive
    correlation that's unlikely to be noise).
    """
    scores_by_agent: dict[str, list[int]] = defaultdict(list)
    for grade in grades:
        scores_by_agent[grade.agent_id].append(grade.score)
    mean_by_agent = {
        agent_id: sum(scores) / len(scores) for agent_id, scores in scores_by_agent.items()
    }

    agent_ids = sorted(set(score_by_agent) & set(mean_by_agent))
    xs = [score_by_agent[a] for a in agent_ids]
    ys = [mean_by_agent[a] for a in agent_ids]

    rho = spearman(xs, ys)
    if len(agent_ids) >= 2:
        ci_lo, ci_hi = bootstrap_ci(list(zip(xs, ys)), spearman, n=n, seed=seed)
    else:
        ci_lo, ci_hi = 0.0, 0.0

    return {
        "rho": round(rho, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "passed": bool(rho >= RHO_THRESHOLD and ci_lo > 0),
        "n_points": len(agent_ids),
        "n_graders": len({g.grader_id for g in grades}),
        "inter_rater": round(inter_rater_agreement(grades), 4),
    }
