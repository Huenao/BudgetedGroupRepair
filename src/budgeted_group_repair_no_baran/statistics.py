"""Paired and cluster-aware statistics for the preliminary experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import binomtest


def percentile_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return (math.nan, math.nan)
    alpha = 1.0 - float(confidence)
    return (float(np.quantile(finite, alpha / 2)), float(np.quantile(finite, 1 - alpha / 2)))


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    cluster_key: str,
    statistic: Callable[[Sequence[Mapping[str, Any]]], float],
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row[cluster_key])].append(row)
    keys = tuple(sorted(clusters))
    if not keys:
        return (math.nan, math.nan)
    rng = np.random.default_rng(int(seed))
    estimates: list[float] = []
    for _ in range(int(replicates)):
        indices = rng.integers(0, len(keys), size=len(keys))
        sample: list[Mapping[str, Any]] = []
        for index in indices:
            sample.extend(clusters[keys[int(index)]])
        estimates.append(float(statistic(sample)))
    return percentile_interval(estimates, confidence)


def two_way_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    first_cluster_key: str,
    second_cluster_key: str,
    value_key: str,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Two-way multiplier bootstrap preserving both query dependence structures."""

    first = tuple(sorted({str(row[first_cluster_key]) for row in rows}))
    second = tuple(sorted({str(row[second_cluster_key]) for row in rows}))
    if not rows or not first or not second:
        return (math.nan, math.nan)
    rng = np.random.default_rng(int(seed))
    estimates: list[float] = []
    for _ in range(int(replicates)):
        first_weight = dict(zip(first, rng.exponential(1.0, len(first))))
        second_weight = dict(zip(second, rng.exponential(1.0, len(second))))
        weighted = 0.0
        total_weight = 0.0
        for row in rows:
            weight = first_weight[str(row[first_cluster_key])] * second_weight[
                str(row[second_cluster_key])
            ]
            weighted += weight * float(row[value_key])
            total_weight += weight
        estimates.append(weighted / total_weight if total_weight else math.nan)
    return percentile_interval(estimates, confidence)


def exact_mcnemar(n10: int, n01: int) -> float:
    discordant = int(n10) + int(n01)
    if discordant == 0:
        return 1.0
    return float(binomtest(min(int(n10), int(n01)), discordant, 0.5, alternative="two-sided").pvalue)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(((name, float(value)) for name, value in p_values.items()), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


__all__ = [
    "cluster_bootstrap",
    "exact_mcnemar",
    "holm_adjust",
    "percentile_interval",
    "two_way_cluster_bootstrap",
]
