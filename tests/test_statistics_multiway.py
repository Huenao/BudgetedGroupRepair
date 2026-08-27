from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from budgeted_group_repair_no_baran.statistics import (
    cluster_bootstrap,
    multiway_cluster_bootstrap,
    percentile_interval,
    two_way_cluster_bootstrap,
)


def _legacy_two_way_reference(
    rows: Sequence[Mapping[str, Any]],
    *,
    first_cluster_key: str,
    second_cluster_key: str,
    value_key: str,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact copy of the pre-multiway implementation for compatibility."""

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


def test_multiway_matches_hand_calculation_for_three_crossed_clusters() -> None:
    rows = [
        {"row": "r1", "structured": "s1", "random": "z1", "effect": 0.0},
        {"row": "r1", "structured": "s2", "random": "z2", "effect": 1.0},
        {"row": "r2", "structured": "s1", "random": "z2", "effect": 3.0},
    ]
    seed = 314

    rng = np.random.default_rng(seed)
    row_weights = rng.exponential(1.0, 2)
    structured_weights = rng.exponential(1.0, 2)
    random_weights = rng.exponential(1.0, 2)
    weights = np.asarray(
        [
            row_weights[0] * structured_weights[0] * random_weights[0],
            row_weights[0] * structured_weights[1] * random_weights[1],
            row_weights[1] * structured_weights[0] * random_weights[1],
        ]
    )
    expected = float(np.dot(weights, np.asarray([0.0, 1.0, 3.0])) / np.sum(weights))

    interval = multiway_cluster_bootstrap(
        rows,
        cluster_keys=("row", "structured", "random"),
        value_key="effect",
        replicates=1,
        seed=seed,
    )

    assert interval == pytest.approx((expected, expected))


def test_multiway_is_deterministic_and_supports_four_cluster_dimensions() -> None:
    rows = [
        {
            "row": f"r{index // 2}",
            "structured": f"s{index % 3}",
            "random": f"z{(index + 1) % 4}",
            "cell": f"c{index % 5}",
            "effect": float((index % 4) - 1),
        }
        for index in range(12)
    ]
    kwargs = {
        "cluster_keys": ("row", "structured", "random", "cell"),
        "value_key": "effect",
        "replicates": 200,
        "seed": 44,
    }

    first = multiway_cluster_bootstrap(rows, **kwargs)
    second = multiway_cluster_bootstrap(rows, **kwargs)

    assert first == second
    assert all(math.isfinite(bound) for bound in first)
    assert first[0] < first[1]


def test_two_way_bootstrap_remains_exactly_compatible_with_legacy_algorithm() -> None:
    rows = [
        {"row": "r2", "query": "q1", "effect": -1.0},
        {"row": "r1", "query": "q2", "effect": 0.5},
        {"row": "r2", "query": "q2", "effect": 1.0},
        {"row": "r3", "query": "q1", "effect": 2.0},
    ]
    kwargs = {
        "first_cluster_key": "row",
        "second_cluster_key": "query",
        "value_key": "effect",
        "replicates": 101,
        "seed": 17,
        "confidence": 0.9,
    }

    assert two_way_cluster_bootstrap(rows, **kwargs) == _legacy_two_way_reference(
        rows, **kwargs
    )


def test_existing_one_way_bootstrap_api_still_accepts_custom_statistic() -> None:
    rows = [
        {"row": "r1", "value": 1.0},
        {"row": "r2", "value": 3.0},
    ]

    interval = cluster_bootstrap(
        rows,
        cluster_key="row",
        statistic=lambda sample: sum(float(row["value"]) for row in sample) / len(sample),
        replicates=1,
        seed=5,
    )

    rng = np.random.default_rng(5)
    sampled = rng.integers(0, 2, size=2)
    expected = float(np.mean(np.asarray([1.0, 3.0])[sampled]))
    assert interval == (expected, expected)


def test_multiway_edge_case_contracts_are_explicit() -> None:
    empty = multiway_cluster_bootstrap(
        [],
        cluster_keys=("row",),
        value_key="effect",
        replicates=10,
        seed=1,
    )
    assert all(math.isnan(bound) for bound in empty)

    with pytest.raises(ValueError, match="at least one"):
        multiway_cluster_bootstrap(
            [], cluster_keys=(), value_key="effect", replicates=10, seed=1
        )
    with pytest.raises(ValueError, match="distinct"):
        multiway_cluster_bootstrap(
            [{"row": "r1", "effect": 1.0}],
            cluster_keys=("row", "row"),
            value_key="effect",
            replicates=10,
            seed=1,
        )
    with pytest.raises(KeyError, match="query"):
        multiway_cluster_bootstrap(
            [{"row": "r1", "effect": 1.0}],
            cluster_keys=("row", "query"),
            value_key="effect",
            replicates=10,
            seed=1,
        )
    with pytest.raises(KeyError, match="effect"):
        multiway_cluster_bootstrap(
            [{"row": "r1"}],
            cluster_keys=("row",),
            value_key="effect",
            replicates=10,
            seed=1,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_multiway_nonfinite_values_produce_nan_interval(value: float) -> None:
    interval = multiway_cluster_bootstrap(
        [
            {"row": "r1", "query": "q1", "effect": 1.0},
            {"row": "r2", "query": "q2", "effect": value},
        ],
        cluster_keys=("row", "query"),
        value_key="effect",
        replicates=10,
        seed=9,
    )

    assert all(math.isnan(bound) for bound in interval)
