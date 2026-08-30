"""Conformance tests for the exact fixed-point MGreedy implementation.

The observed approximation ratios below are regression checks, not an
experimental proof of the approximation theorem.
"""

from __future__ import annotations

from decimal import localcontext
from itertools import combinations
import random

import pytest

from budgeted_group_repair_no_baran.group_objective import (
    DEFAULT_UPLIFT_SCALE,
    GroupUpliftObjective,
    quantize_uplift,
)
from budgeted_group_repair_no_baran.group_optimizer import (
    eager_gain_cost_greedy,
    exhaustive_optimum,
    lazy_gain_cost_greedy,
)


def _reference_value_units(
    objective: GroupUpliftObjective,
    selected: tuple[str, ...],
) -> int:
    best = {cell_id: 0 for cell_id in objective.cell_ids}
    for query_id in selected:
        for cell_id, value in objective.gain_units_for(query_id).items():
            best[cell_id] = max(best[cell_id], value)
    return sum(best.values())


def _reference_exact(
    objective: GroupUpliftObjective,
    costs: dict[str, int],
    budget: int,
) -> tuple[tuple[str, ...], int, int]:
    """Independent strict brute force with the production tie contract."""

    ordered = tuple(sorted(costs))
    best = ((), 0, 0)
    for size in range(1, len(ordered) + 1):
        for selected in combinations(ordered, size):
            total_cost = sum(costs[query_id] for query_id in selected)
            if total_cost > budget:
                continue
            value = _reference_value_units(objective, selected)
            incumbent_selected, incumbent_cost, incumbent_value = best
            if (
                value > incumbent_value
                or (value == incumbent_value and total_cost < incumbent_cost)
                or (
                    value == incumbent_value
                    and total_cost == incumbent_cost
                    and selected < incumbent_selected
                )
            ):
                best = (selected, total_cost, value)
    return best


def test_half_even_quantization_and_tiny_positive_gain_are_not_toleranced() -> None:
    assert quantize_uplift(5e-16) == 0
    assert quantize_uplift(1.5e-15) == 2

    objective = GroupUpliftObjective({"q": {"i": 5e-13}})
    lazy = lazy_gain_cost_greedy(objective, {"q": 1}, 1)
    exact = exhaustive_optimum(objective, {"q": 1}, 1)

    assert objective.value_units(("q",)) == 500
    assert objective.value(("q",)) == 5e-13
    assert lazy.selected_query_ids == ("q",)
    assert exact.selected_query_ids == ("q",)


def test_quantization_is_decimal_context_independent_and_scale_is_exact() -> None:
    expected = quantize_uplift(0.12345678901234567)
    with localcontext() as context:
        context.prec = 3
        assert quantize_uplift(0.12345678901234567) == expected
        objective = GroupUpliftObjective(
            {"qa": {"i": 0.123456}, "qb": {"i": 0.123457}}
        )
        assert lazy_gain_cost_greedy(objective, {"qa": 1, "qb": 1}, 1).selected_query_ids == (
            "qb",
        )

    for invalid_scale in (True, 1.5, 0, -1, float("inf")):
        with pytest.raises(ValueError, match="positive integer"):
            quantize_uplift(1.0, scale=invalid_scale)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive integer"):
            GroupUpliftObjective({"q": {"i": 1.0}}, uplift_scale=invalid_scale)  # type: ignore[arg-type]


def test_cost_and_budget_contract_is_strictly_integer_and_positive() -> None:
    objective = GroupUpliftObjective({"fits": {"i": 1}, "over": {"j": 2}})

    result = lazy_gain_cost_greedy(objective, {"fits": 10, "over": 11}, 10)
    assert result.selected_query_ids == ("fits",)
    assert result.total_cost == result.budget == 10

    with pytest.raises(ValueError, match="positive integer"):
        lazy_gain_cost_greedy(objective, {"fits": 0, "over": 11}, 10)
    with pytest.raises(ValueError, match="integer"):
        lazy_gain_cost_greedy(objective, {"fits": 1.5, "over": 11}, 10)
    with pytest.raises(ValueError, match="integer"):
        lazy_gain_cost_greedy(objective, {"fits": 10, "over": 11}, 10.5)

    above_binary64_exact_range = 2**53 + 1
    exact_serialization = lazy_gain_cost_greedy(
        GroupUpliftObjective({"q": {"i": 1}}),
        {"q": above_binary64_exact_range},
        above_binary64_exact_range,
    ).as_dict()
    assert exact_serialization["total_cost"] == above_binary64_exact_range
    assert exact_serialization["budget"] == above_binary64_exact_range
    assert exact_serialization["steps"][0]["cost"] == above_binary64_exact_range


def test_exact_density_ties_use_lexical_query_id_initially_and_when_stale() -> None:
    initial_tie = GroupUpliftObjective({"qa": {"a": 1}, "qb": {"b": 1}})
    assert lazy_gain_cost_greedy(initial_tie, {"qa": 1, "qb": 1}, 1).selected_query_ids == (
        "qa",
    )

    stale_tie = GroupUpliftObjective(
        {
            "q0": {"shared": 3, "base": 7},
            "qa": {"shared": 3, "a": 2},
            "qb": {"shared": 2, "b": 2},
        },
        uplift_scale=1,
    )
    lazy = lazy_gain_cost_greedy(stale_tie, {"q0": 1, "qa": 1, "qb": 1}, 2)
    eager = eager_gain_cost_greedy(stale_tie, {"q0": 1, "qa": 1, "qb": 1}, 2)
    assert lazy.selected_query_ids == eager.selected_query_ids == ("q0", "qa")


def test_best_single_action_is_compared_with_density_greedy() -> None:
    objective = GroupUpliftObjective(
        {"big": {"i_big": 1.0}, "small": {"i_small": 0.2}}
    )
    result = lazy_gain_cost_greedy(objective, {"big": 10, "small": 1}, 10)
    assert result.selected_query_ids == ("big",)
    assert result.algorithm == "lazy_group_gain_cost_best_single_action"


def test_random_lazy_eager_exact_and_approximation_conformance() -> None:
    rng = random.Random(4_815_162_342)
    for _ in range(150):
        action_count = rng.randint(2, 9)
        cell_count = rng.randint(1, 8)
        gains: dict[str, dict[str, int]] = {}
        for action_index in range(action_count):
            query_id = f"q{action_index:02d}"
            vector = {
                f"i{cell_index:02d}": rng.randint(0, 9)
                for cell_index in range(cell_count)
                if rng.random() < 0.5
            }
            if not vector:
                vector[f"i{rng.randrange(cell_count):02d}"] = rng.randint(0, 9)
            gains[query_id] = vector

        objective = GroupUpliftObjective(gains, uplift_scale=1)
        costs = {query_id: rng.randint(1, 5) for query_id in gains}
        budget = rng.randint(0, sum(costs.values()))
        lazy = lazy_gain_cost_greedy(objective, costs, budget)
        eager = eager_gain_cost_greedy(objective, costs, budget)
        exact = exhaustive_optimum(objective, costs, budget)
        reference_selected, reference_cost, reference_value = _reference_exact(
            objective, costs, budget
        )

        assert lazy.selected_query_ids == eager.selected_query_ids
        assert lazy.objective_units == eager.objective_units
        assert lazy.total_cost == eager.total_cost <= budget
        assert lazy.objective_units == _reference_value_units(
            objective, lazy.selected_query_ids
        )
        assert exact.selected_query_ids == reference_selected
        assert exact.total_cost == reference_cost
        assert exact.objective_units == reference_value
        if reference_value > 0:
            assert lazy.objective_units / reference_value >= 0.405


def test_fixed_point_objective_is_normalized_monotone_and_submodular() -> None:
    objective = GroupUpliftObjective(
        {
            "q0": {"i0": 4, "i1": 1},
            "q1": {"i0": 2, "i2": 5},
            "q2": {"i1": 3, "i2": 2},
        },
        uplift_scale=1,
    )
    assert objective.value_units(()) == 0
    assert objective.value_units(("q0",)) <= objective.value_units(("q0", "q1"))

    small = ("q0",)
    large = ("q0", "q1")
    assert objective.marginal_gain_units("q2", small) >= objective.marginal_gain_units(
        "q2", large
    )
    assert objective.uplift_scale in (1, DEFAULT_UPLIFT_SCALE)
