"""Deterministic budgeted selection for fixed group-query actions."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import combinations
import math
from typing import Mapping, Sequence

from .group_objective import GroupUpliftObjective


_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SelectionStep:
    """One accepted query in the greedy audit trace."""

    query_id: str
    cost: float
    marginal_gain: float
    gain_per_cost: float

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "cost": float(self.cost),
            "marginal_gain": float(self.marginal_gain),
            "gain_per_cost": _serializable_ratio(self.gain_per_cost),
        }


@dataclass(frozen=True)
class SelectionResult:
    """Immutable, serializable result of one budget slice."""

    selected_query_ids: tuple[str, ...]
    total_cost: float
    objective_value: float
    budget: float
    algorithm: str
    steps: tuple[SelectionStep, ...] = ()

    @property
    def selected(self) -> tuple[str, ...]:
        """Compatibility alias for generic optimizer callers."""

        return self.selected_query_ids

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_query_ids": list(self.selected_query_ids),
            "selected": list(self.selected_query_ids),
            "total_cost": float(self.total_cost),
            "objective_value": float(self.objective_value),
            "budget": float(self.budget),
            "algorithm": self.algorithm,
            "steps": [step.as_dict() for step in self.steps],
        }

    def to_dict(self) -> dict[str, object]:
        return self.as_dict()


def lazy_gain_cost_greedy(
    objective: GroupUpliftObjective,
    costs: Mapping[str, float],
    budget: float,
    *,
    candidates: Sequence[str] | None = None,
) -> SelectionResult:
    """Run overlap-aware lazy marginal-gain/cost greedy.

    Initial densities are valid upper bounds because the objective is monotone
    submodular.  Every tie is broken by lexical ``query_id``.  The result is
    finally compared with the best feasible *single arbitrary query action*,
    which can be a singleton or a group.
    """

    ordered, checked_costs, checked_budget = _validate_inputs(
        objective, costs, budget, candidates
    )
    current_best = objective.empty_best()
    heap: list[tuple[float, str]] = []
    for query_id in ordered:
        gain = objective.marginal_gain_from_best(query_id, current_best)
        heapq.heappush(
            heap, (-_gain_per_cost(gain, checked_costs[query_id]), query_id)
        )

    selected: list[str] = []
    selected_set: set[str] = set()
    steps: list[SelectionStep] = []
    total_cost = 0.0

    while heap:
        _, query_id = heapq.heappop(heap)
        if query_id in selected_set:
            continue
        cost = checked_costs[query_id]
        if total_cost + cost > checked_budget + _TOLERANCE:
            continue

        gain = objective.marginal_gain_from_best(query_id, current_best)
        if gain <= _TOLERANCE:
            continue
        ratio = _gain_per_cost(gain, cost)
        next_upper = -heap[0][0] if heap else -math.inf
        if ratio + _TOLERANCE < next_upper:
            heapq.heappush(heap, (-ratio, query_id))
            continue

        selected.append(query_id)
        selected_set.add(query_id)
        total_cost += cost
        if total_cost > checked_budget + _TOLERANCE:
            raise AssertionError("optimizer exceeded its validated budget")
        current_best = objective.updated_best(current_best, query_id)
        steps.append(
            SelectionStep(
                query_id=query_id,
                cost=cost,
                marginal_gain=gain,
                gain_per_cost=ratio,
            )
        )

    greedy = SelectionResult(
        selected_query_ids=tuple(selected),
        total_cost=float(total_cost),
        objective_value=objective.value_from_best(current_best),
        budget=checked_budget,
        algorithm="lazy_group_gain_cost_greedy",
        steps=tuple(steps),
    )
    best_action = _best_single_action(
        objective, checked_costs, checked_budget, ordered
    )
    if best_action is not None and _is_better(best_action, greedy):
        return SelectionResult(
            selected_query_ids=best_action.selected_query_ids,
            total_cost=best_action.total_cost,
            objective_value=best_action.objective_value,
            budget=best_action.budget,
            algorithm="lazy_group_gain_cost_best_single_action",
            steps=best_action.steps,
        )
    return greedy


def select_queries(
    objective: GroupUpliftObjective,
    costs: Mapping[str, float],
    budget: float,
    *,
    candidates: Sequence[str] | None = None,
) -> SelectionResult:
    """Named entry point used by the experiment runner."""

    return lazy_gain_cost_greedy(
        objective, costs, budget, candidates=candidates
    )


def exhaustive_optimum(
    objective: GroupUpliftObjective,
    costs: Mapping[str, float],
    budget: float,
    *,
    candidates: Sequence[str] | None = None,
    max_candidates: int = 22,
) -> SelectionResult:
    """Return the exact optimum for a guarded small audit instance."""

    ordered, checked_costs, checked_budget = _validate_inputs(
        objective, costs, budget, candidates
    )
    if len(ordered) > max_candidates:
        raise ValueError(
            f"exhaustive selection is limited to {max_candidates} candidates; "
            f"received {len(ordered)}"
        )
    best = SelectionResult((), 0.0, 0.0, checked_budget, "exhaustive_optimum")
    for size in range(1, len(ordered) + 1):
        for selected in combinations(ordered, size):
            total_cost = sum(checked_costs[query_id] for query_id in selected)
            if total_cost > checked_budget + _TOLERANCE:
                continue
            candidate = SelectionResult(
                selected_query_ids=tuple(selected),
                total_cost=float(total_cost),
                objective_value=objective.value(selected),
                budget=checked_budget,
                algorithm="exhaustive_optimum",
            )
            if _is_better(candidate, best):
                best = candidate
    return best


def _best_single_action(
    objective: GroupUpliftObjective,
    costs: Mapping[str, float],
    budget: float,
    ordered: Sequence[str],
) -> SelectionResult | None:
    best: SelectionResult | None = None
    for query_id in ordered:
        cost = costs[query_id]
        if cost > budget + _TOLERANCE:
            continue
        value = objective.value((query_id,))
        if value <= _TOLERANCE:
            continue
        candidate = SelectionResult(
            selected_query_ids=(query_id,),
            total_cost=cost,
            objective_value=value,
            budget=budget,
            algorithm="best_single_action",
            steps=(
                SelectionStep(
                    query_id=query_id,
                    cost=cost,
                    marginal_gain=value,
                    gain_per_cost=_gain_per_cost(value, cost),
                ),
            ),
        )
        if best is None or _is_better(candidate, best):
            best = candidate
    return best


def _validate_inputs(
    objective: GroupUpliftObjective,
    costs: Mapping[str, float],
    budget: float,
    candidates: Sequence[str] | None,
) -> tuple[tuple[str, ...], dict[str, float], float]:
    checked_budget = float(budget)
    if not math.isfinite(checked_budget) or checked_budget < 0.0:
        raise ValueError("budget must be finite and non-negative")

    raw = objective.query_ids if candidates is None else tuple(str(value) for value in candidates)
    ordered = tuple(sorted(raw))
    if len(set(ordered)) != len(ordered):
        raise ValueError("candidates must be unique")
    unknown = sorted(set(ordered) - set(objective.query_ids))
    if unknown:
        raise ValueError(f"unknown candidate queries: {unknown}")

    checked_costs: dict[str, float] = {}
    for query_id in ordered:
        if query_id not in costs:
            raise ValueError(f"missing cost for query {query_id!r}")
        cost = float(costs[query_id])
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(
                f"cost for query {query_id!r} must be finite and non-negative"
            )
        checked_costs[query_id] = cost
    return ordered, checked_costs, checked_budget


def _gain_per_cost(gain: float, cost: float) -> float:
    if cost == 0.0:
        return math.inf if gain > _TOLERANCE else 0.0
    return float(gain / cost)


def _is_better(candidate: SelectionResult, incumbent: SelectionResult) -> bool:
    if candidate.objective_value > incumbent.objective_value + _TOLERANCE:
        return True
    if abs(candidate.objective_value - incumbent.objective_value) > _TOLERANCE:
        return False
    if candidate.total_cost < incumbent.total_cost - _TOLERANCE:
        return True
    if abs(candidate.total_cost - incumbent.total_cost) > _TOLERANCE:
        return False
    return tuple(sorted(candidate.selected_query_ids)) < tuple(
        sorted(incumbent.selected_query_ids)
    )


def _serializable_ratio(value: float) -> float | str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return float(value)
