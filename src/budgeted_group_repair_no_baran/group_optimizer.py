"""Deterministic budgeted selection for fixed group-query actions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import heapq
from itertools import combinations
from typing import Mapping, Sequence

from .group_objective import GroupUpliftObjective


NUMERIC_SEMANTICS = "fixed_point_round_half_even_exact_integer_mgreedy_v1"


@dataclass(frozen=True)
class SelectionStep:
    """One accepted query in the greedy audit trace."""

    query_id: str
    cost: int
    marginal_gain: float
    gain_per_cost: float
    marginal_gain_units: int = 0
    uplift_scale: int = 1

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "cost": int(self.cost),
            "marginal_gain": float(self.marginal_gain),
            "gain_per_cost": _serializable_ratio(self.gain_per_cost),
            "cost_tokens": int(self.cost),
            "marginal_gain_units_decimal": str(self.marginal_gain_units),
            "uplift_scale": int(self.uplift_scale),
            "exact_density_numerator_decimal": str(self.marginal_gain_units),
            "exact_density_denominator_tokens": int(self.cost),
        }


@dataclass(frozen=True)
class SelectionResult:
    """Immutable, serializable result of one budget slice."""

    selected_query_ids: tuple[str, ...]
    total_cost: int
    objective_value: float
    budget: int
    algorithm: str
    objective_units: int
    uplift_scale: int
    steps: tuple[SelectionStep, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_cost, bool)
            or not isinstance(self.total_cost, int)
            or self.total_cost < 0
        ):
            raise ValueError("selection total_cost must be a non-negative integer")
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 0:
            raise ValueError("selection budget must be a non-negative integer")
        if self.total_cost > self.budget:
            raise ValueError("selection total_cost exceeds budget")
        if (
            isinstance(self.objective_units, bool)
            or not isinstance(self.objective_units, int)
            or self.objective_units < 0
        ):
            raise ValueError("selection objective_units must be a non-negative integer")
        if (
            isinstance(self.uplift_scale, bool)
            or not isinstance(self.uplift_scale, int)
            or self.uplift_scale <= 0
        ):
            raise ValueError("selection uplift_scale must be a positive integer")
        if float(self.objective_value) != self.objective_units / self.uplift_scale:
            raise ValueError("selection objective_value does not match exact units")

    @property
    def selected(self) -> tuple[str, ...]:
        """Compatibility alias for generic optimizer callers."""

        return self.selected_query_ids

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_query_ids": list(self.selected_query_ids),
            "selected": list(self.selected_query_ids),
            "total_cost": int(self.total_cost),
            "objective_value": float(self.objective_value),
            "budget": int(self.budget),
            "algorithm": self.algorithm,
            "objective_units": int(self.objective_units),
            "objective_units_decimal": str(self.objective_units),
            "uplift_scale": int(self.uplift_scale),
            "numeric_semantics": NUMERIC_SEMANTICS,
            "steps": [step.as_dict() for step in self.steps],
        }

    def to_dict(self) -> dict[str, object]:
        return self.as_dict()


@dataclass(frozen=True)
class _DensityEntry:
    """Exact max-heap priority represented without floating-point division."""

    gain_units: int
    cost: int
    query_id: str

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _DensityEntry):
            return NotImplemented
        left = self.gain_units * other.cost
        right = other.gain_units * self.cost
        if left != right:
            return left > right
        return self.query_id < other.query_id


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
    current_best = objective.empty_best_units()
    heap: list[_DensityEntry] = []
    for query_id in ordered:
        gain_units = objective._marginal_gain_units_from_validated_best(  # noqa: SLF001
            query_id, current_best
        )
        heapq.heappush(heap, _DensityEntry(gain_units, checked_costs[query_id], query_id))

    selected: list[str] = []
    selected_set: set[str] = set()
    steps: list[SelectionStep] = []
    total_cost = 0

    while heap:
        stale = heapq.heappop(heap)
        query_id = stale.query_id
        if query_id in selected_set:
            continue
        cost = checked_costs[query_id]
        if total_cost + cost > checked_budget:
            continue

        gain_units = objective._marginal_gain_units_from_validated_best(  # noqa: SLF001
            query_id, current_best
        )
        if gain_units == 0:
            continue
        refreshed = _DensityEntry(gain_units, cost, query_id)
        if heap and heap[0] < refreshed:
            heapq.heappush(heap, refreshed)
            continue

        selected.append(query_id)
        selected_set.add(query_id)
        total_cost += cost
        if total_cost > checked_budget:
            raise AssertionError("optimizer exceeded its validated budget")
        objective._update_best_units_in_place(current_best, query_id)  # noqa: SLF001
        gain = gain_units / objective.uplift_scale
        steps.append(
            SelectionStep(
                query_id=query_id,
                cost=cost,
                marginal_gain=gain,
                gain_per_cost=gain / cost,
                marginal_gain_units=gain_units,
                uplift_scale=objective.uplift_scale,
            )
        )

    objective_units = objective.value_units_from_best(current_best)
    greedy = SelectionResult(
        selected_query_ids=tuple(selected),
        total_cost=total_cost,
        objective_value=objective_units / objective.uplift_scale,
        budget=checked_budget,
        algorithm="lazy_group_gain_cost_greedy",
        steps=tuple(steps),
        objective_units=objective_units,
        uplift_scale=objective.uplift_scale,
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
            objective_units=best_action.objective_units,
            uplift_scale=best_action.uplift_scale,
        )
    return greedy


def eager_gain_cost_greedy(
    objective: GroupUpliftObjective,
    costs: Mapping[str, float],
    budget: float,
    *,
    candidates: Sequence[str] | None = None,
) -> SelectionResult:
    """Reference MGreedy that recomputes every exact density each round."""

    ordered, checked_costs, checked_budget = _validate_inputs(
        objective, costs, budget, candidates
    )
    remaining = set(ordered)
    current_best = objective.empty_best_units()
    selected: list[str] = []
    steps: list[SelectionStep] = []
    total_cost = 0

    while remaining:
        entries = [
            _DensityEntry(
                objective._marginal_gain_units_from_validated_best(  # noqa: SLF001
                    query_id, current_best
                ),
                checked_costs[query_id],
                query_id,
            )
            for query_id in remaining
        ]
        chosen = min(entries)
        remaining.remove(chosen.query_id)
        if chosen.gain_units == 0:
            break
        if total_cost + chosen.cost > checked_budget:
            continue

        selected.append(chosen.query_id)
        total_cost += chosen.cost
        objective._update_best_units_in_place(  # noqa: SLF001
            current_best, chosen.query_id
        )
        gain = chosen.gain_units / objective.uplift_scale
        steps.append(
            SelectionStep(
                query_id=chosen.query_id,
                cost=chosen.cost,
                marginal_gain=gain,
                gain_per_cost=gain / chosen.cost,
                marginal_gain_units=chosen.gain_units,
                uplift_scale=objective.uplift_scale,
            )
        )

    objective_units = objective.value_units_from_best(current_best)
    greedy = SelectionResult(
        selected_query_ids=tuple(selected),
        total_cost=total_cost,
        objective_value=objective_units / objective.uplift_scale,
        budget=checked_budget,
        algorithm="eager_group_gain_cost_greedy",
        steps=tuple(steps),
        objective_units=objective_units,
        uplift_scale=objective.uplift_scale,
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
            algorithm="eager_group_gain_cost_best_single_action",
            steps=best_action.steps,
            objective_units=best_action.objective_units,
            uplift_scale=best_action.uplift_scale,
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
    best = SelectionResult(
        (),
        0,
        0.0,
        checked_budget,
        "exhaustive_optimum",
        objective_units=0,
        uplift_scale=objective.uplift_scale,
    )
    for size in range(1, len(ordered) + 1):
        for selected in combinations(ordered, size):
            total_cost = sum(checked_costs[query_id] for query_id in selected)
            if total_cost > checked_budget:
                continue
            objective_units = objective.value_units(selected)
            candidate = SelectionResult(
                selected_query_ids=tuple(selected),
                total_cost=total_cost,
                objective_value=objective_units / objective.uplift_scale,
                budget=checked_budget,
                algorithm="exhaustive_optimum",
                objective_units=objective_units,
                uplift_scale=objective.uplift_scale,
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
        if cost > budget:
            continue
        value_units = objective.singleton_value_units(query_id)
        if value_units == 0:
            continue
        value = value_units / objective.uplift_scale
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
                    marginal_gain_units=value_units,
                    uplift_scale=objective.uplift_scale,
                ),
            ),
            objective_units=value_units,
            uplift_scale=objective.uplift_scale,
        )
        if best is None or _is_better(candidate, best):
            best = candidate
    return best


def _validate_inputs(
    objective: GroupUpliftObjective,
    costs: Mapping[str, float],
    budget: float,
    candidates: Sequence[str] | None,
) -> tuple[tuple[str, ...], dict[str, int], int]:
    checked_budget = _exact_integer(budget, label="budget", minimum=0)

    raw = objective.query_ids if candidates is None else tuple(str(value) for value in candidates)
    ordered = tuple(sorted(raw))
    if len(set(ordered)) != len(ordered):
        raise ValueError("candidates must be unique")
    unknown = sorted(set(ordered) - set(objective.query_ids))
    if unknown:
        raise ValueError(f"unknown candidate queries: {unknown}")

    checked_costs: dict[str, int] = {}
    for query_id in ordered:
        if query_id not in costs:
            raise ValueError(f"missing cost for query {query_id!r}")
        cost = _exact_integer(
            costs[query_id],
            label=f"cost for query {query_id!r}",
            minimum=1,
        )
        checked_costs[query_id] = cost
    return ordered, checked_costs, checked_budget


def _gain_per_cost(gain: float, cost: float) -> float:
    return float(gain / cost)


def _is_better(candidate: SelectionResult, incumbent: SelectionResult) -> bool:
    if candidate.uplift_scale != incumbent.uplift_scale:
        raise ValueError("cannot compare results with different uplift scales")
    if candidate.objective_units > incumbent.objective_units:
        return True
    if candidate.objective_units < incumbent.objective_units:
        return False
    if candidate.total_cost < incumbent.total_cost:
        return True
    if candidate.total_cost > incumbent.total_cost:
        return False
    return tuple(sorted(candidate.selected_query_ids)) < tuple(
        sorted(incumbent.selected_query_ids)
    )


def _exact_integer(value: object, *, label: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} must be an integer >= {minimum}") from error
    if not decimal.is_finite() or decimal != decimal.to_integral_value():
        raise ValueError(f"{label} must be an integer >= {minimum}")
    checked = int(decimal)
    if checked < minimum:
        adjective = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{label} must be a {adjective} integer")
    return checked


def _serializable_ratio(value: float) -> float | str:
    return float(value)
