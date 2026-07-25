"""The non-negative max-over-query objective for Budgeted Group Repair."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PairGain:
    """One fixed conservative uplift for a cell-query incidence."""

    cell_id: str
    query_id: str
    conservative_uplift: float

    def __post_init__(self) -> None:
        if not self.cell_id or not self.query_id:
            raise ValueError("cell_id and query_id must be non-empty")
        value = float(self.conservative_uplift)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("conservative_uplift must be finite and non-negative")
        object.__setattr__(self, "conservative_uplift", value)

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "query_id": self.query_id,
            "conservative_uplift": self.conservative_uplift,
        }


GainInput = (
    Iterable[PairGain]
    | Mapping[tuple[str, str], float]
    | Mapping[str, Mapping[str, float]]
)


class GroupUpliftObjective:
    """Evaluate ``sum_i max(0, max_{q in S} ell[i,q])``.

    The constructor accepts any of these audit-friendly shapes:

    * an iterable of :class:`PairGain`;
    * ``{(cell_id, query_id): ell}``;
    * ``{query_id: {cell_id: ell}}``.

    There is no double counting: if several selected queries cover a cell,
    only that cell's largest fixed conservative uplift contributes.
    """

    def __init__(self, gains: GainInput) -> None:
        pairs = _materialize_pairs(gains)
        if not pairs:
            raise ValueError("gains must contain at least one cell-query pair")
        by_query: dict[str, dict[str, float]] = {}
        seen: set[tuple[str, str]] = set()
        for pair in pairs:
            key = (pair.cell_id, pair.query_id)
            if key in seen:
                raise ValueError(
                    f"duplicate gain for cell {pair.cell_id!r}, query {pair.query_id!r}"
                )
            seen.add(key)
            by_query.setdefault(pair.query_id, {})[pair.cell_id] = float(
                pair.conservative_uplift
            )

        self._by_query = {
            query_id: dict(sorted(cell_gains.items()))
            for query_id, cell_gains in sorted(by_query.items())
        }
        self._query_ids = tuple(self._by_query)
        self._cell_ids = tuple(
            sorted({cell_id for values in self._by_query.values() for cell_id in values})
        )

    @property
    def query_ids(self) -> tuple[str, ...]:
        return self._query_ids

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self._cell_ids

    def gains_for(self, query_id: str) -> dict[str, float]:
        """Return a copy of the conservative uplift vector for one query."""

        self._require_query(query_id)
        return dict(self._by_query[query_id])

    def empty_best(self) -> dict[str, float]:
        """Return the all-zero per-cell state for an empty selection."""

        return {cell_id: 0.0 for cell_id in self._cell_ids}

    def empty_state(self) -> dict[str, float]:
        """Compatibility alias used by optimizers."""

        return self.empty_best()

    def best_for(self, selected: Iterable[str]) -> dict[str, float]:
        current = self.empty_best()
        seen: set[str] = set()
        for query_id in selected:
            query = str(query_id)
            self._require_query(query)
            if query in seen:
                continue
            seen.add(query)
            current = self.updated_best(current, query)
        return current

    def value(self, selected: Iterable[str]) -> float:
        return self.value_from_best(self.best_for(selected))

    def value_from_best(self, current_best: Mapping[str, float]) -> float:
        self._validate_best(current_best)
        return float(sum(float(value) for value in current_best.values()))

    def marginal_gain(self, query_id: str, selected: Iterable[str]) -> float:
        selected_ids = tuple(str(value) for value in selected)
        if query_id in selected_ids:
            return 0.0
        return self.marginal_gain_from_best(query_id, self.best_for(selected_ids))

    def marginal_gain_from_best(
        self,
        query_id: str,
        current_best: Mapping[str, float],
    ) -> float:
        """Compute the exact overlap-aware marginal value in ``O(|G_q|)``."""

        self._require_query(query_id)
        self._validate_best(current_best)
        return float(
            sum(
                max(0.0, ell - float(current_best[cell_id]))
                for cell_id, ell in self._by_query[query_id].items()
            )
        )

    def marginal_gain_from_state(
        self, query_id: str, current_best: Mapping[str, float]
    ) -> float:
        return self.marginal_gain_from_best(query_id, current_best)

    def updated_best(
        self,
        current_best: Mapping[str, float],
        query_id: str,
    ) -> dict[str, float]:
        """Return a new state after adding ``query_id``; never mutate input."""

        self._require_query(query_id)
        self._validate_best(current_best)
        updated = {cell_id: float(value) for cell_id, value in current_best.items()}
        for cell_id, ell in self._by_query[query_id].items():
            updated[cell_id] = max(updated[cell_id], ell)
        return updated

    def updated_state(
        self, current_best: Mapping[str, float], query_id: str
    ) -> dict[str, float]:
        return self.updated_best(current_best, query_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "query_ids": list(self._query_ids),
            "cell_ids": list(self._cell_ids),
            "gains": {
                query_id: {
                    cell_id: float(value)
                    for cell_id, value in self._by_query[query_id].items()
                }
                for query_id in self._query_ids
            },
        }

    def _require_query(self, query_id: str) -> None:
        if query_id not in self._by_query:
            raise KeyError(f"unknown query_id: {query_id}")

    def _validate_best(self, current_best: Mapping[str, float]) -> None:
        if set(current_best) != set(self._cell_ids):
            raise ValueError("current_best keys do not match objective cells")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in current_best.values()
        ):
            raise ValueError("current_best values must be finite and non-negative")


def _materialize_pairs(gains: GainInput) -> list[PairGain]:
    if isinstance(gains, Mapping):
        pairs: list[PairGain] = []
        for raw_key, raw_value in gains.items():
            if isinstance(raw_key, tuple):
                if len(raw_key) != 2 or isinstance(raw_value, Mapping):
                    raise TypeError(
                        "tuple-key gains must use {(cell_id, query_id): number}"
                    )
                pairs.append(
                    PairGain(str(raw_key[0]), str(raw_key[1]), float(raw_value))
                )
                continue
            if not isinstance(raw_value, Mapping):
                raise TypeError(
                    "nested gains must use {query_id: {cell_id: number}}"
                )
            query_id = str(raw_key)
            for cell_id, value in raw_value.items():
                pairs.append(PairGain(str(cell_id), query_id, float(value)))
        return pairs

    materialized = list(gains)
    if not all(isinstance(pair, PairGain) for pair in materialized):
        raise TypeError("iterable gains must contain PairGain records")
    return list(materialized)


def build_group_objective(
    records: Sequence[Mapping[str, object]],
    *,
    cell_field: str = "cell_id",
    query_field: str = "query_id",
    gain_field: str = "conservative_uplift",
) -> GroupUpliftObjective:
    """Build the objective from CSV/JSON-like prediction records."""

    pairs: list[PairGain] = []
    for index, record in enumerate(records):
        if cell_field not in record or query_field not in record or gain_field not in record:
            raise ValueError(f"row {index} is missing a required objective field")
        pairs.append(
            PairGain(
                cell_id=str(record[cell_field]),
                query_id=str(record[query_field]),
                conservative_uplift=float(record[gain_field]),
            )
        )
    return GroupUpliftObjective(pairs)
