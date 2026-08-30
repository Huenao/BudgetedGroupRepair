"""The non-negative max-over-query objective for Budgeted Group Repair."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence


DEFAULT_UPLIFT_SCALE = 10**15


def _validate_uplift_scale(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("uplift scale must be a positive integer")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("uplift scale must be a positive integer") from error
    if (
        not decimal.is_finite()
        or decimal != decimal.to_integral_value()
        or decimal <= 0
    ):
        raise ValueError("uplift scale must be a positive integer")
    return int(decimal)


def quantize_uplift(
    value: float,
    *,
    scale: int = DEFAULT_UPLIFT_SCALE,
) -> int:
    """Map one frozen score to the exact fixed-point theorem objective.

    Conversion starts from Python's canonical decimal rendering instead of
    multiplying a binary float.  Round-half-even and ``scale`` are therefore
    deterministic parts of the optimization problem, not numerical tolerances.
    """

    checked_scale = _validate_uplift_scale(scale)
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError("conservative_uplift must be finite and non-negative")
    # Decimal construction and integer arithmetic are independent of the
    # process-global Decimal context.  Implement round-half-even on the exact
    # rational value instead of relying on context-limited multiplication.
    numerator, denominator = Decimal(str(number)).as_integer_ratio()
    quotient, remainder = divmod(numerator * checked_scale, denominator)
    doubled = 2 * remainder
    if doubled > denominator or (doubled == denominator and quotient % 2 == 1):
        quotient += 1
    return quotient


@dataclass(frozen=True)
class PairGain:
    """One fixed uncertainty-penalized routing score for an incidence.

    ``conservative_uplift`` is retained as the serialized field name for
    compatibility with frozen artifacts; it is not a confidence lower bound.
    """

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
    only that cell's largest fixed routing score contributes.
    """

    def __init__(
        self,
        gains: GainInput,
        *,
        uplift_scale: int = DEFAULT_UPLIFT_SCALE,
    ) -> None:
        pairs = _materialize_pairs(gains)
        if not pairs:
            raise ValueError("gains must contain at least one cell-query pair")
        checked_scale = _validate_uplift_scale(uplift_scale)
        by_query: dict[str, dict[str, float]] = {}
        by_query_units: dict[str, dict[str, int]] = {}
        seen: set[tuple[str, str]] = set()
        for pair in pairs:
            key = (pair.cell_id, pair.query_id)
            if key in seen:
                raise ValueError(
                    f"duplicate gain for cell {pair.cell_id!r}, query {pair.query_id!r}"
                )
            seen.add(key)
            value = float(pair.conservative_uplift)
            by_query.setdefault(pair.query_id, {})[pair.cell_id] = value
            by_query_units.setdefault(pair.query_id, {})[pair.cell_id] = quantize_uplift(
                value,
                scale=checked_scale,
            )

        self._uplift_scale = checked_scale
        self._by_query = {
            query_id: dict(sorted(cell_gains.items()))
            for query_id, cell_gains in sorted(by_query.items())
        }
        self._by_query_units = {
            query_id: dict(sorted(cell_gains.items()))
            for query_id, cell_gains in sorted(by_query_units.items())
        }
        self._query_ids = tuple(self._by_query)
        self._cell_ids = tuple(
            sorted({cell_id for values in self._by_query.values() for cell_id in values})
        )
        canonical_units = [
            (query_id, cell_id, value)
            for query_id in self._query_ids
            for cell_id, value in self._by_query_units[query_id].items()
        ]
        self._quantized_gains_sha256 = hashlib.sha256(
            json.dumps(
                canonical_units,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @property
    def query_ids(self) -> tuple[str, ...]:
        return self._query_ids

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self._cell_ids

    @property
    def uplift_scale(self) -> int:
        """Fixed-point scale defining the exact surrogate objective."""

        return self._uplift_scale

    @property
    def quantized_gains_sha256(self) -> str:
        """Digest that binds the complete pair-level integer objective."""

        return self._quantized_gains_sha256

    def gains_for(self, query_id: str) -> dict[str, float]:
        """Return a copy of the routing-score vector for one query."""

        self._require_query(query_id)
        return dict(self._by_query[query_id])

    def gain_units_for(self, query_id: str) -> dict[str, int]:
        """Return exact non-negative fixed-point gains for one query."""

        self._require_query(query_id)
        return dict(self._by_query_units[query_id])

    def empty_best(self) -> dict[str, float]:
        """Return the all-zero per-cell state for an empty selection."""

        return {cell_id: 0.0 for cell_id in self._cell_ids}

    def empty_best_units(self) -> dict[str, int]:
        """Return the exact all-zero fixed-point state."""

        return {cell_id: 0 for cell_id in self._cell_ids}

    def empty_state(self) -> dict[str, float]:
        """Compatibility alias used by optimizers."""

        return self.empty_best()

    def best_for(self, selected: Iterable[str]) -> dict[str, float]:
        return {
            cell_id: units / self._uplift_scale
            for cell_id, units in self.best_units_for(selected).items()
        }

    def best_units_for(self, selected: Iterable[str]) -> dict[str, int]:
        """Return the exact per-cell fixed-point maximum for ``selected``."""

        current = self.empty_best_units()
        seen: set[str] = set()
        for query_id in selected:
            query = str(query_id)
            self._require_query(query)
            if query in seen:
                continue
            seen.add(query)
            current = self.updated_best_units(current, query)
        return current

    def value(self, selected: Iterable[str]) -> float:
        return self.value_units(selected) / self._uplift_scale

    def value_units(self, selected: Iterable[str]) -> int:
        """Evaluate the exact integer objective for ``selected``."""

        return self.value_units_from_best(self.best_units_for(selected))

    def singleton_value_units(self, query_id: str) -> int:
        """Return the exact value of one action without materializing a state."""

        self._require_query(query_id)
        return sum(self._by_query_units[query_id].values())

    def raw_value(self, selected: Iterable[str]) -> float:
        """Evaluate the unquantized float score for drift diagnostics only."""

        best = {cell_id: 0.0 for cell_id in self._cell_ids}
        seen: set[str] = set()
        for raw_query_id in selected:
            query_id = str(raw_query_id)
            self._require_query(query_id)
            if query_id in seen:
                continue
            seen.add(query_id)
            for cell_id, value in self._by_query[query_id].items():
                best[cell_id] = max(best[cell_id], value)
        return float(sum(best.values()))

    def value_from_best(self, current_best: Mapping[str, float]) -> float:
        self._validate_best(current_best)
        units = {
            cell_id: quantize_uplift(value, scale=self._uplift_scale)
            for cell_id, value in current_best.items()
        }
        return self.value_units_from_best(units) / self._uplift_scale

    def value_units_from_best(self, current_best: Mapping[str, int]) -> int:
        self._validate_best_units(current_best)
        return sum(int(value) for value in current_best.values())

    def marginal_gain(self, query_id: str, selected: Iterable[str]) -> float:
        return self.marginal_gain_units(query_id, selected) / self._uplift_scale

    def marginal_gain_units(self, query_id: str, selected: Iterable[str]) -> int:
        selected_ids = tuple(str(value) for value in selected)
        if query_id in selected_ids:
            return 0
        return self.marginal_gain_units_from_best(
            query_id,
            self.best_units_for(selected_ids),
        )

    def marginal_gain_from_best(
        self,
        query_id: str,
        current_best: Mapping[str, float],
    ) -> float:
        """Compute the exact overlap-aware marginal value in ``O(|G_q|)``."""

        self._validate_best(current_best)
        current_units = {
            cell_id: quantize_uplift(value, scale=self._uplift_scale)
            for cell_id, value in current_best.items()
        }
        return (
            self.marginal_gain_units_from_best(query_id, current_units)
            / self._uplift_scale
        )

    def marginal_gain_units_from_best(
        self,
        query_id: str,
        current_best: Mapping[str, int],
    ) -> int:
        """Compute the exact fixed-point marginal value in ``O(|G_q|)``."""

        self._require_query(query_id)
        self._validate_best_units(current_best)
        return self._marginal_gain_units_from_validated_best(query_id, current_best)

    def _marginal_gain_units_from_validated_best(
        self,
        query_id: str,
        current_best: Mapping[str, int],
    ) -> int:
        """Internal hot path for a state previously created by this objective."""

        return sum(
            max(0, ell - int(current_best[cell_id]))
            for cell_id, ell in self._by_query_units[query_id].items()
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

        self._validate_best(current_best)
        current_units = {
            cell_id: quantize_uplift(value, scale=self._uplift_scale)
            for cell_id, value in current_best.items()
        }
        return {
            cell_id: units / self._uplift_scale
            for cell_id, units in self.updated_best_units(
                current_units,
                query_id,
            ).items()
        }

    def updated_best_units(
        self,
        current_best: Mapping[str, int],
        query_id: str,
    ) -> dict[str, int]:
        """Return the exact state after adding ``query_id``."""

        self._require_query(query_id)
        self._validate_best_units(current_best)
        return self._updated_best_units_from_validated_best(current_best, query_id)

    def _updated_best_units_from_validated_best(
        self,
        current_best: Mapping[str, int],
        query_id: str,
    ) -> dict[str, int]:
        """Internal update for a state previously created by this objective."""

        updated = {cell_id: int(value) for cell_id, value in current_best.items()}
        for cell_id, ell in self._by_query_units[query_id].items():
            updated[cell_id] = max(updated[cell_id], ell)
        return updated

    def _update_best_units_in_place(
        self,
        current_best: dict[str, int],
        query_id: str,
    ) -> None:
        """Internal O(|G_q|) update for an optimizer-owned exact state."""

        for cell_id, ell in self._by_query_units[query_id].items():
            if ell > current_best[cell_id]:
                current_best[cell_id] = ell

    def updated_state(
        self, current_best: Mapping[str, float], query_id: str
    ) -> dict[str, float]:
        return self.updated_best(current_best, query_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "query_ids": list(self._query_ids),
            "cell_ids": list(self._cell_ids),
            "uplift_scale": self._uplift_scale,
            "quantization": "canonical_decimal_round_half_even",
            "quantized_gains_sha256": self._quantized_gains_sha256,
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

    def _validate_best_units(self, current_best: Mapping[str, int]) -> None:
        if set(current_best) != set(self._cell_ids):
            raise ValueError("current_best keys do not match objective cells")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in current_best.values()
        ):
            raise ValueError("current_best units must be non-negative integers")


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
