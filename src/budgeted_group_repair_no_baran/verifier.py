"""Dirty-table-only validation and deterministic overlap arbitration."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import html
import math
import re
from typing import Mapping, Sequence


_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_DATE_RE = re.compile(
    r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$"
)
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?$", re.I)
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_URL_RE = re.compile(r"^(?:https?://|www\.)", re.I)


@dataclass(frozen=True)
class VerifierConfig:
    minimum_llm_confidence: float = 0.55
    minimum_net_gain: float = 0.0
    acceptance_score: float = 0.55
    require_comparative_signal: bool = True

    def __post_init__(self) -> None:
        for name in ("minimum_llm_confidence", "acceptance_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        gain = float(self.minimum_net_gain)
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("minimum_net_gain must be finite and non-negative")


@dataclass(frozen=True)
class VerificationDecision:
    accept_llm: bool
    final_prediction: str
    final_source: str
    score: float
    reason: str
    query_id: str | None
    llm_confidence: float
    conservative_uplift: float
    type_compatible: bool
    format_compatible: bool
    support_advantage: bool
    candidate_column_support: int
    fd_violations_before: int
    fd_violations_after: int

    @property
    def constraint_advantage(self) -> bool:
        return self.fd_violations_after < self.fd_violations_before

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["constraint_advantage"] = self.constraint_advantage
        return result

    def to_dict(self) -> dict[str, object]:
        return self.as_dict()


@dataclass(frozen=True)
class RankedRepairCandidate:
    """One selected query's proposed output for overlap arbitration."""

    query_id: str
    item: Mapping[str, object] | object
    conservative_uplift: float
    cost: float
    group_size: int

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query_id must be non-empty")
        for name in ("conservative_uplift", "cost"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if int(self.group_size) < 1:
            raise ValueError("group_size must be positive")
        object.__setattr__(self, "group_size", int(self.group_size))


@dataclass(frozen=True)
class ArbitrationResult:
    """Final per-cell result plus a complete deterministic attempt trace."""

    decision: VerificationDecision
    attempted_query_ids: tuple[str, ...]
    rejected_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            **self.decision.as_dict(),
            "attempted_query_ids": list(self.attempted_query_ids),
            "rejected_reasons": list(self.rejected_reasons),
        }


@dataclass(frozen=True)
class _PublicFD:
    rule_id: str
    determinant: tuple[str, ...]
    dependent: str


@dataclass(frozen=True)
class _ColumnProfile:
    counts: Counter[str]
    dominant_type: str
    dominant_format: str
    empty_share: float


class GroupRepairVerifier:
    """Validate proposed repairs without reference answers or annotations.

    Inputs are restricted to the dirty table, known target coordinates, public
    functional dependencies, the output schema, and pre-selection uplift.  FD
    evidence is the exact change in pair conflicts involving the candidate row
    before and after the proposed replacement.
    """

    def __init__(
        self,
        dirty_table: object,
        safe_cells: Sequence[Mapping[str, object] | object] = (),
        public_fds: Sequence[Mapping[str, object] | object] = (),
        config: VerifierConfig | None = None,
    ) -> None:
        self.config = config or VerifierConfig()
        self._rows, self._columns = _table_records(dirty_table)
        if not self._rows or not self._columns:
            raise ValueError("dirty_table must contain rows and columns")
        self._positions = {column: index for index, column in enumerate(self._columns)}
        self._error_coordinates = {
            (_cell_row(cell), _cell_column(cell, self._columns)) for cell in safe_cells
        }
        self._profiles = {
            column: self._build_profile(column) for column in self._columns
        }
        self._rules = tuple(_coerce_fd(rule, self._positions) for rule in public_fds)
        self._rules_by_column: dict[str, tuple[_PublicFD, ...]] = {
            column: tuple(
                rule
                for rule in self._rules
                if column == rule.dependent or column in rule.determinant
            )
            for column in self._columns
        }

    def verify(
        self,
        cell: Mapping[str, object] | object,
        baran_record: Mapping[str, object] | object,
        llm_item: Mapping[str, object] | object | None,
        conservative_uplift: float | Mapping[str, object] | object,
        *,
        query_id: str | None = None,
    ) -> VerificationDecision:
        row = _cell_row(cell)
        column = _cell_column(cell, self._columns)
        dirty_value = normalize_value(_field(cell, "dirty_value", self._rows[row][column]))
        fallback = _baran_fallback(baran_record, dirty_value)
        ell = _coerce_uplift(conservative_uplift)

        if llm_item is None:
            return _rejection(fallback, "invalid_llm_output", query_id, ell)
        parse_status = _field(llm_item, "parse_status", None)
        if parse_status is not None and str(parse_status) not in {"ok", "partial", "ok_item"}:
            return _rejection(fallback, "invalid_llm_output", query_id, ell)
        decision = str(_field(llm_item, "decision", "")).strip().lower()
        if decision != "propose":
            return _rejection(fallback, "fallback_decision", query_id, ell)
        raw_candidate = _field(
            llm_item, "repair", _field(llm_item, "prediction", None)
        )
        if raw_candidate is None:
            return _rejection(fallback, "missing_repair", query_id, ell)
        candidate = normalize_value(raw_candidate)
        confidence = _clamp(_field(llm_item, "confidence", 0.0))

        profile = self._profiles[column]
        candidate_type = infer_value_type(candidate)
        baran_type = infer_value_type(fallback)
        type_compatible = (
            candidate_type == profile.dominant_type
            or profile.dominant_type in {"text", "empty"}
        )
        baran_type_compatible = (
            baran_type == profile.dominant_type
            or profile.dominant_type in {"text", "empty"}
        )
        candidate_format = format_signature(candidate)
        baran_format = format_signature(fallback)
        format_compatible = (
            candidate_format == profile.dominant_format
            or profile.dominant_type == "text"
        )
        baran_format_compatible = (
            baran_format == profile.dominant_format
            or profile.dominant_type == "text"
        )
        candidate_support = int(profile.counts.get(candidate, 0))
        support_advantage = candidate_support > int(profile.counts.get(fallback, 0))
        fd_before, fd_after = self._fd_conflicts(row, column, candidate)
        fd_advantage = fd_after < fd_before

        if candidate == "":
            return _rejection_with_signals(
                fallback,
                "empty_repair",
                query_id,
                ell,
                confidence,
                type_compatible,
                format_compatible,
                support_advantage,
                candidate_support,
                fd_before,
                fd_after,
            )
        if candidate == dirty_value:
            return _rejection_with_signals(
                fallback,
                "unchanged_dirty_value",
                query_id,
                ell,
                confidence,
                type_compatible,
                format_compatible,
                support_advantage,
                candidate_support,
                fd_before,
                fd_after,
            )
        if candidate == fallback:
            return _rejection_with_signals(
                fallback,
                "equivalent_to_baran",
                query_id,
                ell,
                confidence,
                type_compatible,
                format_compatible,
                support_advantage,
                candidate_support,
                fd_before,
                fd_after,
            )
        if ell <= 0.0 or ell < self.config.minimum_net_gain:
            return _rejection_with_signals(
                fallback,
                "non_positive_predicted_gain",
                query_id,
                ell,
                confidence,
                type_compatible,
                format_compatible,
                support_advantage,
                candidate_support,
                fd_before,
                fd_after,
            )
        if confidence < self.config.minimum_llm_confidence:
            return _rejection_with_signals(
                fallback,
                "low_llm_confidence",
                query_id,
                ell,
                confidence,
                type_compatible,
                format_compatible,
                support_advantage,
                candidate_support,
                fd_before,
                fd_after,
            )
        if fd_after > fd_before:
            return _rejection_with_signals(
                fallback,
                "public_fd_worse",
                query_id,
                ell,
                confidence,
                type_compatible,
                format_compatible,
                support_advantage,
                candidate_support,
                fd_before,
                fd_after,
            )
        if not type_compatible and baran_type_compatible and not fd_advantage:
            return _rejection_with_signals(
                fallback,
                "type_worse_than_baran",
                query_id,
                ell,
                confidence,
                type_compatible,
                format_compatible,
                support_advantage,
                candidate_support,
                fd_before,
                fd_after,
            )
        if not format_compatible and baran_format_compatible and not fd_advantage:
            return _rejection_with_signals(
                fallback,
                "format_worse_than_baran",
                query_id,
                ell,
                confidence,
                type_compatible,
                format_compatible,
                support_advantage,
                candidate_support,
                fd_before,
                fd_after,
            )

        type_advantage = type_compatible and not baran_type_compatible
        format_advantage = format_compatible and not baran_format_compatible
        comparative_signal = (
            support_advantage
            or fd_advantage
            or type_advantage
            or format_advantage
        )
        if self.config.require_comparative_signal and not comparative_signal:
            return _rejection_with_signals(
                fallback,
                "no_comparative_signal",
                query_id,
                ell,
                confidence,
                type_compatible,
                format_compatible,
                support_advantage,
                candidate_support,
                fd_before,
                fd_after,
            )

        score = (
            0.40 * confidence
            + 0.25 * _clamp(ell / 0.5)
            + 0.10 * float(type_compatible)
            + 0.05 * float(format_compatible)
            + 0.10 * float(support_advantage)
            + 0.10 * float(fd_advantage)
        )
        accepted = score >= self.config.acceptance_score
        return VerificationDecision(
            accept_llm=accepted,
            final_prediction=candidate if accepted else fallback,
            final_source="llm" if accepted else "baran",
            score=float(score),
            reason="accepted" if accepted else "verification_score_below_threshold",
            query_id=query_id,
            llm_confidence=confidence,
            conservative_uplift=ell,
            type_compatible=type_compatible,
            format_compatible=format_compatible,
            support_advantage=support_advantage,
            candidate_column_support=candidate_support,
            fd_violations_before=fd_before,
            fd_violations_after=fd_after,
        )

    def arbitrate(
        self,
        cell: Mapping[str, object] | object,
        baran_record: Mapping[str, object] | object,
        candidates: Sequence[RankedRepairCandidate | Mapping[str, object]],
    ) -> ArbitrationResult:
        """Try candidates in fixed uplift/cost/size/query order."""

        normalized = tuple(_coerce_ranked_candidate(value) for value in candidates)
        ordered = sorted(
            normalized,
            key=lambda value: (
                -value.conservative_uplift,
                value.cost,
                value.group_size,
                value.query_id,
            ),
        )
        attempts: list[str] = []
        reasons: list[str] = []
        for candidate in ordered:
            attempts.append(candidate.query_id)
            decision = self.verify(
                cell,
                baran_record,
                candidate.item,
                candidate.conservative_uplift,
                query_id=candidate.query_id,
            )
            if decision.accept_llm:
                return ArbitrationResult(decision, tuple(attempts), tuple(reasons))
            reasons.append(decision.reason)

        dirty_value = normalize_value(
            _field(
                cell,
                "dirty_value",
                self._rows[_cell_row(cell)][_cell_column(cell, self._columns)],
            )
        )
        fallback = _baran_fallback(baran_record, dirty_value)
        decision = _rejection(
            fallback,
            "all_candidates_rejected" if ordered else "no_candidate",
            None,
            0.0,
        )
        return ArbitrationResult(decision, tuple(attempts), tuple(reasons))

    def _build_profile(self, column: str) -> _ColumnProfile:
        values = [
            normalize_value(row[column])
            for row_index, row in enumerate(self._rows)
            if (row_index, column) not in self._error_coordinates
        ]
        if not values:
            values = [normalize_value(row[column]) for row in self._rows]
        nonempty = [value for value in values if value]
        types = Counter(infer_value_type(value) for value in nonempty)
        formats = Counter(format_signature(value) for value in nonempty)
        return _ColumnProfile(
            counts=Counter(values),
            dominant_type=types.most_common(1)[0][0] if types else "empty",
            dominant_format=formats.most_common(1)[0][0] if formats else "EMPTY",
            empty_share=(len(values) - len(nonempty)) / len(values) if values else 0.0,
        )

    def _fd_conflicts(
        self, row_index: int, column: str, candidate: str
    ) -> tuple[int, int]:
        before = 0
        after = 0
        for rule in self._rules_by_column[column]:
            for peer_index in range(len(self._rows)):
                if peer_index == row_index:
                    continue
                before += int(
                    self._pair_conflicts(rule, row_index, peer_index, None, None)
                )
                after += int(
                    self._pair_conflicts(
                        rule, row_index, peer_index, column, candidate
                    )
                )
        return before, after

    def _pair_conflicts(
        self,
        rule: _PublicFD,
        row_index: int,
        peer_index: int,
        override_column: str | None,
        override_value: str | None,
    ) -> bool:
        def value(index: int, column: str) -> str:
            if index == row_index and column == override_column:
                return normalize_value(override_value)
            return normalize_value(self._rows[index][column])

        left_key = tuple(value(row_index, column) for column in rule.determinant)
        right_key = tuple(value(peer_index, column) for column in rule.determinant)
        left_dependent = value(row_index, rule.dependent)
        right_dependent = value(peer_index, rule.dependent)
        if (
            any(item == "" for item in left_key)
            or any(item == "" for item in right_key)
            or left_dependent == ""
            or right_dependent == ""
        ):
            return False
        return left_key == right_key and left_dependent != right_dependent


ComparativeVerifier = GroupRepairVerifier


def _coerce_ranked_candidate(
    value: RankedRepairCandidate | Mapping[str, object],
) -> RankedRepairCandidate:
    if isinstance(value, RankedRepairCandidate):
        return value
    return RankedRepairCandidate(
        query_id=str(value.get("query_id", "")),
        item=value.get("item", value),
        conservative_uplift=float(
            value.get("conservative_uplift", value.get("ell", 0.0))
        ),
        cost=float(value.get("cost", value.get("estimated_total_tokens", 0.0))),
        group_size=int(value.get("group_size", 1)),
    )


def _coerce_fd(rule: Mapping[str, object] | object, positions: Mapping[str, int]) -> _PublicFD:
    rule_id = str(_field(rule, "rule_id", ""))
    raw_determinant = _field(rule, "determinant", ())
    if isinstance(raw_determinant, (str, bytes)):
        raise TypeError("FD determinant must be a sequence of columns")
    determinant = tuple(str(value) for value in raw_determinant)  # type: ignore[arg-type]
    dependent = str(_field(rule, "dependent", ""))
    if not rule_id or not determinant or not dependent:
        raise ValueError("public FD requires rule_id, determinant, and dependent")
    if len(set(determinant)) != len(determinant) or dependent in determinant:
        raise ValueError("public FD must be non-trivial and use unique determinants")
    unknown = sorted(set((*determinant, dependent)) - set(positions))
    if unknown:
        raise ValueError(f"public FD {rule_id!r} refers to unknown columns: {unknown}")
    return _PublicFD(rule_id, determinant, dependent)


def _table_records(table: object) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    if hasattr(table, "to_dict") and hasattr(table, "columns"):
        columns = tuple(str(value) for value in list(table.columns))
        raw = table.to_dict(orient="records")
        return (
            [{str(key): value for key, value in row.items()} for row in raw],
            columns,
        )
    if isinstance(table, Mapping) or isinstance(table, (str, bytes)):
        raise TypeError("dirty_table must be a sequence of row mappings")
    rows = list(table)  # type: ignore[arg-type]
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("dirty_table must be a non-empty sequence of row mappings")
    columns = tuple(sorted({str(key) for row in rows for key in row.keys()}))
    records = [
        {column: row.get(column, "") for column in columns}  # type: ignore[union-attr]
        for row in rows
    ]
    return records, columns


def _cell_row(cell: Mapping[str, object] | object) -> int:
    row = int(_field(cell, "row", -1))
    if row < 0:
        raise ValueError("cell row must be non-negative")
    return row


def _cell_column(
    cell: Mapping[str, object] | object, columns: Sequence[str]
) -> str:
    column = str(_field(cell, "column", ""))
    if column:
        if column not in columns:
            raise ValueError(f"unknown cell column: {column}")
        return column
    col = int(_field(cell, "col", -1))
    if col < 0 or col >= len(columns):
        raise ValueError("cell must provide a valid column or col")
    return columns[col]


def _baran_fallback(record: Mapping[str, object] | object, dirty_value: str) -> str:
    status = str(_field(record, "parse_status", ""))
    prediction = _field(record, "prediction", None)
    if status.startswith("ok") and prediction is not None:
        return normalize_value(prediction)
    return dirty_value


def _coerce_uplift(value: float | Mapping[str, object] | object) -> float:
    if isinstance(value, Mapping) or not isinstance(value, (int, float)):
        raw = _field(
            value,
            "conservative_uplift",
            _field(value, "ell", _field(value, "net_gain", 0.0)),
        )
    else:
        raw = value
    try:
        result = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _rejection(
    fallback: str, reason: str, query_id: str | None, ell: float
) -> VerificationDecision:
    return VerificationDecision(
        False,
        fallback,
        "baran",
        0.0,
        reason,
        query_id,
        0.0,
        ell,
        False,
        False,
        False,
        0,
        0,
        0,
    )


def _rejection_with_signals(
    fallback: str,
    reason: str,
    query_id: str | None,
    ell: float,
    confidence: float,
    type_compatible: bool,
    format_compatible: bool,
    support_advantage: bool,
    candidate_support: int,
    fd_before: int,
    fd_after: int,
) -> VerificationDecision:
    return VerificationDecision(
        False,
        fallback,
        "baran",
        0.0,
        reason,
        query_id,
        confidence,
        ell,
        type_compatible,
        format_compatible,
        support_advantage,
        candidate_support,
        fd_before,
        fd_after,
    )


def normalize_value(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[\t\n ]+", " ", html.unescape(text)).strip("\t\n ")


def infer_value_type(value: object) -> str:
    text = normalize_value(value)
    if text == "":
        return "empty"
    if _NUMBER_RE.fullmatch(text):
        return "number"
    if _DATE_RE.fullmatch(text):
        return "date"
    if _TIME_RE.fullmatch(text):
        return "time"
    if _EMAIL_RE.fullmatch(text):
        return "email"
    if _URL_RE.search(text):
        return "url"
    if text.lower() in {"true", "false", "yes", "no", "y", "n"}:
        return "boolean"
    return "text"


def format_signature(value: object) -> str:
    text = normalize_value(value)
    if not text:
        return "EMPTY"
    mapped: list[str] = []
    previous = ""
    for char in text[:80]:
        current = (
            "A"
            if char.isalpha()
            else "9"
            if char.isdigit()
            else "_"
            if char.isspace()
            else char
        )
        if current != previous:
            mapped.append(current)
            previous = current
    return "".join(mapped)[:40]


def _clamp(value: object, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(number):
        return low
    return float(max(low, min(high, number)))


def _field(item: Mapping[str, object] | object, name: str, default: object) -> object:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)
