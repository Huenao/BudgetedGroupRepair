"""Leakage-safe cell signatures used by the BGR group generators.

Only :class:`~budgeted_group_repair.data.SafeCell`, the dirty table, and a
small allow-list of fresh Baran diagnostics are consumed here.  Sparse Python
vectors keep the implementation deterministic and avoid fitting a vocabulary
outside the current dataset.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .data import SafeCell, normalize_value


_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_DATE_RE = re.compile(r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?$", re.I)
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_URL_RE = re.compile(r"^(?:https?://|www\.)", re.I)

# These are deployment-time Baran outputs.  Values outside this allow-list are
# intentionally never copied into a feature or prompt.
_BARAN_PREDICTION_KEYS = ("prediction", "repair", "baran_prediction")
_BARAN_NUMERIC_KEYS = (
    "candidate_count",
    "top_candidate_support",
    "top_support",
    "second_candidate_support",
    "candidate_margin",
    "support_margin",
    "source_agreement",
    "source_group_agreement",
    "source_vote_count",
    "predicted_support",
)
_BARAN_VECTOR_KEYS = ("corrector_support", "corrector_support_vector", "source_support")

SparseVector = dict[str, float]


def infer_value_type(value: object) -> str:
    """Return a stable coarse type derived solely from the supplied value."""

    text = normalize_value(value)
    if not text:
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


def format_signature(value: object, *, max_chars: int = 80) -> str:
    """Compress letters, digits and whitespace into a bounded format shape."""

    text = normalize_value(value)
    if not text:
        return "EMPTY"
    mapped: list[str] = []
    previous = ""
    for char in text[:max_chars]:
        token = "A" if char.isalpha() else "9" if char.isdigit() else "_" if char.isspace() else char
        if token != previous:
            mapped.append(token)
            previous = token
    return "".join(mapped)[:40]


def _normalise_sparse(vector: Mapping[str, float]) -> SparseVector:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector.values()))
    if norm <= 0.0:
        return {}
    return {str(key): float(value) / norm for key, value in sorted(vector.items()) if float(value) != 0.0}


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Cosine similarity for already-normalised or arbitrary sparse vectors."""

    if not left or not right:
        return 1.0 if not left and not right else 0.0
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    dot = sum(float(value) * float(larger.get(key, 0.0)) for key, value in smaller.items())
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def cosine_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return max(0.0, min(2.0, 1.0 - cosine_similarity(left, right)))


def _as_baran_map(records: Any) -> dict[str, Mapping[str, Any]]:
    if records is None:
        return {}
    if isinstance(records, Mapping):
        return {
            str(identifier): record
            for identifier, record in records.items()
            if isinstance(record, Mapping)
        }
    mapped: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        identifier = record.get("cell_id")
        if identifier is not None and str(identifier):
            mapped[str(identifier)] = record
    return mapped


def _baran_prediction(record: Mapping[str, Any], fallback: str) -> str:
    for key in _BARAN_PREDICTION_KEYS:
        if key in record:
            return normalize_value(record[key])
    return fallback


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _baran_support(record: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    values: dict[str, float] = {}
    for key in _BARAN_NUMERIC_KEYS:
        number = _finite_number(record.get(key))
        if number is not None:
            values[key] = number
    for key in _BARAN_VECTOR_KEYS:
        raw = record.get(key)
        if isinstance(raw, Mapping):
            for source, value in sorted(raw.items(), key=lambda item: str(item[0])):
                number = _finite_number(value)
                if number is not None:
                    values[f"{key}:{source}"] = number
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for index, value in enumerate(raw):
                number = _finite_number(value)
                if number is not None:
                    values[f"{key}:{index}"] = number
    return tuple(sorted(values.items()))


def _frame_value(frame: Any, row: int, col: int) -> str:
    return normalize_value(frame.iloc[int(row), int(col)])


def _columns(frame: Any) -> tuple[str, ...]:
    return tuple(str(column) for column in frame.columns)


def _masked_row_text(
    frame: Any,
    cell: SafeCell,
    known_error_coordinates: set[tuple[int, int]],
    *,
    max_value_chars: int,
) -> str:
    fields: list[str] = []
    for col, column in enumerate(_columns(frame)):
        if (int(cell.row), col) in known_error_coordinates:
            value = "<MASKED_ERROR>"
        else:
            value = _frame_value(frame, int(cell.row), col)[:max_value_chars]
        fields.append(f"{column}={value}")
    return " | ".join(fields)


@dataclass(frozen=True)
class CellFeatures:
    """One immutable, deployment-safe cell signature."""

    cell_id: str
    suite: str
    dataset: str
    row: int
    col: int
    column: str
    dirty_type: str
    dirty_format: str
    baran_prediction: str
    baran_type: str
    baran_format: str
    baran_support: tuple[tuple[str, float], ...]
    masked_row_text: str

    @property
    def signature_tokens(self) -> tuple[str, ...]:
        support_tokens = tuple(
            f"support:{name}:{_support_bucket(value)}" for name, value in self.baran_support
        )
        return (
            f"column:{self.column}",
            f"dirty_type:{self.dirty_type}",
            f"dirty_format:{self.dirty_format}",
            f"baran_type:{self.baran_type}",
            f"baran_format:{self.baran_format}",
            *support_tokens,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "suite": self.suite,
            "dataset": self.dataset,
            "row": self.row,
            "col": self.col,
            "column": self.column,
            "dirty_type": self.dirty_type,
            "dirty_format": self.dirty_format,
            "baran_prediction": self.baran_prediction,
            "baran_type": self.baran_type,
            "baran_format": self.baran_format,
            "baran_support": dict(self.baran_support),
            "masked_row_text": self.masked_row_text,
        }


def _support_bucket(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude == 0:
        return "zero"
    if magnitude < 0.25:
        return "low"
    if magnitude < 0.75:
        return "medium"
    if magnitude <= 1.0:
        return "high"
    return f"count_{min(8, int(math.log2(magnitude + 1)))}"


def build_cell_features(
    dataset: Any,
    cells: Sequence[SafeCell],
    baran_by_cell: Any = None,
    *,
    max_value_chars: int = 160,
) -> tuple[CellFeatures, ...]:
    """Build signatures from a safe dataset view and fresh Baran records."""

    frame = getattr(dataset, "dirty", None)
    if frame is None:
        raise TypeError("dataset must expose a dirty dataframe")
    ordered = sorted(cells, key=lambda cell: cell.cell_id)
    if len({cell.cell_id for cell in ordered}) != len(ordered):
        raise ValueError("cells must have unique cell_id values")
    dataset_identity = {
        (str(cell.suite), str(cell.dataset)) for cell in ordered
    }
    if len(dataset_identity) > 1:
        raise ValueError("all cells must belong to one dataset")
    known = {(int(cell.row), int(cell.col)) for cell in ordered}
    baran = _as_baran_map(baran_by_cell)
    features: list[CellFeatures] = []
    for cell in ordered:
        record = baran.get(cell.cell_id, {})
        dirty = normalize_value(cell.dirty_value)
        prediction = _baran_prediction(record, dirty)
        features.append(
            CellFeatures(
                cell_id=cell.cell_id,
                suite=str(cell.suite),
                dataset=str(cell.dataset),
                row=int(cell.row),
                col=int(cell.col),
                column=str(cell.column),
                dirty_type=infer_value_type(dirty),
                dirty_format=format_signature(dirty),
                baran_prediction=prediction,
                baran_type=infer_value_type(prediction),
                baran_format=format_signature(prediction),
                baran_support=_baran_support(record),
                masked_row_text=_masked_row_text(
                    frame,
                    cell,
                    known,
                    max_value_chars=max(16, int(max_value_chars)),
                ),
            )
        )
    return tuple(features)


def pattern_vectors(features: Sequence[CellFeatures]) -> dict[str, SparseVector]:
    """Type/format and Baran-support vectors for pattern grouping."""

    result: dict[str, SparseVector] = {}
    for feature in features:
        vector: SparseVector = {
            f"dirty_type={feature.dirty_type}": 1.0,
            f"dirty_format={feature.dirty_format}": 1.0,
            f"baran_type={feature.baran_type}": 1.0,
            f"baran_format={feature.baran_format}": 1.0,
        }
        for name, value in feature.baran_support:
            signed = math.copysign(math.log1p(abs(float(value))), float(value))
            vector[f"support={name}"] = signed
        result[feature.cell_id] = _normalise_sparse(vector)
    return result


def _char_terms(text: str, ngram_range: tuple[int, int]) -> Counter[str]:
    compact = " ".join(normalize_value(text).lower().split())
    counts: Counter[str] = Counter()
    lower, upper = ngram_range
    for size in range(max(1, int(lower)), max(1, int(upper)) + 1):
        if len(compact) < size:
            continue
        counts.update(f"char{size}:{compact[index:index + size]}" for index in range(len(compact) - size + 1))
    return counts


def semantic_vectors(
    features: Sequence[CellFeatures],
    *,
    ngram_range: tuple[int, int] = (2, 5),
    signature_weight: float = 1.5,
) -> dict[str, SparseVector]:
    """Masked-row character TF-IDF enriched with the repair signature."""

    ordered = sorted(features, key=lambda feature: feature.cell_id)
    term_counts = [_char_terms(feature.masked_row_text, ngram_range) for feature in ordered]
    document_frequency: Counter[str] = Counter()
    for counts in term_counts:
        document_frequency.update(counts.keys())
    document_count = max(1, len(ordered))
    result: dict[str, SparseVector] = {}
    for feature, counts in zip(ordered, term_counts):
        vector: SparseVector = {}
        for term, count in counts.items():
            tf = 1.0 + math.log(float(count))
            idf = math.log((1.0 + document_count) / (1.0 + document_frequency[term])) + 1.0
            vector[term] = tf * idf
        for token in feature.signature_tokens:
            vector[f"signature:{token}"] = float(signature_weight)
        result[feature.cell_id] = _normalise_sparse(vector)
    return result


def merge_vectors(*vectors: Mapping[str, float], weights: Iterable[float] | None = None) -> SparseVector:
    """Combine sparse views without allowing dimensional name collisions."""

    weight_values = list(weights) if weights is not None else [1.0] * len(vectors)
    if len(weight_values) != len(vectors):
        raise ValueError("weights must match the number of vectors")
    combined: SparseVector = {}
    for view_index, (vector, weight) in enumerate(zip(vectors, weight_values)):
        for key, value in vector.items():
            combined[f"v{view_index}:{key}"] = float(weight) * float(value)
    return _normalise_sparse(combined)


__all__ = [
    "CellFeatures",
    "SparseVector",
    "build_cell_features",
    "cosine_distance",
    "cosine_similarity",
    "format_signature",
    "infer_value_type",
    "merge_vectors",
    "pattern_vectors",
    "semantic_vectors",
]
