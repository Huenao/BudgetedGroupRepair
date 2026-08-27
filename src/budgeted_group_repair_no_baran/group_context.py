"""Canonical dirty-evidence-only group prompt construction."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .data import SafeCell, normalize_value
from .prompt_policy import (
    INFORMATION_POLICY,
    PROMPT_SCHEMA_VERSION,
    assert_messages_safe,
    assert_payload_safe,
)


MASKED_TARGET = "<TARGET_ERROR_MASKED>"
MASKED_OTHER = "<OTHER_DETECTED_ERROR_MASKED>"
_CJK_RE = re.compile(
    "["
    "\u3400-\u4dbf"
    "\u4e00-\u9fff"
    "\uf900-\ufaff"
    "\u3040-\u30ff"
    "\uac00-\ud7af"
    "]"
)

CanonicalMessages = tuple[Mapping[str, str], ...]

SYSTEM_PROMPT = """You are a conservative data-repair candidate generator.
Repair every supplied target independently using only the supplied dirty-table evidence and listed public functional dependencies. Propose a value only when the evidence is sufficient; otherwise abstain.
Return exactly one JSON object and no prose:
{
  "query_id": "copy the supplied query_id exactly",
  "repairs": [
    {
      "cell_id": "copy one supplied cell_id exactly",
      "repair": "string candidate value",
      "confidence": 0.0,
      "decision": "propose | abstain",
      "evidence": "one short clause grounded in supplied evidence",
      "affected_constraints": ["supplied public FD identifiers"]
    }
  ]
}
Return at most one item for each supplied cell_id. Never invent a cell_id or an FD identifier."""


def _as_record_map(records: Any) -> dict[str, Mapping[str, Any]]:
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
        if isinstance(record, Mapping) and record.get("cell_id") is not None:
            mapped[str(record["cell_id"])] = record
    return mapped


def _bounded_text(value: Any, max_chars: int) -> str:
    text = normalize_value(value)
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)] + "…"


def _json_safe_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else None


def canonical_messages(messages: Sequence[Mapping[str, Any]]) -> CanonicalMessages:
    """Deep-freeze chat messages after validating their request shape."""

    canonical: list[Mapping[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported chat role: {role!r}")
        canonical.append(
            MappingProxyType({"role": role, "content": str(message.get("content", ""))})
        )
    if not canonical:
        raise ValueError("messages must not be empty")
    return tuple(canonical)


def messages_as_dicts(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]


def completion_token_ceiling(group_size: int) -> int:
    size = int(group_size)
    if size <= 0:
        raise ValueError("group_size must be positive")
    return 192 if size == 1 else 64 + 192 * size


def estimate_prompt_tokens(messages: Sequence[Mapping[str, str]]) -> int:
    """Conservative tokenizer-free estimate for mixed Latin/CJK JSON."""

    encoded = json.dumps(
        messages_as_dicts(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    cjk_count = len(_CJK_RE.findall(encoded))
    non_cjk = _CJK_RE.sub("", encoded)
    return max(1, cjk_count + math.ceil(len(non_cjk.encode("utf-8")) / 4))


def compute_query_id(
    suite: str,
    dataset: str,
    group_view: str,
    cell_ids: Sequence[str],
    *,
    arm: str = "structured",
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
    information_policy: str = INFORMATION_POLICY,
) -> str:
    ordered = tuple(sorted(str(identifier) for identifier in cell_ids))
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("query cell_ids must be non-empty and unique")
    payload = {
        "suite": str(suite),
        "dataset": str(dataset),
        "arm": str(arm),
        "group_view": str(group_view),
        "cell_ids": ordered,
        "prompt_schema_version": str(prompt_schema_version),
        "prompt_information_policy": str(information_policy),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"bgrq_{digest}"


def compute_ordered_query_id(
    suite: str,
    dataset: str,
    ordered_cell_ids: Sequence[str | SafeCell],
    *,
    group_view: str = "matched_multi_target",
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
    information_policy: str = INFORMATION_POLICY,
) -> str:
    """Compute an order-sensitive identity for a neutral evidence prompt.

    This identity intentionally excludes experimental provenance such as arm
    and source view.  Those fields belong in checkpoint metadata rather than
    in the primary structured/random prompt, so byte-identical physical
    requests can be deduplicated safely.
    """

    if str(group_view) != "matched_multi_target":
        raise ValueError("ordered evidence prompts require group_view='matched_multi_target'")
    ordered = tuple(
        str(value.cell_id) if isinstance(value, SafeCell) else str(value)
        for value in ordered_cell_ids
    )
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("ordered_cell_ids must be non-empty and unique")
    payload = {
        "suite": str(suite),
        "dataset": str(dataset),
        "group_view": str(group_view),
        "ordered_cell_ids": ordered,
        "prompt_schema_version": str(prompt_schema_version),
        "prompt_information_policy": str(information_policy),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"bgrq_{digest}"


def compute_prompt_hash(
    messages: Sequence[Mapping[str, str]],
    max_tokens: int,
    *,
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
    information_policy: str = INFORMATION_POLICY,
) -> str:
    """Hash every action-level request field except provider/model settings."""

    payload = {
        "prompt_schema_version": str(prompt_schema_version),
        "prompt_information_policy": str(information_policy),
        "messages": messages_as_dicts(messages),
        "max_tokens": int(max_tokens),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PromptMaterial:
    messages: CanonicalMessages
    prompt_hash: str
    estimated_prompt_tokens: int
    completion_token_ceiling: int
    estimated_total_tokens: int


class GroupContextBuilder:
    """Build fixed group prompts from a safe dataset projection.

    The builder deliberately accesses only ``dataset.dirty`` plus the public
    identity fields on ``SafeCell``.  All detected cells are masked whenever a
    row is exposed to the model.
    """

    def __init__(
        self,
        dataset: Any,
        cells: Sequence[SafeCell],
        external_features_by_cell: Any = None,
        *,
        known_error_cells: Sequence[SafeCell] | None = None,
        fd_components: Sequence[Any] = (),
        similar_row_count: int = 3,
        column_top_values: int = 6,
        max_value_chars: int = 160,
    ) -> None:
        self.dataset = dataset
        self.dirty = getattr(dataset, "dirty", None)
        if self.dirty is None:
            raise TypeError("dataset must expose a dirty dataframe")
        self.cells = tuple(sorted(cells, key=lambda cell: cell.cell_id))
        self.cell_by_id = {cell.cell_id: cell for cell in self.cells}
        if len(self.cell_by_id) != len(self.cells):
            raise ValueError("cells must have unique cell_id values")
        # External features may be used by the group generator, but this
        # context builder never reads or serializes them.
        del external_features_by_cell
        self.known_error_cells = tuple(
            sorted(
                known_error_cells if known_error_cells is not None else self.cells,
                key=lambda cell: cell.cell_id,
            )
        )
        self.fd_components = tuple(fd_components)
        self.similar_row_count = max(0, int(similar_row_count))
        self.column_top_values = max(1, int(column_top_values))
        self.max_value_chars = max(16, int(max_value_chars))
        self.columns = tuple(str(column) for column in self.dirty.columns)
        self.known_error_coordinates = {
            (int(cell.row), int(cell.col)) for cell in self.known_error_cells
        }
        self.error_rows_by_col: dict[int, set[int]] = defaultdict(set)
        for row, col in self.known_error_coordinates:
            self.error_rows_by_col[col].add(row)
        self._column_profile_cache: dict[int, dict[str, Any]] = {}
        self._value_index_cache: dict[int, dict[str, tuple[int, ...]]] = {}
        self._target_cache: dict[str, dict[str, Any]] = {}

    def _value(self, row: int, col: int) -> str:
        return _bounded_text(self.dirty.iloc[int(row), int(col)], self.max_value_chars)

    def _masked_row(self, target: SafeCell) -> dict[str, str]:
        values: dict[str, str] = {}
        for col, column in enumerate(self.columns):
            coordinate = (int(target.row), col)
            if coordinate == (int(target.row), int(target.col)):
                values[column] = MASKED_TARGET
            elif coordinate in self.known_error_coordinates:
                values[column] = MASKED_OTHER
            else:
                values[column] = self._value(int(target.row), col)
        return values

    def _column_profile(self, col: int) -> dict[str, Any]:
        cached = self._column_profile_cache.get(int(col))
        if cached is not None:
            return dict(cached)
        excluded = self.error_rows_by_col.get(int(col), set())
        values = [
            self._value(row, int(col))
            for row in range(int(self.dirty.shape[0]))
            if row not in excluded
        ]
        nonempty = [value for value in values if value]
        counts = Counter(nonempty)
        numeric_count = 0
        for value in nonempty:
            try:
                float(value.replace(",", ""))
                numeric_count += 1
            except ValueError:
                pass
        profile = {
            "column": self.columns[int(col)],
            "rows_examined": len(values),
            "nonempty_count": len(nonempty),
            "unique_nonempty_count": len(counts),
            "missing_fraction": round(1.0 - len(nonempty) / len(values), 6) if values else 0.0,
            "numeric_fraction": round(numeric_count / len(nonempty), 6) if nonempty else 0.0,
            "top_values": [
                {"value": value, "count": count}
                for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
                    : self.column_top_values
                ]
            ],
        }
        self._column_profile_cache[int(col)] = profile
        return dict(profile)

    def _value_index(self, col: int) -> dict[str, tuple[int, ...]]:
        cached = self._value_index_cache.get(int(col))
        if cached is not None:
            return cached
        index: dict[str, list[int]] = defaultdict(list)
        for row in range(int(self.dirty.shape[0])):
            if (row, int(col)) in self.known_error_coordinates:
                continue
            index[self._value(row, int(col))].append(row)
        frozen = {key: tuple(rows) for key, rows in index.items()}
        self._value_index_cache[int(col)] = frozen
        return frozen

    def _similar_rows(self, target: SafeCell) -> list[dict[str, Any]]:
        if self.similar_row_count == 0:
            return []
        usable_cols = [
            col
            for col in range(len(self.columns))
            if col != int(target.col)
            and (int(target.row), col) not in self.known_error_coordinates
        ]
        scores: Counter[int] = Counter()
        for col in usable_cols:
            value = self._value(int(target.row), col)
            for row in self._value_index(col).get(value, ()):
                if row != int(target.row) and row not in self.error_rows_by_col.get(int(target.col), set()):
                    scores[row] += 1
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        rows: list[dict[str, Any]] = []
        for row, exact_matches in ranked[: self.similar_row_count]:
            values = {
                column: (
                    MASKED_OTHER
                    if (row, col) in self.known_error_coordinates
                    else self._value(row, col)
                )
                for col, column in enumerate(self.columns)
            }
            rows.append(
                {
                    "row_index": row,
                    "exact_context_matches": exact_matches,
                    "values": values,
                }
            )
        return rows

    def _target_payload(self, cell: SafeCell) -> dict[str, Any]:
        cached = self._target_cache.get(cell.cell_id)
        if cached is not None:
            return cached
        payload = {
            "cell_id": cell.cell_id,
            "target": {
                "row_index": int(cell.row),
                "column_index": int(cell.col),
                "column": str(cell.column),
                "dirty_value": _bounded_text(cell.dirty_value, self.max_value_chars),
            },
            "masked_dirty_row": self._masked_row(cell),
            "column_profile": self._column_profile(int(cell.col)),
            "similar_dirty_rows": self._similar_rows(cell),
        }
        self._target_cache[cell.cell_id] = payload
        return payload

    @staticmethod
    def _component_context(component: Any) -> dict[str, Any]:
        return {
            "component_id": str(getattr(component, "component_id", "")),
            "rule_id": str(getattr(component, "rule_id", "")),
            "cell_ids": sorted(str(value) for value in getattr(component, "cell_ids", ())),
            "row_indices": sorted(int(value) for value in getattr(component, "row_indices", ())),
        }

    def _relevant_fd_components(self, cell_ids: set[str]) -> list[dict[str, Any]]:
        relevant: list[dict[str, Any]] = []
        for component in self.fd_components:
            members = {str(value) for value in getattr(component, "cell_ids", ())}
            if members.intersection(cell_ids):
                relevant.append(self._component_context(component))
        return sorted(relevant, key=lambda item: (item["component_id"], item["rule_id"]))

    def payload(
        self,
        query_id: str,
        group_view: str,
        group_cells: Sequence[SafeCell],
    ) -> dict[str, Any]:
        ordered = tuple(sorted(group_cells, key=lambda cell: cell.cell_id))
        return self._payload_from_ordered(query_id, group_view, ordered)

    def _payload_from_ordered(
        self,
        query_id: str,
        group_view: str,
        ordered: Sequence[SafeCell],
    ) -> dict[str, Any]:
        ordered = tuple(ordered)
        if not ordered:
            raise ValueError("group_cells must not be empty")
        if any(cell.cell_id not in self.cell_by_id for cell in ordered):
            raise ValueError("every group cell must come from this builder")
        cell_ids = {cell.cell_id for cell in ordered}
        payload = {
            "query_id": str(query_id),
            "task": "propose independent repairs from dirty-table evidence",
            "dataset": {
                "suite": str(ordered[0].suite),
                "name": str(ordered[0].dataset),
                "columns": list(self.columns),
            },
            "group": {
                "view": str(group_view),
                "size": len(ordered),
                "cell_ids": [cell.cell_id for cell in ordered],
            },
            "public_fd_components": self._relevant_fd_components(cell_ids),
            "targets": [self._target_payload(cell) for cell in ordered],
            "instructions": [
                "Return one independent repairs item for every target when possible.",
                "Copy query_id and cell_id values exactly.",
                "Use only supplied dirty-table and public FD evidence.",
                "Choose abstain when evidence for a replacement is weak.",
            ],
        }
        assert_payload_safe(payload)
        return payload

    def _resolve_ordered_cells(
        self,
        ordered_cell_ids: Sequence[str | SafeCell],
    ) -> tuple[SafeCell, ...]:
        identifiers = tuple(
            str(value.cell_id) if isinstance(value, SafeCell) else str(value)
            for value in ordered_cell_ids
        )
        if not identifiers:
            raise ValueError("ordered_cell_ids must not be empty")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("ordered_cell_ids must be unique")
        unknown = tuple(identifier for identifier in identifiers if identifier not in self.cell_by_id)
        if unknown:
            raise ValueError(
                "every ordered cell must come from this builder: " + ",".join(unknown)
            )
        # Always resolve through the builder so a caller cannot smuggle a
        # same-ID SafeCell carrying different values into prompt material.
        return tuple(self.cell_by_id[identifier] for identifier in identifiers)

    def ordered_payload(
        self,
        query_id: str,
        ordered_cell_ids: Sequence[str | SafeCell],
        *,
        group_view: str = "matched_multi_target",
    ) -> dict[str, Any]:
        """Build a neutral evidence payload without reordering its targets."""

        if str(group_view) != "matched_multi_target":
            raise ValueError("ordered evidence prompts require group_view='matched_multi_target'")
        ordered = self._resolve_ordered_cells(ordered_cell_ids)
        return self._payload_from_ordered(query_id, group_view, ordered)

    def messages(
        self,
        query_id: str,
        group_view: str,
        group_cells: Sequence[SafeCell],
    ) -> CanonicalMessages:
        user_payload = json.dumps(
            self.payload(query_id, group_view, group_cells),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        messages = canonical_messages(
            (
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            )
        )
        assert_messages_safe(messages)
        return messages

    def ordered_messages(
        self,
        query_id: str,
        ordered_cell_ids: Sequence[str | SafeCell],
        *,
        group_view: str = "matched_multi_target",
    ) -> CanonicalMessages:
        """Build neutral messages whose target arrays preserve frozen order."""

        user_payload = json.dumps(
            self.ordered_payload(
                query_id,
                ordered_cell_ids,
                group_view=group_view,
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        messages = canonical_messages(
            (
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            )
        )
        assert_messages_safe(messages)
        return messages

    def build_material(
        self,
        query_id: str,
        group_view: str,
        group_cells: Sequence[SafeCell],
        *,
        prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
        call_overhead_tokens: int = 0,
    ) -> PromptMaterial:
        messages = self.messages(query_id, group_view, group_cells)
        ceiling = completion_token_ceiling(len(group_cells))
        prompt_tokens = estimate_prompt_tokens(messages)
        overhead = max(0, int(call_overhead_tokens))
        return PromptMaterial(
            messages=messages,
            prompt_hash=compute_prompt_hash(
                messages,
                ceiling,
                prompt_schema_version=prompt_schema_version,
                information_policy=INFORMATION_POLICY,
            ),
            estimated_prompt_tokens=prompt_tokens,
            completion_token_ceiling=ceiling,
            estimated_total_tokens=prompt_tokens + ceiling + overhead,
        )

    def build_ordered_material(
        self,
        query_id: str,
        ordered_cell_ids: Sequence[str | SafeCell],
        *,
        group_view: str = "matched_multi_target",
        prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
        call_overhead_tokens: int = 0,
    ) -> PromptMaterial:
        """Build evidence-only material while preserving frozen member order.

        The production ``build_material`` API remains sorting and therefore
        unchanged.  This method is reserved for matched evidence experiments.
        """

        ordered = self._resolve_ordered_cells(ordered_cell_ids)
        messages = self.ordered_messages(
            query_id,
            ordered,
            group_view=group_view,
        )
        ceiling = completion_token_ceiling(len(ordered))
        prompt_tokens = estimate_prompt_tokens(messages)
        overhead = max(0, int(call_overhead_tokens))
        return PromptMaterial(
            messages=messages,
            prompt_hash=compute_prompt_hash(
                messages,
                ceiling,
                prompt_schema_version=prompt_schema_version,
                information_policy=INFORMATION_POLICY,
            ),
            estimated_prompt_tokens=prompt_tokens,
            completion_token_ceiling=ceiling,
            estimated_total_tokens=prompt_tokens + ceiling + overhead,
        )


__all__ = [
    "CanonicalMessages",
    "GroupContextBuilder",
    "MASKED_OTHER",
    "MASKED_TARGET",
    "PROMPT_SCHEMA_VERSION",
    "PromptMaterial",
    "canonical_messages",
    "completion_token_ceiling",
    "compute_prompt_hash",
    "compute_query_id",
    "compute_ordered_query_id",
    "estimate_prompt_tokens",
    "messages_as_dicts",
]
