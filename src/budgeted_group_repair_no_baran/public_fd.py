"""Dirty-only public functional-dependency grouping support.

Rules come exclusively from ``configs/public_fds.json``.  Components are
derived from the dirty table and known error coordinates; TableEG generation
logs are neither accepted nor read by this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .data import SafeCell, normalize_value


@dataclass(frozen=True, slots=True)
class PublicFD:
    rule_id: str
    determinant: tuple[str, ...]
    dependent: str

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "determinant": list(self.determinant),
            "dependent": self.dependent,
        }


@dataclass(frozen=True, slots=True)
class FDViolationComponent:
    """A connected dirty-table violation region for one declared rule."""

    component_id: str
    suite: str
    dataset: str
    rule_id: str
    cell_ids: tuple[str, ...]
    row_indices: tuple[int, ...]
    violation_pair_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _parse_rule(raw: object, location: str) -> PublicFD:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{location}: rule must be an object")
    rule_id = str(raw.get("rule_id", "")).strip()
    determinant_raw = raw.get("determinant")
    dependent = str(raw.get("dependent", "")).strip()
    if not rule_id or not dependent:
        raise ValueError(f"{location}: rule_id and dependent are required")
    if not isinstance(determinant_raw, list) or not determinant_raw:
        raise ValueError(f"{location}: determinant must be a non-empty list")
    determinant = tuple(str(column).strip() for column in determinant_raw)
    if any(not column for column in determinant) or len(set(determinant)) != len(determinant):
        raise ValueError(f"{location}: determinant columns must be non-empty and unique")
    if dependent in determinant:
        raise ValueError(f"{location}: dependent cannot also be a determinant column")
    return PublicFD(rule_id=rule_id, determinant=determinant, dependent=dependent)


def load_public_fds(path: str | Path) -> dict[tuple[str, str], tuple[PublicFD, ...]]:
    """Load and validate the public rule registry without touching data logs."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("public FD config must be a JSON object")
    registry = raw.get("functional_dependencies", raw)
    if not isinstance(registry, Mapping):
        raise ValueError("functional_dependencies must be an object")
    result: dict[tuple[str, str], tuple[PublicFD, ...]] = {}
    for raw_key, raw_rules in sorted(registry.items(), key=lambda item: str(item[0])):
        key = str(raw_key)
        parts = key.split("/", 1)
        if len(parts) != 2 or parts[0] not in {"source", "tableeg"} or not parts[1]:
            raise ValueError(f"invalid public FD dataset key: {key!r}")
        if not isinstance(raw_rules, list):
            raise ValueError(f"{key}: rules must be a list")
        rules = tuple(_parse_rule(rule, f"{key}[{index}]") for index, rule in enumerate(raw_rules))
        rule_ids = [rule.rule_id for rule in rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError(f"{key}: duplicate rule IDs")
        result[(parts[0], parts[1])] = rules
    return result


def fds_for_dataset(
    registry: Mapping[tuple[str, str], Sequence[PublicFD]],
    suite: str,
    dataset: str,
) -> tuple[PublicFD, ...]:
    return tuple(registry.get((suite, dataset), ()))


def validate_rule_columns(dirty: pd.DataFrame, fds: Sequence[PublicFD]) -> None:
    columns = {str(column) for column in dirty.columns}
    for rule in fds:
        missing = (set(rule.determinant) | {rule.dependent}) - columns
        if missing:
            raise ValueError(f"public FD {rule.rule_id!r} references missing columns: {sorted(missing)}")


def _component_digest(
    suite: str,
    dataset: str,
    rule_id: str,
    determinant_key: tuple[str, ...],
) -> str:
    material = "\x1f".join((suite, dataset, rule_id, *determinant_key)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:24]


def build_fd_violation_components(
    dirty: pd.DataFrame,
    suite: str,
    dataset: str,
    error_cells: Sequence[SafeCell],
    fds: Sequence[PublicFD],
) -> list[FDViolationComponent]:
    """Construct stable connected violation regions from dirty values only.

    For each determinant-value group whose dependent column has more than one
    value, the conflicting rows form a connected complete multipartite graph.
    Only known error coordinates in the rule's determinant/dependent columns
    are exposed as candidate members.
    """

    if not all(isinstance(cell, SafeCell) for cell in error_cells):
        raise TypeError("public FD construction accepts SafeCell inputs only")
    if any(cell.suite != suite or cell.dataset != dataset for cell in error_cells):
        raise ValueError("all cells must belong to the supplied dataset")
    validate_rule_columns(dirty, fds)
    cells_by_row: dict[int, list[SafeCell]] = {}
    for cell in error_cells:
        if cell.row < 0 or cell.row >= len(dirty):
            raise ValueError(f"cell row is outside dirty table: {cell.cell_id}")
        cells_by_row.setdefault(cell.row, []).append(cell)

    result: list[FDViolationComponent] = []
    for rule in sorted(fds, key=lambda value: value.rule_id):
        groups: dict[tuple[str, ...], list[int]] = {}
        for row_index in range(len(dirty)):
            determinant_key = tuple(
                normalize_value(dirty.at[row_index, column]) for column in rule.determinant
            )
            groups.setdefault(determinant_key, []).append(row_index)
        relevant_columns = set(rule.determinant) | {rule.dependent}
        for determinant_key, rows in sorted(groups.items(), key=lambda item: item[0]):
            if len(rows) < 2:
                continue
            dependent_counts: dict[str, int] = {}
            for row_index in rows:
                value = normalize_value(dirty.at[row_index, rule.dependent])
                dependent_counts[value] = dependent_counts.get(value, 0) + 1
            if len(dependent_counts) < 2:
                continue
            cell_ids = tuple(
                sorted(
                    str(cell.cell_id)
                    for row_index in rows
                    for cell in cells_by_row.get(row_index, ())
                    if cell.column in relevant_columns
                )
            )
            if not cell_ids:
                continue
            row_count = len(rows)
            same_value_pairs = sum(count * (count - 1) // 2 for count in dependent_counts.values())
            all_pairs = row_count * (row_count - 1) // 2
            digest = _component_digest(suite, dataset, rule.rule_id, determinant_key)
            result.append(
                FDViolationComponent(
                    component_id=f"{suite}:{dataset}:{rule.rule_id}:{digest}",
                    suite=suite,
                    dataset=dataset,
                    rule_id=rule.rule_id,
                    cell_ids=cell_ids,
                    row_indices=tuple(sorted(rows)),
                    violation_pair_count=all_pairs - same_value_pairs,
                )
            )
    result.sort(key=lambda component: component.component_id)
    return result


__all__ = [
    "FDViolationComponent",
    "PublicFD",
    "build_fd_violation_components",
    "fds_for_dataset",
    "load_public_fds",
    "validate_rule_columns",
]
