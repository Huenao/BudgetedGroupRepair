"""Deterministic, label-blind sampling for the preliminary experiments."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from .data import SafeCell


SELECTED_SOURCE_DATASETS = ("hospital", "flights", "beers", "rayyan", "movies_1")
SELECTED_TABLEEG_DATASETS = ("company", "marketing", "restaurant_20", "soccer")
SELECTED_DATASETS = tuple(("source", name) for name in SELECTED_SOURCE_DATASETS) + tuple(
    ("tableeg", name) for name in SELECTED_TABLEEG_DATASETS
)
EXPECTED_SELECTED_ORACLE_ERRORS = 22_198


def stable_digest(seed: int, *parts: object) -> str:
    material = "|".join([str(int(seed)), *(str(part) for part in parts)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SampleRecord:
    suite: str
    dataset: str
    cell_id: str
    row: int
    col: int
    row_id: str
    column: str
    dirty_value: str
    sample_seed: int
    sample_stratum: str
    sample_rank: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _hamilton_quotas(cells: Sequence[SafeCell], sample_n: int, seed: int) -> dict[str, int]:
    counts = Counter(str(cell.column) for cell in cells)
    if sample_n < 0 or sample_n > len(cells):
        raise ValueError("sample_n must be between zero and the dataset cell count")
    if sample_n == len(cells):
        return dict(counts)
    exact = {column: sample_n * count / len(cells) for column, count in counts.items()}
    quotas = {column: min(counts[column], math.floor(value)) for column, value in exact.items()}
    remaining = sample_n - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda column: (
            -(exact[column] - math.floor(exact[column])),
            stable_digest(seed, "quota", column),
        ),
    )
    while remaining:
        progressed = False
        for column in order:
            if quotas[column] >= counts[column]:
                continue
            quotas[column] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("unable to allocate deterministic sample quotas")
    return quotas


def sample_dataset_cells(
    cells: Sequence[SafeCell],
    *,
    sample_n: int,
    seed: int,
) -> tuple[SampleRecord, ...]:
    ordered = tuple(cells)
    identities = {(cell.suite, cell.dataset) for cell in ordered}
    if len(identities) != 1:
        raise ValueError("sampling input must contain exactly one dataset")
    quotas = _hamilton_quotas(ordered, int(sample_n), int(seed))
    by_column: dict[str, list[SafeCell]] = defaultdict(list)
    for cell in ordered:
        by_column[str(cell.column)].append(cell)
    selected: list[SampleRecord] = []
    for column in sorted(by_column):
        ranked = sorted(
            by_column[column], key=lambda cell: stable_digest(seed, str(cell.cell_id))
        )
        for rank, cell in enumerate(ranked[: quotas[column]], start=1):
            selected.append(
                SampleRecord(
                    suite=str(cell.suite),
                    dataset=str(cell.dataset),
                    cell_id=str(cell.cell_id),
                    row=int(cell.row),
                    col=int(cell.col),
                    row_id=str(cell.row_id),
                    column=str(cell.column),
                    dirty_value=str(cell.dirty_value),
                    sample_seed=int(seed),
                    sample_stratum=str(column),
                    sample_rank=rank,
                )
            )
    result = tuple(sorted(selected, key=lambda row: row.cell_id))
    if len(result) != int(sample_n) or len({row.cell_id for row in result}) != len(result):
        raise RuntimeError("sample must contain the requested number of unique cells")
    return result


def build_sample_manifest(
    cells_by_dataset: Mapping[tuple[str, str], Sequence[SafeCell]],
    *,
    mode: str,
    sample_n_per_dataset: int,
    seed: int,
) -> tuple[SampleRecord, ...]:
    if set(cells_by_dataset) != set(SELECTED_DATASETS):
        raise ValueError("sample input must contain exactly the frozen 5+4 datasets")
    records: list[SampleRecord] = []
    for key in SELECTED_DATASETS:
        cells = tuple(cells_by_dataset[key])
        count = len(cells) if mode == "full_selected_datasets" else int(sample_n_per_dataset)
        if mode not in {"full_selected_datasets", "stratified_fixed_n_per_dataset"}:
            raise ValueError(f"unsupported sample mode: {mode}")
        records.extend(sample_dataset_cells(cells, sample_n=count, seed=seed))
    return tuple(records)


def records_to_cells(
    records: Iterable[SampleRecord],
    safe_cells: Mapping[str, SafeCell],
) -> tuple[SafeCell, ...]:
    cells = tuple(safe_cells[record.cell_id] for record in records)
    if len(cells) != len({str(cell.cell_id) for cell in cells}):
        raise ValueError("sample manifest contains duplicate cell IDs")
    return cells


__all__ = [
    "EXPECTED_SELECTED_ORACLE_ERRORS",
    "SELECTED_DATASETS",
    "SELECTED_SOURCE_DATASETS",
    "SELECTED_TABLEEG_DATASETS",
    "SampleRecord",
    "build_sample_manifest",
    "records_to_cells",
    "sample_dataset_cells",
    "stable_digest",
]
