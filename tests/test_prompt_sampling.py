from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from budgeted_group_repair_no_baran.data import SafeCell, SafeDataset
from budgeted_group_repair_no_baran.group_context import (
    GroupContextBuilder,
    compute_query_id,
)
from budgeted_group_repair_no_baran.prompt_policy import (
    INFORMATION_POLICY,
    PromptPolicyError,
    assert_messages_safe,
    assert_payload_safe,
)
from budgeted_group_repair_no_baran.sampling import sample_dataset_cells


def _dataset() -> tuple[SafeDataset, tuple[SafeCell, ...]]:
    dirty = pd.DataFrame(
        {
            "row_id": [str(index) for index in range(10)],
            "name": [f"dirty-{index}" for index in range(10)],
            "country": ["US"] * 5 + ["GB"] * 5,
        }
    )
    cells = tuple(
        SafeCell("source", "toy", row, 1 if row < 7 else 2, "name" if row < 7 else "country", str(row), str(dirty.iloc[row, 1 if row < 7 else 2]))
        for row in range(10)
    )
    return SafeDataset("source", "toy", Path("dirty.csv"), dirty, cells), cells


def test_safe_cell_physically_excludes_oracle_fields() -> None:
    names = {field.name for field in fields(SafeCell)}
    assert "clean_value" not in names
    assert "error_type" not in names


def test_hamilton_sample_is_deterministic_and_column_stratified() -> None:
    _, cells = _dataset()
    first = sample_dataset_cells(cells, sample_n=5, seed=42)
    second = sample_dataset_cells(tuple(reversed(cells)), sample_n=5, seed=42)
    assert first == second
    assert len(first) == 5
    assert {row.column for row in first} == {"name", "country"}
    assert all(not hasattr(row, "clean_value") for row in first)


def test_prompt_ignores_external_features_and_passes_recursive_audit() -> None:
    dataset, cells = _dataset()
    query_id = compute_query_id(
        "source", "toy", "singleton", (str(cells[0].cell_id),), arm="singleton"
    )
    material = GroupContextBuilder(
        dataset,
        cells,
        {str(cells[0].cell_id): {"baran_prediction": "POISON"}},
        similar_row_count=1,
    ).build_material(query_id, "singleton", (cells[0],))
    assert_messages_safe(material.messages)
    serialized = str(material.messages).lower()
    assert "baran" not in serialized
    assert "poison" not in serialized


def test_information_policy_and_arm_bind_query_identity() -> None:
    base = compute_query_id("source", "toy", "row", ("c1", "c2"), arm="structured")
    random = compute_query_id("source", "toy", "row", ("c1", "c2"), arm="random")
    revised = compute_query_id(
        "source",
        "toy",
        "row",
        ("c1", "c2"),
        arm="structured",
        information_policy=INFORMATION_POLICY + "-revision",
    )
    assert len({base, random, revised}) == 3


def test_forbidden_nested_field_and_natural_language_are_rejected() -> None:
    with pytest.raises(PromptPolicyError):
        assert_payload_safe({"safe": [{"clean_value": "x"}]})
    with pytest.raises(PromptPolicyError):
        assert_payload_safe({"note": "candidate support is high"})
